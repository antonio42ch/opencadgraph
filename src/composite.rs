//! Composite nodes: a graph of operations used as one node.
//!
//! A composite carries its own inner graph and names which inner input
//! ports it exposes as its inputs and which inner outputs as its outputs.
//! Evaluating it sets the exposed inputs on a copy of the inner graph,
//! evaluates that, and reads the exposed outputs back, so a composite takes
//! whole values (lists included) and its inner nodes lace as they would
//! anywhere else.
//!
//! Inner graphs hold operations and other composites only: a host object
//! inside one would have no identity of its own across the composite's
//! uses, so [`Composite::new`] refuses them.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::{Graph, Kind, Link, NodeId, Port};

#[derive(Clone, Debug)]
pub struct Composite {
    pub name: String,
    /// Exposed inputs, each an inner op input that nothing inside feeds.
    pub inputs: Vec<(String, Port)>,
    /// Exposed outputs, each an inner node's output.
    pub outputs: Vec<(String, Port)>,
    pub graph: Graph,
    /// What the host keeps with the composite (names, layout); the engine
    /// only carries it.
    pub meta: Value,
}

impl Composite {
    /// Whether every step inside computes from its inputs alone.
    pub fn is_pure(&self) -> bool {
        self.graph.nodes.iter().all(|node| match &node.kind {
            crate::Kind::Op(spec) => crate::ops::is_pure(spec.name),
            crate::Kind::Composite(inner) => inner.is_pure(),
            crate::Kind::Script(_) => true,
            crate::Kind::Object(_) => false,
        })
    }

    /// Checks that the ports exist and the inner graph holds no host
    /// objects.
    pub fn new(
        name: String,
        graph: Graph,
        inputs: Vec<(String, Port)>,
        outputs: Vec<(String, Port)>,
        meta: Value,
    ) -> Result<Self, String> {
        if let Some(node) = graph
            .nodes
            .iter()
            .find(|node| matches!(node.kind, Kind::Object(_)))
        {
            return Err(format!(
                "node {} draws objects; a composite holds operations only",
                node.id
            ));
        }
        for (name, port) in &inputs {
            let takes = graph
                .node(port.node)
                .is_some_and(|node| node.params.contains_key(&port.field));
            if !takes || graph.is_linked(port) {
                return Err(format!("input `{name}` is not a free input inside"));
            }
        }
        for (name, port) in &outputs {
            if graph.node(port.node).is_none() {
                return Err(format!("output `{name}` reads a node that is not inside"));
            }
        }
        Ok(Self {
            name,
            inputs,
            outputs,
            graph,
            meta,
        })
    }

    /// The exposed inputs' values while nothing feeds them: the inner
    /// parameters they stand for.
    pub fn defaults(&self) -> BTreeMap<String, Value> {
        self.inputs
            .iter()
            .map(|(name, port)| (name.clone(), self.graph.input(port)))
            .collect()
    }

    pub fn evaluate(
        &self,
        inputs: &BTreeMap<String, Value>,
        host: &mut dyn crate::Host,
    ) -> BTreeMap<String, Value> {
        self.evaluate_checked(inputs, host).0
    }

    /// [`evaluate`](Self::evaluate), with the issues of the steps inside.
    pub fn evaluate_checked(
        &self,
        inputs: &BTreeMap<String, Value>,
        host: &mut dyn crate::Host,
    ) -> (BTreeMap<String, Value>, Vec<crate::Issue>) {
        let mut inner = self.graph.clone();
        for (name, port) in &self.inputs {
            if let (Some(value), Some(node)) = (inputs.get(name), inner.node_mut(port.node)) {
                node.params.insert(port.field.clone(), value.clone());
            }
        }
        inner.evaluate(host);
        let issues = inner
            .nodes
            .iter()
            .flat_map(|node| {
                let step = match &node.kind {
                    Kind::Op(spec) => spec.name.to_owned(),
                    Kind::Composite(composite) => composite.name.clone(),
                    Kind::Script(script) => script.name.clone(),
                    Kind::Object(name) => name.clone(),
                };
                node.issues
                    .iter()
                    .map(move |issue| crate::Issue::Inner(step.clone(), Box::new(issue.clone())))
            })
            .take(crate::MAX_ISSUES)
            .collect();
        let outputs = self
            .outputs
            .iter()
            .map(|(name, port)| {
                let value = inner
                    .node(port.node)
                    .and_then(|node| node.outputs.get(&port.field))
                    .cloned()
                    .unwrap_or(Value::Null);
                (name.clone(), value)
            })
            .collect();
        (outputs, issues)
    }

    pub fn to_json(&self) -> Value {
        let ports = |ports: &[(String, Port)]| {
            ports
                .iter()
                .map(|(name, port)| json!({ "name": name, "node": port.node, "field": port.field }))
                .collect::<Vec<_>>()
        };
        json!({
            "name": self.name,
            "inputs": ports(&self.inputs),
            "outputs": ports(&self.outputs),
            "graph": self.graph.to_json(),
            "meta": self.meta,
        })
    }

    pub fn from_json(value: &Value) -> Result<Self, String> {
        let ports = |key: &str| -> Result<Vec<(String, Port)>, String> {
            value[key]
                .as_array()
                .ok_or_else(|| format!("composite has no {key}"))?
                .iter()
                .map(|port| {
                    Some((
                        port["name"].as_str()?.to_owned(),
                        Port {
                            node: NodeId::try_from(port["node"].as_u64()?).ok()?,
                            field: port["field"].as_str()?.to_owned(),
                        },
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| format!("composite has a malformed {key} entry"))
        };
        Self::new(
            value["name"].as_str().unwrap_or("Composite").to_owned(),
            Graph::from_json(&value["graph"])?,
            ports("inputs")?,
            ports("outputs")?,
            value["meta"].clone(),
        )
    }
}

impl Graph {
    /// Replaces `members` with one composite node holding them, where the
    /// last of them was. Links from outside into the members become its
    /// inputs, links from the members to outside its outputs (or, when
    /// nothing outside reads them, every output of the last member).
    /// `port_name` names an exposed port after the inner port it stands for.
    ///
    /// Refused when a member draws objects, or when a node outside both
    /// reads from the members and feeds them, which no single node could
    /// sit in the middle of.
    pub fn extract(
        &mut self,
        members: &[NodeId],
        name: &str,
        port_name: &dyn Fn(&Port) -> String,
        meta: Value,
    ) -> Result<NodeId, String> {
        let inside = |id: NodeId| members.contains(&id);
        if members.is_empty() || members.iter().any(|id| self.node(*id).is_none()) {
            return Err("no such nodes".to_owned());
        }
        for node in self.nodes.iter().filter(|node| !inside(node.id)) {
            let fed = members.iter().any(|m| self.reaches(*m, node.id));
            let feeds = members.iter().any(|m| self.reaches(node.id, *m));
            if fed && feeds {
                return Err(format!("node {} sits between the members", node.id));
            }
        }

        // The inner graph, in the members' order.
        let mut inner = Graph::default();
        let mut ids = BTreeMap::new();
        for node in self.nodes.iter().filter(|node| inside(node.id)) {
            if matches!(node.kind, Kind::Object(_)) {
                return Err(format!(
                    "node {} draws objects; a composite holds operations only",
                    node.id
                ));
            }
            let id = inner.add(node.kind.clone());
            if let Some(copy) = inner.node_mut(id) {
                copy.params = node.params.clone();
            }
            ids.insert(node.id, id);
        }
        let map = |port: &Port| Port {
            node: ids[&port.node],
            field: port.field.clone(),
        };
        let unique = |taken: &[(String, Port)], wanted: String| -> String {
            (1..)
                .map(|n| {
                    if n == 1 {
                        wanted.clone()
                    } else {
                        format!("{wanted} {n}")
                    }
                })
                .find(|name| !taken.iter().any(|(other, _)| other == name))
                .unwrap_or(wanted)
        };
        let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
        let (mut feeding, mut reading): (Vec<Link>, Vec<Link>) = (Vec::new(), Vec::new());
        for link in &self.links {
            match (inside(link.from.node), inside(link.to.node)) {
                (true, true) => {
                    inner.connect(map(&link.from), map(&link.to));
                }
                (false, true) => feeding.push(link.clone()),
                (true, false) => reading.push(link.clone()),
                (false, false) => {}
            }
        }
        // One composite input per fed inner port, one output per read one.
        let mut input_of: Vec<(Port, String)> = Vec::new();
        for link in &feeding {
            if !input_of.iter().any(|(port, _)| *port == link.to) {
                let name = unique(&inputs, port_name(&link.to));
                inputs.push((name.clone(), map(&link.to)));
                input_of.push((link.to.clone(), name));
            }
        }
        let mut output_of: Vec<(Port, String)> = Vec::new();
        for link in &reading {
            if !output_of.iter().any(|(port, _)| *port == link.from) {
                let name = unique(&outputs, port_name(&link.from));
                outputs.push((name.clone(), map(&link.from)));
                output_of.push((link.from.clone(), name));
            }
        }
        if outputs.is_empty() {
            let last = *members
                .iter()
                .max_by_key(|id| self.nodes.iter().position(|node| node.id == **id))
                .expect("members are not empty");
            for field in self.output_names(last) {
                let port = Port { node: last, field };
                let name = unique(&outputs, port_name(&port));
                outputs.push((name, map(&port)));
            }
        }

        // Inner inputs fed from outside now take the composite's values.
        let mut inner_free = inner.clone();
        for (_, port) in &inputs {
            inner_free.links.retain(|link| &link.to != port);
        }
        let composite = Composite::new(name.to_owned(), inner_free, inputs, outputs, meta)?;

        let at = self
            .nodes
            .iter()
            .rposition(|node| inside(node.id))
            .expect("members exist");
        let id = self.add(Kind::Composite(Arc::new(composite)));
        let node = self.nodes.pop().expect("just added");
        self.nodes.insert(at + 1, node);
        for member in members {
            self.remove(*member);
        }
        for link in feeding {
            let name = &input_of
                .iter()
                .find(|(port, _)| *port == link.to)
                .expect("named")
                .1;
            self.connect(
                link.from,
                Port {
                    node: id,
                    field: name.clone(),
                },
            );
        }
        for link in reading {
            let name = &output_of
                .iter()
                .find(|(port, _)| *port == link.from)
                .expect("named")
                .1;
            self.connect(
                Port {
                    node: id,
                    field: name.clone(),
                },
                link.to,
            );
        }
        Ok(id)
    }

    /// Replaces composite `id` with the nodes inside it, where it was, wired
    /// as it was. Its own values for unlinked inputs become the inner
    /// parameters. Returns the new nodes' ids, keyed by their inner ids.
    pub fn expand(&mut self, id: NodeId) -> Result<BTreeMap<NodeId, NodeId>, String> {
        let Some(Kind::Composite(composite)) = self.node(id).map(|node| node.kind.clone()) else {
            return Err(format!("node {id} is not a composite"));
        };
        let at = self
            .nodes
            .iter()
            .position(|node| node.id == id)
            .expect("exists");
        let mut ids = BTreeMap::new();
        let mut added = Vec::new();
        for node in &composite.graph.nodes {
            let new = self.add(node.kind.clone());
            let mut copy = self.nodes.pop().expect("just added");
            copy.params = node.params.clone();
            added.push(copy);
            ids.insert(node.id, new);
        }
        let own = self
            .node(id)
            .map(|node| node.params.clone())
            .unwrap_or_default();
        for (name, port) in &composite.inputs {
            if let (Some(value), Some(node)) = (
                own.get(name),
                added.iter_mut().find(|n| n.id == ids[&port.node]),
            ) {
                node.params.insert(port.field.clone(), value.clone());
            }
        }
        for (k, node) in added.into_iter().enumerate() {
            self.nodes.insert(at + 1 + k, node);
        }
        let map = |port: &Port| Port {
            node: ids[&port.node],
            field: port.field.clone(),
        };
        for link in &composite.graph.links {
            self.links.push(Link {
                from: map(&link.from),
                to: map(&link.to),
            });
        }
        let outside: Vec<Link> = self
            .links
            .iter()
            .filter(|link| link.from.node == id || link.to.node == id)
            .cloned()
            .collect();
        self.remove(id);
        for link in outside {
            if link.to.node == id {
                if let Some((_, port)) = composite
                    .inputs
                    .iter()
                    .find(|(name, _)| *name == link.to.field)
                {
                    self.connect(link.from, map(port));
                }
            } else if let Some((_, port)) = composite
                .outputs
                .iter()
                .find(|(name, _)| *name == link.from.field)
            {
                self.connect(map(port), link.to);
            }
        }
        Ok(ids)
    }

    /// The outputs node `id` gives.
    pub fn output_names(&self, id: NodeId) -> Vec<String> {
        match self.node(id).map(|node| &node.kind) {
            Some(Kind::Op(spec)) => spec.outputs.iter().map(|name| (*name).to_owned()).collect(),
            Some(Kind::Composite(composite)) => composite
                .outputs
                .iter()
                .map(|(name, _)| name.clone())
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec;

    struct NoDrawing;

    impl crate::Host for NoDrawing {
        fn objects(
            &mut self,
            _: NodeId,
            _: &str,
            _: &[Vec<(String, Value)>],
        ) -> Vec<Vec<(String, Value)>> {
            Vec::new()
        }
        fn curve(&self, _: &Value) -> Option<kernel::space::PlanarCurve> {
            None
        }
        fn expression(&self, _: &str) -> Option<f64> {
            None
        }
    }

    fn port(node: NodeId, field: &str) -> Port {
        Port {
            node,
            field: field.to_owned(),
        }
    }

    #[test]
    fn nodes_say_why_they_give_nothing() {
        let mut graph = Graph::default();
        let line = graph.add(Kind::Op(spec("Line Curve").unwrap()));
        let circle = graph.add(Kind::Op(spec("Circle Curve").unwrap()));
        let count = graph.add(Kind::Op(spec("Sum").unwrap()));
        graph.node_mut(circle).unwrap().params.insert("radius".into(), json!(-1));
        graph.node_mut(circle).unwrap().params.insert("center".into(), json!({"x": 0, "y": 0, "z": 0}));
        graph.node_mut(count).unwrap().params.insert("list".into(), json!([1, 2]));
        graph.evaluate(&mut NoDrawing);
        let issues = |id: NodeId| graph.node(id).unwrap().issues.clone();
        assert_eq!(issues(line), vec![crate::Issue::Missing("start".into())]);
        assert_eq!(issues(circle), vec![crate::Issue::NoResult]);
        assert!(issues(count).is_empty());
        assert_eq!(crate::Issue::Missing("start".into()).message(), "Nothing feeds start");

        // A composite reports the issues of its steps.
        let id = graph
            .extract(&[circle], "Ring", &|port: &Port| port.field.clone(), Value::Null)
            .unwrap();
        graph.evaluate(&mut NoDrawing);
        assert_eq!(
            graph.node(id).unwrap().issues,
            vec![crate::Issue::Inner("Circle Curve".into(), Box::new(crate::Issue::NoResult))]
        );
    }

    /// width → (Add width+1) → (Multiply ×2) → Negate
    fn chain() -> (Graph, [NodeId; 4]) {
        let mut graph = Graph::default();
        let width = graph.add(Kind::Op(spec("Number").unwrap()));
        let add = graph.add(Kind::Op(spec("Add").unwrap()));
        let mul = graph.add(Kind::Op(spec("Multiply").unwrap()));
        let neg = graph.add(Kind::Op(spec("Negate").unwrap()));
        graph
            .node_mut(width)
            .unwrap()
            .params
            .insert("value".into(), json!(4));
        graph
            .node_mut(add)
            .unwrap()
            .params
            .insert("b".into(), json!(1));
        graph
            .node_mut(mul)
            .unwrap()
            .params
            .insert("b".into(), json!(2));
        graph.connect(port(width, "value"), port(add, "a"));
        graph.connect(port(add, "result"), port(mul, "a"));
        graph.connect(port(mul, "result"), port(neg, "x"));
        (graph, [width, add, mul, neg])
    }

    #[test]
    fn an_extracted_composite_computes_what_its_members_did() {
        let (mut graph, [width, add, mul, neg]) = chain();
        graph.evaluate(&mut NoDrawing);
        assert_eq!(graph.node(neg).unwrap().outputs["result"], json!(-10.0));

        let id = graph
            .extract(
                &[add, mul],
                "Grow",
                &|p: &Port| p.field.clone(),
                Value::Null,
            )
            .unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[1].id, id, "where the members were");
        let Kind::Composite(composite) = &graph.node(id).unwrap().kind else {
            panic!()
        };
        assert_eq!(composite.inputs.len(), 1);
        assert_eq!(composite.outputs.len(), 1);

        graph.evaluate(&mut NoDrawing);
        assert_eq!(graph.node(neg).unwrap().outputs["result"], json!(-10.0));
        graph
            .node_mut(width)
            .unwrap()
            .params
            .insert("value".into(), json!(9));
        graph.evaluate(&mut NoDrawing);
        assert_eq!(graph.node(neg).unwrap().outputs["result"], json!(-20.0));

        // It survives a save, and expands back into the same computation.
        let mut loaded = Graph::from_json(&graph.to_json()).unwrap();
        loaded.evaluate(&mut NoDrawing);
        assert_eq!(loaded.node(neg).unwrap().outputs["result"], json!(-20.0));
        let inner = loaded.expand(id).unwrap();
        assert_eq!(inner.len(), 2);
        assert_eq!(loaded.nodes.len(), 4);
        loaded.evaluate(&mut NoDrawing);
        assert_eq!(loaded.node(neg).unwrap().outputs["result"], json!(-20.0));
    }

    #[test]
    fn a_node_between_the_members_is_refused() {
        let (mut graph, [_, add, _, neg]) = chain();
        assert!(graph
            .extract(
                &[add, neg],
                "Split",
                &|p: &Port| p.field.clone(),
                Value::Null
            )
            .is_err());
    }

    #[test]
    fn composites_hold_operations_only() {
        let mut inner = Graph::default();
        inner.add(Kind::Object("Line".into()));
        assert!(Composite::new("Bad".into(), inner, Vec::new(), Vec::new(), Value::Null).is_err());
    }
}
