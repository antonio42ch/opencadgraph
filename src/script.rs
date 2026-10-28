//! Script nodes: a step whose computation is a short Rhai program.
//!
//! A script declares its ports in its first comment lines:
//!
//! ```text
//! // inputs: axis: curve, spacing: number = 2.5, names: text[]
//! // outputs: posts: point[], count: number
//! ```
//!
//! and ends with a map of its outputs (`#{ posts: posts, count: n }`), or the
//! value of its only output. An input without `[]` takes one value: a list
//! arriving there runs the script once per element, as with every other node.
//! Points are maps `#{x, y, z}` and curves the curve values of
//! [`crate::value`]. Scripts compute only; they cannot read files, the
//! drawing or the clock, and they stop after a budget of operations.

use std::sync::Arc;

use serde_json::Value;

use crate::docs::{PortType, Type};

/// A declared port: its name, kind and value while nothing is linked.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptPort {
    pub name: String,
    pub kind: PortType,
    pub default: Value,
}

/// A script node's program and the ports its header declares.
#[derive(Debug)]
pub struct Script {
    pub name: String,
    pub source: String,
    pub inputs: Vec<ScriptPort>,
    pub outputs: Vec<ScriptPort>,
    /// Why the header or the program does not read, with its line.
    pub error: Option<String>,
    #[cfg(feature = "script")]
    ast: Option<rhai::AST>,
}

/// The template a new script node starts from.
pub const TEMPLATE: &str = "// inputs: a: number = 1, b: number = 2
// outputs: sum: number, product: number
#{ sum: a + b, product: a * b }
";

/// Steps a script may take per run before it is stopped.
pub const MAX_OPERATIONS: u64 = 2_000_000;

impl Script {
    /// Reads `source`: its header's ports, and the program compiled.
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Arc<Self> {
        let source = source.into();
        let (inputs, outputs, mut error) = match header(&source) {
            Ok((inputs, outputs)) => (inputs, outputs, None),
            Err(why) => (Vec::new(), Vec::new(), Some(why)),
        };
        #[cfg(feature = "script")]
        let ast = match engine().compile(&source) {
            Ok(ast) => Some(ast),
            Err(why) => {
                error.get_or_insert_with(|| describe(&why.to_string(), why.position()));
                None
            }
        };
        #[cfg(not(feature = "script"))]
        error.get_or_insert_with(|| "scripts are not available in this build".to_owned());
        Arc::new(Script {
            name: name.into(),
            source,
            inputs,
            outputs,
            error,
            #[cfg(feature = "script")]
            ast,
        })
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({ "name": self.name, "source": self.source })
    }

    pub fn from_json(value: &Value) -> Result<Arc<Self>, String> {
        let source = value["source"].as_str().ok_or("a script needs its source")?;
        Ok(Script::new(value["name"].as_str().unwrap_or("Script"), source))
    }

    /// The input defaults, as a new node's parameters.
    pub fn defaults(&self) -> std::collections::BTreeMap<String, Value> {
        self.inputs
            .iter()
            .map(|port| (port.name.clone(), port.default.clone()))
            .collect()
    }

    /// Runs the script over `args`, one per input, lacing lists into the
    /// inputs that take one value. Gives the outputs and, when a run
    /// failed, why.
    pub fn evaluate(&self, args: &[Value]) -> (Vec<Value>, Option<String>) {
        let empty = vec![Value::Null; self.outputs.len()];
        if let Some(why) = &self.error {
            return (empty, Some(why.clone()));
        }
        #[cfg(feature = "script")]
        {
            let Some(ast) = &self.ast else {
                return (empty, None);
            };
            let engine = engine();
            let scalar: Vec<bool> = self.inputs.iter().map(|port| !port.kind.list).collect();
            let mut failure = None;
            let values = crate::value::lace(args, &scalar, self.outputs.len(), &mut |args| {
                match self.run(&engine, ast, args) {
                    Ok(values) => values,
                    Err(why) => {
                        failure.get_or_insert(why);
                        vec![Value::Null; self.outputs.len()]
                    }
                }
            });
            (values, failure)
        }
        #[cfg(not(feature = "script"))]
        {
            let _ = args;
            (empty, None)
        }
    }

    #[cfg(feature = "script")]
    fn run(&self, engine: &rhai::Engine, ast: &rhai::AST, args: &[Value]) -> Result<Vec<Value>, String> {
        let mut scope = rhai::Scope::new();
        for (port, arg) in self.inputs.iter().zip(args) {
            let value = if arg.is_null() { &port.default } else { arg };
            scope.push_dynamic(port.name.clone(), to_dynamic(value));
        }
        let result: rhai::Dynamic = engine
            .eval_ast_with_scope(&mut scope, ast)
            .map_err(|why| describe(&why.to_string(), why.position()))?;
        let result = from_dynamic(&result);
        Ok(match (&result, self.outputs.as_slice()) {
            (Value::Object(map), _) if self.outputs.iter().any(|port| map.contains_key(&port.name)) => self
                .outputs
                .iter()
                .map(|port| map.get(&port.name).cloned().unwrap_or(Value::Null))
                .collect(),
            (_, [_]) => vec![result],
            _ => {
                let names: Vec<&str> = self.outputs.iter().map(|port| port.name.as_str()).collect();
                return Err(format!("the script must end with a map of {}", names.join(", ")));
            }
        })
    }
}

/// An error's text with its line, as `line 4: …`.
#[cfg(feature = "script")]
fn describe(text: &str, position: rhai::Position) -> String {
    // Rhai appends the position itself; the line is said once, first.
    let text = text.split(" (line ").next().unwrap_or(text).trim();
    match position.line() {
        Some(line) => format!("line {line}: {text}"),
        None => text.to_owned(),
    }
}

/// The ports a script's header declares: its `// inputs:` and
/// `// outputs:` comment lines.
fn header(source: &str) -> Result<(Vec<ScriptPort>, Vec<ScriptPort>), String> {
    let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
    for (k, line) in source.lines().enumerate() {
        let Some(comment) = line.trim().strip_prefix("//") else {
            if line.trim().is_empty() {
                continue;
            }
            break;
        };
        let comment = comment.trim();
        let lower = comment.to_lowercase();
        let (is_input, rest) = if lower.starts_with("inputs:") {
            (true, &comment["inputs:".len()..])
        } else if lower.starts_with("outputs:") {
            (false, &comment["outputs:".len()..])
        } else {
            continue;
        };
        for item in split_ports(rest) {
            let port = port(item).map_err(|why| format!("line {}: {why}", k + 1))?;
            if inputs.iter().chain(outputs.iter()).any(|other: &ScriptPort| other.name == port.name) {
                return Err(format!("line {}: `{}` is declared twice", k + 1, port.name));
            }
            if is_input { inputs.push(port) } else { outputs.push(port) }
        }
    }
    if outputs.is_empty() {
        return Err("line 1: declare the outputs, as // outputs: result: number".to_owned());
    }
    Ok((inputs, outputs))
}

/// Port declarations split at commas outside quotes and brackets.
fn split_ports(text: &str) -> Vec<&str> {
    let (mut parts, mut depth, mut quoted, mut start) = (Vec::new(), 0i32, false, 0);
    for (k, ch) in text.char_indices() {
        match ch {
            '"' => quoted = !quoted,
            '[' | '{' if !quoted => depth += 1,
            ']' | '}' if !quoted => depth -= 1,
            ',' if !quoted && depth == 0 => {
                parts.push(&text[start..k]);
                start = k + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts.into_iter().map(str::trim).filter(|part| !part.is_empty()).collect()
}

/// One `name: kind = default` declaration.
fn port(item: &str) -> Result<ScriptPort, String> {
    let (declaration, default) = match item.split_once('=') {
        Some((declaration, default)) => (declaration, Some(default.trim())),
        None => (item, None),
    };
    let (name, kind) = match declaration.split_once(':') {
        Some((name, kind)) => (name.trim(), kind.trim()),
        None => (declaration.trim(), "any"),
    };
    let identifier = name.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_alphanumeric() || c == '_');
    if !identifier {
        return Err(format!("`{name}` is not a name: use letters, digits and _"));
    }
    let (base, list) = match kind.strip_suffix("[]") {
        Some(base) => (base.trim(), true),
        None => (kind, false),
    };
    let kind = PortType {
        kind: Type::parse(base).ok_or_else(|| {
            format!("`{base}` is not a kind: use number, text, bool, point, vector, curve, plane, geometry or any")
        })?,
        list,
    };
    let default = match default {
        None => Value::Null,
        Some(text) => serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_owned())),
    };
    Ok(ScriptPort { name: name.to_owned(), kind, default })
}

#[cfg(feature = "script")]
fn to_dynamic(value: &Value) -> rhai::Dynamic {
    match value {
        Value::Null => rhai::Dynamic::UNIT,
        Value::Bool(b) => (*b).into(),
        Value::Number(n) => match n.as_i64() {
            Some(i) if n.is_i64() => i.into(),
            _ => n.as_f64().unwrap_or(f64::NAN).into(),
        },
        Value::String(s) => s.clone().into(),
        Value::Array(items) => rhai::Dynamic::from_array(items.iter().map(to_dynamic).collect()),
        Value::Object(map) => rhai::Dynamic::from_map(
            map.iter().map(|(key, value)| (key.as_str().into(), to_dynamic(value))).collect(),
        ),
    }
}

#[cfg(feature = "script")]
fn from_dynamic(value: &rhai::Dynamic) -> Value {
    if value.is_unit() {
        Value::Null
    } else if let Ok(b) = value.as_bool() {
        Value::Bool(b)
    } else if let Ok(i) = value.as_int() {
        Value::from(i)
    } else if let Ok(x) = value.as_float() {
        crate::value::number(x)
    } else if value.is_string() {
        Value::String(value.clone().into_string().unwrap_or_default())
    } else if value.is_array() {
        let items = value.read_lock::<rhai::Array>().map(|items| items.iter().map(from_dynamic).collect());
        Value::Array(items.unwrap_or_default())
    } else if value.is_map() {
        let map = value.read_lock::<rhai::Map>().map(|map| {
            map.iter().map(|(key, value)| (key.to_string(), from_dynamic(value))).collect()
        });
        Value::Object(map.unwrap_or_default())
    } else {
        Value::String(value.to_string())
    }
}

/// A host for library operations a script calls: values only, so a script
/// never reaches the drawing.
#[cfg(feature = "script")]
struct Values;

#[cfg(feature = "script")]
impl crate::Host for Values {
    fn objects(
        &mut self,
        _: crate::NodeId,
        _: &str,
        _: &[Vec<(String, Value)>],
    ) -> Vec<Vec<(String, Value)>> {
        Vec::new()
    }

    fn curve(&self, _: &Value) -> Option<kernel::space::PlanarCurve> {
        None
    }

    fn expression(&self, text: &str) -> Option<f64> {
        crate::expr::eval(text, &|_| None).ok().map(crate::expr::Val::num)
    }
}

/// Runs library operation `name` on the inputs `map` names (the others at
/// their defaults): its only output, or a map of its outputs when it has
/// several.
#[cfg(feature = "script")]
fn call_op(name: &str, map: rhai::Map) -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
    let spec = crate::spec(name).ok_or_else(|| format!("there is no node `{name}`"))?;
    if !crate::ops::is_pure(spec.name) {
        return Err(format!("`{name}` reads the drawing or files; scripts cannot call it").into());
    }
    for key in map.keys() {
        if !spec.inputs.iter().any(|input| input.name == key.as_str()) {
            let names: Vec<&str> = spec.inputs.iter().map(|input| input.name).collect();
            return Err(format!("`{name}` has no input `{key}`; its inputs are {}", names.join(", ")).into());
        }
    }
    let args: Vec<Value> = spec
        .inputs
        .iter()
        .map(|input| map.get(input.name).map_or_else(|| input.default.value(), from_dynamic))
        .collect();
    let scalar: Vec<bool> = spec.inputs.iter().map(|input| !input.list).collect();
    let values = crate::value::lace(&args, &scalar, spec.outputs.len(), &mut |args| (spec.eval)(args, &Values));
    Ok(if let [only] = values.as_slice() {
        to_dynamic(only)
    } else {
        rhai::Dynamic::from_map(
            spec.outputs.iter().zip(&values).map(|(name, value)| ((*name).into(), to_dynamic(value))).collect(),
        )
    })
}

/// Output `output` of operation `name` on `map`.
#[cfg(feature = "script")]
fn op_output(name: &str, output: &str, map: rhai::Map) -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
    let result = call_op(name, map)?;
    // An operation with one output gives it as is, a map only when the
    // output is itself one (a point, a curve).
    if crate::spec(name).is_some_and(|spec| spec.outputs.len() == 1) {
        return Ok(result);
    }
    let picked = result
        .read_lock::<rhai::Map>()
        .map(|outputs| outputs.get(output).cloned().unwrap_or(rhai::Dynamic::UNIT));
    Ok(picked.unwrap_or(result))
}

/// The engine every script runs in: limited, and with the geometry API.
#[cfg(feature = "script")]
fn engine() -> rhai::Engine {
    use rhai::{Dynamic, Map, FLOAT};
    let mut engine = rhai::Engine::new();
    engine.set_max_operations(MAX_OPERATIONS);
    engine.set_max_expr_depths(64, 32);
    engine.set_max_call_levels(32);
    engine.set_max_string_size(1 << 20);
    engine.set_max_array_size(100_000);
    engine.set_max_map_size(10_000);
    engine.on_print(|_| {});
    engine.on_debug(|_, _, _| {});
    let map = |pairs: &[(&str, Dynamic)]| -> Map { pairs.iter().map(|(k, v)| ((*k).into(), v.clone())).collect() };
    let p = |x: FLOAT, y: FLOAT, z: FLOAT| to_dynamic(&crate::value::point_value([x, y, z]));

    engine.register_fn("op", |name: &str, inputs: Map| call_op(name, inputs));
    engine.register_fn("point", move |x: FLOAT, y: FLOAT| p(x, y, 0.0));
    engine.register_fn("point", move |x: FLOAT, y: FLOAT, z: FLOAT| p(x, y, z));
    engine.register_fn("point", move |x: rhai::INT, y: rhai::INT| p(x as FLOAT, y as FLOAT, 0.0));
    engine.register_fn("line", move |a: Dynamic, b: Dynamic| {
        op_output("Line Curve", "curve", map(&[("start", a), ("end", b)]))
    });
    engine.register_fn("polyline", move |points: rhai::Array| {
        op_output("Polyline Curve", "curve", map(&[("points", Dynamic::from_array(points))]))
    });
    engine.register_fn("polyline", move |points: rhai::Array, closed: bool| {
        op_output("Polyline Curve", "curve", map(&[("points", Dynamic::from_array(points)), ("closed", closed.into())]))
    });
    engine.register_fn("circle", move |center: Dynamic, radius: FLOAT| {
        op_output("Circle Curve", "curve", map(&[("center", center), ("radius", radius.into())]))
    });
    engine.register_fn("length", move |curve: Map| {
        op_output("Curve Length", "result", map(&[("curve", Dynamic::from_map(curve))]))
    });
    engine.register_fn("point_at", move |curve: Map, distance: FLOAT| {
        op_output("Frame At Distance", "point", map(&[("curve", Dynamic::from_map(curve)), ("distance", distance.into())]))
    });
    engine.register_fn("angle_at", move |curve: Map, distance: FLOAT| {
        op_output("Frame At Distance", "angle", map(&[("curve", Dynamic::from_map(curve)), ("distance", distance.into())]))
    });
    engine.register_fn("offset", move |curve: Map, distance: FLOAT| {
        op_output("Offset Curve", "curve", map(&[("curve", Dynamic::from_map(curve)), ("distance", distance.into())]))
    });
    engine.register_fn("distance", |a: Map, b: Map| -> FLOAT {
        let axis = |m: &Map, k: &str| m.get(k).and_then(|v| v.as_float().ok().or(v.as_int().ok().map(|i| i as FLOAT))).unwrap_or(0.0);
        let d = ["x", "y", "z"].map(|k| axis(&a, k) - axis(&b, k));
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    });
    // Degrees, as everywhere in the library.
    engine.register_fn("sind", |x: FLOAT| x.to_radians().sin());
    engine.register_fn("cosd", |x: FLOAT| x.to_radians().cos());
    engine.register_fn("tand", |x: FLOAT| x.to_radians().tan());
    engine.register_fn("atan2d", |y: FLOAT, x: FLOAT| y.atan2(x).to_degrees());
    engine
}

#[cfg(all(test, feature = "script"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn headers_declare_ports_with_kinds_and_defaults() {
        let script = Script::new("S", TEMPLATE);
        assert_eq!(script.error, None);
        let names: Vec<&str> = script.inputs.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(script.inputs[0].default, json!(1));
        assert_eq!(script.outputs[1].kind, PortType { kind: Type::Number, list: false });
        let (out, issue) = script.evaluate(&[json!(3), json!(4.5)]);
        assert_eq!((out, issue), (vec![json!(7.5), json!(13.5)], None));
        // A list on a single-value input runs once per element.
        let (out, _) = script.evaluate(&[json!([1, 2]), json!(10)]);
        assert_eq!(out[0], json!([11, 12]));

        let bad = Script::new("S", "// inputs: 2x: number\n// outputs: y: number\n1");
        assert!(bad.error.as_deref().is_some_and(|why| why.starts_with("line 1:")), "{:?}", bad.error);
        let kinds = Script::new("S", "// inputs: names: text[] = [\"a\", \"b\"], k: weird\n// outputs: y\n1");
        assert!(kinds.error.as_deref().is_some_and(|why| why.contains("`weird` is not a kind")));
        assert!(Script::new("S", "let x = 1;").error.is_some(), "outputs are required");
    }

    #[test]
    fn scripts_report_errors_with_their_line_and_stop_runaway_loops() {
        let script = Script::new("S", "// outputs: y: number\nlet a = 1;\nlet b = a + c;\nb");
        let (out, issue) = script.evaluate(&[]);
        assert_eq!(out, vec![Value::Null]);
        assert!(issue.as_deref().is_some_and(|why| why.starts_with("line 3:") && why.contains('c')), "{issue:?}");
        let syntax = Script::new("S", "// outputs: y: number\nlet a = (1;");
        assert!(syntax.error.as_deref().is_some_and(|why| why.starts_with("line 2:")), "{:?}", syntax.error);
        let forever = Script::new("S", "// outputs: y: number\nloop { }");
        let (_, issue) = forever.evaluate(&[]);
        assert!(issue.is_some(), "stopped");
        let shape = Script::new("S", "// outputs: a: number, b: number\n5");
        assert!(shape.evaluate(&[]).1.is_some_and(|why| why.contains("map of a, b")));
    }

    #[test]
    fn helpers_give_curves_points_and_numbers() {
        let source = r#"
// outputs: line: curve, ring: curve, path: curve, side: curve, length: number, mid: point, angle: number
let a = point(0.0, 0.0);
let b = point(4, 3);
let line = line(a, b);
#{
    line: line,
    ring: circle(a, 2.0),
    path: polyline([a, b, point(4.0, 0.0)], true),
    side: offset(line, 1.0),
    length: length(line),
    mid: point_at(line, 2.5),
    angle: angle_at(line, 0.0),
}
"#;
        let (out, issue) = Script::new("S", source).evaluate(&[]);
        assert_eq!(issue, None);
        for k in 0..4 {
            assert!(crate::value::curve(&out[k]).is_some(), "output {k}: {}", out[k]);
        }
        assert_eq!(out[4], json!(5.0));
        assert_eq!(crate::value::point(&out[5]), Some([2.0, 1.5, 0.0]));
        assert!((crate::value::num(&out[6]).unwrap() - 36.86989764584402).abs() < 1e-9);
    }

    #[test]
    fn script_nodes_link_save_and_load_like_any_step() {
        let mut graph = crate::Graph::default();
        let number = graph.add(crate::Kind::Op(crate::spec("Number").unwrap()));
        let script = graph.add(crate::Kind::Script(Script::new(
            "Double",
            "// inputs: x: number = 0\n// outputs: y: number\nx * 2",
        )));
        assert_eq!(graph.node(script).unwrap().params["x"], json!(0));
        let port = |node, field: &str| crate::Port { node, field: field.to_owned() };
        assert!(graph.connect(port(number, "value"), port(script, "x")));
        graph.node_mut(number).unwrap().params.insert("value".into(), json!(21));
        graph.evaluate(&mut Values);
        assert_eq!(graph.node(script).unwrap().outputs["y"], json!(42.0));

        let saved = graph.to_json();
        let mut loaded = crate::Graph::from_json(&saved).unwrap();
        loaded.evaluate(&mut Values);
        assert_eq!(loaded.node(script).unwrap().outputs["y"], json!(42.0));
        let crate::Kind::Script(copy) = &loaded.node(script).unwrap().kind else { panic!() };
        assert_eq!(copy.name, "Double");

        let broken = graph.add(crate::Kind::Script(Script::new("Bad", "// outputs: y\nnope")));
        graph.evaluate(&mut Values);
        let issues = &graph.node(broken).unwrap().issues;
        assert!(matches!(&issues[..], [crate::Issue::Script(why)] if why.contains("nope")), "{issues:?}");
    }

    #[test]
    fn scripts_build_geometry_with_the_library() {
        let source = r#"
// inputs: axis: curve, spacing: number = 2.5
// outputs: posts: point[], count: number, side: curve
let total = length(axis);
let posts = [];
let d = 0.0;
while d <= total + 1e-9 {
    posts.push(point_at(axis, d));
    d += spacing;
}
let side = op("Offset Curve", #{ curve: axis, distance: 1.0 });
#{ posts: posts, count: posts.len(), side: side }
"#;
        let script = Script::new("Posts", source);
        assert_eq!(script.error, None);
        let axis = crate::ops::spec("Line Curve").unwrap();
        let axis = (axis.eval)(
            &[crate::value::point_value([0.0, 0.0, 0.0]), crate::value::point_value([10.0, 0.0, 0.0])],
            &Values,
        )
        .remove(0);
        let (out, issue) = script.evaluate(&[axis, json!(2.5)]);
        assert_eq!(issue, None);
        assert_eq!(out[1], json!(5));
        assert_eq!(crate::value::point(&out[0][4]), Some([10.0, 0.0, 0.0]));
        assert!(crate::value::curve(&out[2]).is_some());
        let wrong = Script::new("S", "// outputs: y\nop(\"Add\", #{ q: 1 })");
        assert!(wrong.evaluate(&[]).1.is_some_and(|why| why.contains("its inputs are a, b")));
        let drawing = Script::new("S", "// outputs: y\nop(\"Layer Objects\", #{})");
        assert!(drawing.evaluate(&[]).1.is_some_and(|why| why.contains("cannot call")));
    }
}
