//! What each library operation does and the kind of value each of its ports
//! holds, for interfaces to show and for automation clients to read.
//!
//! The kinds are a description, not a contract: values stay plain JSON and
//! the operations coerce what they get (see [`crate::value`]).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ops::Spec;

/// The kind of value a port holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Any,
    Number,
    Text,
    Bool,
    Point,
    /// A direction or displacement, written like a point.
    Vector,
    /// A curve value or a drawn object a curve can be read from.
    Curve,
    Plane,
    /// A point, curve, plane or a list of them: what a transform carries.
    Geometry,
    /// A drawn object, as an object node gives it.
    Object,
}

impl Type {
    /// The kind's name, as written in [`describe`] signatures.
    pub fn name(self) -> &'static str {
        match self {
            Type::Any => "any",
            Type::Number => "number",
            Type::Text => "text",
            Type::Bool => "bool",
            Type::Point => "point",
            Type::Vector => "vector",
            Type::Curve => "curve",
            Type::Plane => "plane",
            Type::Geometry => "geometry",
            Type::Object => "object",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        [
            Type::Any,
            Type::Number,
            Type::Text,
            Type::Bool,
            Type::Point,
            Type::Vector,
            Type::Curve,
            Type::Plane,
            Type::Geometry,
            Type::Object,
        ]
        .into_iter()
        .find(|kind| kind.name() == name)
    }
}

/// A port's kind; `list` when it holds a whole list of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortType {
    pub kind: Type,
    pub list: bool,
}

/// What an operation does and the kinds of its ports, in port order.
#[derive(Clone, Debug)]
pub struct Doc {
    pub description: &'static str,
    pub inputs: Vec<PortType>,
    pub outputs: Vec<PortType>,
}

/// The documentation of `spec`.
pub fn doc(spec: &Spec) -> Option<&'static Doc> {
    docs().get(spec.name)
}

/// The kind input `field` of `spec` holds.
pub fn input_type(spec: &Spec, field: &str) -> Option<PortType> {
    let index = spec.inputs.iter().position(|input| input.name == field)?;
    doc(spec)?.inputs.get(index).copied()
}

/// The kind output `field` of `spec` holds.
pub fn output_type(spec: &Spec, field: &str) -> Option<PortType> {
    let index = spec.outputs.iter().position(|output| *output == field)?;
    doc(spec)?.outputs.get(index).copied()
}

fn docs() -> &'static HashMap<&'static str, Doc> {
    static DOCS: OnceLock<HashMap<&'static str, Doc>> = OnceLock::new();
    DOCS.get_or_init(|| {
        TABLE
            .iter()
            .filter_map(|(name, signature, description)| {
                let (inputs, outputs) = signature.split_once("->")?;
                Some((
                    *name,
                    Doc {
                        description,
                        inputs: parse(inputs)?,
                        outputs: parse(outputs)?,
                    },
                ))
            })
            .collect()
    })
}

/// `"point, number*2, curve[]"` as port kinds.
fn parse(side: &str) -> Option<Vec<PortType>> {
    let mut ports = Vec::new();
    for item in side.split(',').map(str::trim).filter(|item| !item.is_empty()) {
        let (item, count) = match item.split_once('*') {
            Some((item, count)) => (item, count.trim().parse().ok()?),
            None => (item, 1),
        };
        let (name, list) = match item.trim().strip_suffix("[]") {
            Some(name) => (name, true),
            None => (item.trim(), false),
        };
        let port = PortType { kind: Type::parse(name)?, list };
        ports.extend(std::iter::repeat_n(port, count));
    }
    Some(ports)
}

/// Every operation: its name, `inputs -> outputs` kinds and what it does.
const TABLE: &[(&str, &str, &str)] = &[
    // ── Input ────────────────────────────────────────────────────────────
    ("Number", "number -> number", "A number typed by hand."),
    ("Number Slider", "number*4 -> number", "A number set with a slider between a minimum and a maximum, in steps."),
    ("Integer Slider", "number*4 -> number", "A whole number set with a slider between a minimum and a maximum."),
    ("Text", "text -> text", "A text typed by hand."),
    ("Boolean", "bool -> bool", "A true or false switch."),
    ("Formula", "text, number*4 -> number", "Evaluates an arithmetic expression of a, b, c and d, such as a*2+b."),
    ("Watch", "any -> any", "Shows the value it receives and passes it on unchanged."),
    ("Table", "any[], bool -> any[]*9, number", "A table typed or pasted by hand: each column comes out as a list, the rows as a list of lists, and the count of rows. With header, the first row names the columns."),
    // ── Math ─────────────────────────────────────────────────────────────
    ("Add", "number*2 -> number", "a + b."),
    ("Subtract", "number*2 -> number", "a − b."),
    ("Multiply", "number*2 -> number", "a × b."),
    ("Divide", "number*2 -> number", "a ÷ b."),
    ("Modulo", "number*2 -> number", "The remainder of a ÷ b, never negative."),
    ("Power", "number*2 -> number", "a raised to the power b."),
    ("Minimum", "number*2 -> number", "The smaller of a and b."),
    ("Maximum", "number*2 -> number", "The larger of a and b."),
    ("Negate", "number -> number", "The number with its sign changed."),
    ("Absolute", "number -> number", "The number without its sign."),
    ("Square Root", "number -> number", "The square root of the number."),
    ("Exponential", "number -> number", "e raised to the number."),
    ("Natural Log", "number -> number", "The natural logarithm of the number."),
    ("Log10", "number -> number", "The base 10 logarithm of the number."),
    ("Sine", "number -> number", "The sine of an angle in degrees."),
    ("Cosine", "number -> number", "The cosine of an angle in degrees."),
    ("Tangent", "number -> number", "The tangent of an angle in degrees."),
    ("Arcsine", "number -> number", "The angle in degrees whose sine is the number."),
    ("Arccosine", "number -> number", "The angle in degrees whose cosine is the number."),
    ("Arctangent", "number -> number", "The angle in degrees whose tangent is the number."),
    ("Arctangent2", "number*2 -> number", "The angle in degrees of the direction (x, y), from −180 to 180."),
    ("Round", "number*2 -> number", "The number rounded to the given count of decimal digits."),
    ("Floor", "number -> number", "The largest whole number not above the number."),
    ("Ceiling", "number -> number", "The smallest whole number not below the number."),
    ("Clamp", "number*3 -> number", "The number kept between a minimum and a maximum."),
    ("Remap", "number*5 -> number", "Maps the number from one range onto another, proportionally."),
    ("To Radians", "number -> number", "Converts degrees to radians."),
    ("To Degrees", "number -> number", "Converts radians to degrees."),
    ("Pi", " -> number", "The number π."),
    ("E", " -> number", "The number e."),
    ("Random", "number*2 -> number[]", "A list of random numbers from 0 to 1; the same seed gives the same list."),
    // ── Logic ────────────────────────────────────────────────────────────
    ("Equal", "any*2 -> bool", "Whether a equals b."),
    ("Not Equal", "any*2 -> bool", "Whether a differs from b."),
    ("Less", "any*2 -> bool", "Whether a is less than b; texts compare alphabetically."),
    ("Less Or Equal", "any*2 -> bool", "Whether a is less than or equal to b."),
    ("Greater", "any*2 -> bool", "Whether a is greater than b."),
    ("Greater Or Equal", "any*2 -> bool", "Whether a is greater than or equal to b."),
    ("And", "bool*2 -> bool", "True when a and b are both true."),
    ("Or", "bool*2 -> bool", "True when a, b or both are true."),
    ("Xor", "bool*2 -> bool", "True when exactly one of a and b is true."),
    ("Not", "bool -> bool", "The opposite of a."),
    ("If", "bool, any*2 -> any", "Passes on then when the test is true, else otherwise."),
    ("Choose", "number, any*5 -> any", "The option at an index, counting from 0: one of several values, curves or lists picked by a number."),
    // ── List ─────────────────────────────────────────────────────────────
    ("Range", "number*3, bool -> number[]", "Numbers from start to end in steps; include end adds the end when the steps fall short of it."),
    ("Sequence", "number*3 -> number[]", "A count of numbers from start, in steps."),
    ("Create List", "any*5 -> any[]", "A list of the items given, in order; empty items are skipped."),
    ("Count", "any[] -> number", "How many items the list has."),
    ("Get Item", "any[], number -> any", "The item at an index, counting from 0; negative indexes count from the end."),
    ("First Item", "any[] -> any", "The first item of the list."),
    ("Last Item", "any[] -> any", "The last item of the list."),
    ("Reverse", "any[] -> any[]", "The list in reverse order."),
    ("Sort", "any[] -> any[]", "The list sorted from smallest to largest, or alphabetically."),
    ("Sum", "number[] -> number", "The sum of the numbers in the list."),
    ("Average", "number[] -> number", "The average of the numbers in the list."),
    ("List Minimum", "number[] -> number", "The smallest number in the list."),
    ("List Maximum", "number[] -> number", "The largest number in the list."),
    ("Flatten", "any[] -> any[]", "Every item of a nested list, in one flat list."),
    ("Slice", "any[], number*2 -> any[]", "The items from start up to end; negative bounds count from the end."),
    ("Join Lists", "any[]*2 -> any[]", "List a followed by list b."),
    ("Repeat Item", "any, number -> any[]", "A list with the item repeated count times."),
    ("Filter By Mask", "any[], bool[] -> any[]*2", "Splits a list in the items whose mask is true and the rest."),
    ("Unique Items", "any[] -> any[]", "The list without repeated items, in first-seen order."),
    ("Index Of", "any[], any -> number", "The index of the first item equal to the one given, or −1."),
    ("Contains", "any[], any -> bool", "Whether the list holds the item."),
    ("Chop", "any[], number -> any[]", "Splits the list into lists of a given size."),
    ("Transpose", "any[] -> any[]", "Swaps rows and columns of a list of lists."),
    ("Graft", "any[] -> any[]", "Puts every item in a list of its own, so each one is handled apart."),
    ("Flatten Once", "any[] -> any[]", "Removes one level of nesting from a list of lists."),
    ("List Depth", "any[] -> number", "How many levels of lists the value has; 0 for a single item."),
    ("Cross Reference", "any[]*2 -> any[]*2", "Every pairing of the items of a with the items of b."),
    ("Weave", "any[]*2 -> any[]", "Alternates the items of a and b in one list."),
    ("Shift List", "any[], number, bool -> any[]", "Moves the items by an offset, wrapping round the ends or dropping them."),
    ("Cull Pattern", "any[], bool[] -> any[]", "Keeps the items where a repeating pattern of true and false is true."),
    // ── Text ─────────────────────────────────────────────────────────────
    ("Concatenate", "text*2 -> text", "Text a followed by text b. Lists join item by item."),
    ("Text Length", "text -> number", "How many characters the text has."),
    ("Upper Case", "text -> text", "The text in capital letters."),
    ("Lower Case", "text -> text", "The text in small letters."),
    ("Split", "text*2 -> text[]", "Splits a text at a separator into a list of texts."),
    ("Join Text", "text[], text -> text", "Joins a list of texts into one, with a separator between them."),
    ("Replace", "text*3 -> text", "The text with every occurrence of search replaced."),
    ("Substring", "text, number*2 -> text", "Part of the text: length characters from start, counting from 0."),
    ("Text Contains", "text*2 -> bool", "Whether the text contains search."),
    ("To Number", "text -> number", "Reads a number from a text."),
    ("To Text", "any -> text", "Writes the whole value, a list included, as one text."),
    ("Format Number", "number*4, text*2 -> text", "Writes a number with fixed decimals, at least digits integer digits, and the digits in groups joined by a separator: 1,234.50 or the station 0+030.00 (4 digits, groups of 3, separator +)."),
    // ── Geometry ─────────────────────────────────────────────────────────
    ("Point", "number*3 -> point", "A point from its x, y and z coordinates."),
    ("Point Components", "point -> number*3", "The x, y and z coordinates of a point."),
    ("Distance", "point*2 -> number", "The distance between two points."),
    ("Midpoint", "point*2 -> point", "The point halfway between two points."),
    ("Interpolate", "point*2, number -> point", "The point at fraction t of the way from a to b."),
    ("Angle", "point*2 -> number", "The angle in degrees of the direction from a to b, in the XY plane."),
    ("Translate", "point, number*3 -> point", "The point moved by dx, dy and dz."),
    ("Polar Point", "point, number*2 -> point", "The point at a distance and an angle in degrees from another."),
    ("Rotate Point", "point*2, number -> point", "The point rotated about a centre by an angle in degrees."),
    ("Scale Point", "point*2, number -> point", "The point scaled from a centre by a factor."),
    ("Curve Length", "curve -> number", "The length of a curve."),
    ("Curve Area", "curve -> number", "The area a closed curve encloses."),
    ("Curve Endpoints", "curve -> point*2", "The start and end points of a curve."),
    ("Point At Parameter", "curve, number -> point", "The point at parameter t of a curve, 0 at its start and 1 at its end."),
    ("Point At Distance", "curve, number -> point", "The point at a distance along a curve from its start."),
    ("Parameter At Point", "curve, point -> number", "The parameter t of a curve at one of its points."),
    ("Divide Curve", "curve, number -> point[]", "Points dividing a curve into count equal lengths, both ends included."),
    ("Intersect", "curve*2 -> point[]", "The points where two curves on the same plane cross."),
    // ── Curve ────────────────────────────────────────────────────────────
    ("Line Curve", "point*2 -> curve", "A line from a start point to an end point."),
    ("Circle Curve", "point, number -> curve", "A circle from its centre and radius."),
    ("Arc Curve", "point, number*3 -> curve", "An arc from its centre, radius and start and end angles in degrees."),
    ("Polyline Curve", "point[], bool -> curve", "A polyline through a list of points, open or closed."),
    ("Offset Curve", "curve, number -> curve", "A copy of a curve at a distance to its side; negative distances go to the other side."),
    ("Line By Direction", "point, vector, number -> curve", "A line from a point along a direction; a length of 0 uses the direction's own length."),
    ("Arc 3 Points", "point*3 -> curve", "The arc from a through b to c."),
    ("Circle 3 Points", "point*3 -> curve", "The circle through three points."),
    ("Rectangle", "point, number*2 -> curve", "A closed rectangle from a corner, a width and a height."),
    ("Polygon", "point, number*2 -> curve", "A regular polygon from its centre, radius and number of sides."),
    ("Explode Curve", "curve -> curve[]", "The lines and arcs a curve is made of."),
    ("Join Curves", "curve[] -> curve[]", "Joins curves whose ends meet into polylines."),
    ("Reverse Curve", "curve -> curve", "The curve running the other way."),
    ("Close Curve", "curve -> curve", "The curve closed back to its start."),
    ("Extend Curve", "curve, number*2 -> curve", "The curve lengthened at its start and its end by the given lengths."),
    ("Fillet Corners", "curve, number -> curve", "The curve with its corners rounded with arcs of a radius."),
    ("Offset Both Sides", "curve, number -> curve, curve*2", "The two parallels of a curve at half a width to each side, and the closed outline between them: a wall, a path or a trench from its axis. A closed curve gives both sides as its outline."),
    ("Region Union", "curve[]*2 -> curve[], number", "Everything the closed curves of a and b cover, as regions (each its outline and its holes), and its area: walls that meet, merged."),
    ("Region Difference", "curve[]*2 -> curve[], number", "What the closed curves of a cover and those of b do not, as regions (each its outline and its holes), and its area: a slab less its openings."),
    ("Region Intersection", "curve[]*2 -> curve[], number", "What the closed curves of a and of b both cover, as regions (each its outline and its holes), and its area."),
    ("Sub Curve", "curve, number*2 -> curve", "The part of a curve between two distances along it."),
    ("Split At Distances", "curve, number[] -> curve[]*3", "Cuts a curve at distances along it: every piece, and the even (1st, 3rd…) and odd (2nd, 4th…) pieces apart, such as wall stretches and openings."),
    ("Closest Point", "curve, point -> point, number*2", "The point of a curve nearest to a point, its parameter t and the distance to it."),
    // ── Vector ───────────────────────────────────────────────────────────
    ("Vector", "number*3 -> vector", "A vector from its x, y and z components."),
    ("Vector Between", "point*2 -> vector", "The vector from one point to another."),
    ("Vector Length", "vector -> number", "The length of a vector."),
    ("Unit Vector", "vector -> vector", "The vector scaled to length 1."),
    ("Scale Vector", "vector, number -> vector", "The vector multiplied by a factor."),
    ("Add Vectors", "vector*2 -> vector", "The sum of two vectors."),
    ("Dot Product", "vector*2 -> number", "The dot product of two vectors."),
    ("Cross Product", "vector*2 -> vector", "The vector perpendicular to two vectors."),
    ("Plane", "point, vector*2 -> plane", "A plane from an origin and its x and y directions."),
    ("XY Plane", "point -> plane", "The world XY plane at an origin."),
    ("Curve Frame", "curve, number -> plane", "The plane at parameter t of a curve: x along the curve, y across it."),
    ("Frame At Distance", "curve, number -> plane, point, vector*2, number", "The plane at a distance along a curve, its point, the tangent and perpendicular directions and the tangent's angle in degrees."),
    ("Plane Components", "plane -> point, vector*3", "The origin, x and y directions and normal of a plane."),
    // ── Transform ────────────────────────────────────────────────────────
    ("Move", "geometry, vector -> geometry", "The geometry moved by a vector."),
    ("Rotate", "geometry, point, number -> geometry", "The geometry rotated about a centre by an angle in degrees."),
    ("Scale", "geometry, point, number -> geometry", "The geometry scaled from a centre by a factor."),
    ("Mirror", "geometry, point*2 -> geometry", "The geometry mirrored across the line through two points."),
    ("Orient", "geometry, plane*2 -> geometry", "The geometry carried from one plane to another."),
    ("Linear Array", "geometry, vector, number -> geometry[]", "Count copies of the geometry, each one a vector further."),
    ("Polar Array", "geometry, point, number*2 -> geometry[]", "Count copies of the geometry around a centre, spread over an angle."),
    ("Curve Array", "geometry, curve, number -> geometry[]", "Count copies of the geometry along a curve, turned with it; the geometry is drawn about the origin."),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LIBRARY;

    #[test]
    fn every_operation_is_documented_port_by_port() {
        let names: std::collections::HashSet<_> = TABLE.iter().map(|(name, ..)| *name).collect();
        assert_eq!(names.len(), TABLE.len(), "an operation is documented twice");
        assert_eq!(docs().len(), TABLE.len(), "a signature does not parse");
        for spec in LIBRARY {
            let doc = doc(spec).unwrap_or_else(|| panic!("{} has no doc", spec.name));
            assert_eq!(doc.inputs.len(), spec.inputs.len(), "{} inputs", spec.name);
            assert_eq!(doc.outputs.len(), spec.outputs.len(), "{} outputs", spec.name);
            for (input, kind) in spec.inputs.iter().zip(&doc.inputs) {
                // A list kind runs no lacing: the input takes the whole list.
                assert!(!kind.list || input.list, "{}.{} lists", spec.name, input.name);
            }
        }
        assert!(names.iter().all(|name| crate::spec(name).is_some()));
    }

    #[test]
    fn ports_have_kinds() {
        let spec = crate::spec("Frame At Distance").unwrap();
        assert_eq!(input_type(spec, "curve").unwrap().kind, Type::Curve);
        assert_eq!(output_type(spec, "tangent").unwrap().kind, Type::Vector);
        let range = crate::spec("Range").unwrap();
        assert_eq!(output_type(range, "list"), Some(PortType { kind: Type::Number, list: true }));
    }
}
