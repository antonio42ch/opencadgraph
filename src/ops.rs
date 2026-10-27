//! The built-in node library.
//!
//! Each [`Spec`] names its inputs and outputs and evaluates one call. Inputs
//! not marked `list` are scalar: a list arriving there makes the node run once
//! per element (see [`crate::value::lace`]), so `Add` over two lists adds them
//! pairwise without any list-aware code in `Add` itself. Angles are degrees.

use std::cmp::Ordering;

use kernel::geom2d::{cross, Tolerance};
use serde_json::Value;

use crate::geometry as g;
use crate::value::{
    boolean, curve, curve_value, list, num, number, plane, plane_at, plane_value, point,
    point_value, text,
};
use crate::Host;

/// An input's value while nothing is linked to it.
#[derive(Clone, Copy, Debug)]
pub enum Initial {
    None,
    Num(f64),
    Text(&'static str),
    Bool(bool),
}

impl Initial {
    pub fn value(self) -> Value {
        match self {
            Initial::None => Value::Null,
            Initial::Num(x) => number(x),
            Initial::Text(s) => Value::String(s.to_owned()),
            Initial::Bool(b) => Value::Bool(b),
        }
    }
}

#[derive(Debug)]
pub struct Input {
    pub name: &'static str,
    pub default: Initial,
    /// Takes a whole list instead of running once per element.
    pub list: bool,
}

pub struct Spec {
    pub name: &'static str,
    pub category: &'static str,
    pub inputs: &'static [Input],
    pub outputs: &'static [&'static str],
    pub eval: fn(&[Value], &dyn Host) -> Vec<Value>,
}

impl std::fmt::Debug for Spec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

pub const CATEGORIES: &[&str] =
    &["Input", "Math", "Logic", "List", "Text", "Geometry", "Vector", "Curve", "Transform"];

/// The spec called `name`.
pub fn spec(name: &str) -> Option<&'static Spec> {
    LIBRARY.iter().find(|spec| spec.name == name)
}

const fn n(name: &'static str, x: f64) -> Input {
    Input { name, default: Initial::Num(x), list: false }
}
const fn s(name: &'static str, value: &'static str) -> Input {
    Input { name, default: Initial::Text(value), list: false }
}
const fn b(name: &'static str, value: bool) -> Input {
    Input { name, default: Initial::Bool(value), list: false }
}
/// A scalar input with no default (points, curves, arbitrary items).
const fn v(name: &'static str) -> Input {
    Input { name, default: Initial::None, list: false }
}
/// A whole-list input.
const fn l(name: &'static str) -> Input {
    Input { name, default: Initial::None, list: true }
}

const X: &[Input] = &[n("x", 0.0)];
const AB: &[Input] = &[n("a", 0.0), n("b", 0.0)];
const PQ: &[Input] = &[v("a"), v("b")];
const LIST: &[Input] = &[l("list")];
const TEXT: &[Input] = &[s("text", "")];
const CURVE: &[Input] = &[v("curve")];
const RESULT: &[&str] = &["result"];
const VALUE: &[&str] = &["value"];
const POINT: &[&str] = &["point"];

fn out(value: Value) -> Vec<Value> {
    vec![value]
}

fn f1(a: &[Value], f: fn(f64) -> f64) -> Vec<Value> {
    out(num(&a[0]).map_or(Value::Null, |x| number(f(x))))
}

fn f2(a: &[Value], f: fn(f64, f64) -> f64) -> Vec<Value> {
    out(match (num(&a[0]), num(&a[1])) {
        (Some(x), Some(y)) => number(f(x, y)),
        _ => Value::Null,
    })
}

fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x.partial_cmp(&y),
        _ => Some(text(a).cmp(&text(b))),
    }
}

fn equal(a: &Value, b: &Value) -> bool {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
        _ => a == b,
    }
}

fn test(a: &[Value], f: fn(Ordering) -> bool) -> Vec<Value> {
    out(compare(&a[0], &a[1]).map_or(Value::Null, |o| Value::Bool(f(o))))
}

fn logic(a: &[Value], f: fn(bool, bool) -> bool) -> Vec<Value> {
    out(match (boolean(&a[0]), boolean(&a[1])) {
        (Some(x), Some(y)) => Value::Bool(f(x, y)),
        _ => Value::Null,
    })
}

fn numbers(value: &Value) -> Vec<f64> {
    list(value).iter().filter_map(num).collect()
}

fn index(len: usize, i: f64) -> Option<usize> {
    let i = i as i64;
    let i = if i < 0 { len as i64 + i } else { i };
    (0..len as i64).contains(&i).then_some(i as usize)
}

/// Upper bound on generated list lengths, so a mistyped step cannot hang the
/// evaluation.
const MAX_ITEMS: usize = 100_000;

fn range(start: f64, count: usize, step: f64) -> Value {
    Value::Array((0..count.min(MAX_ITEMS)).map(|k| number(start + step * k as f64)).collect())
}

/// `x` with `decimals` decimals, at least `digits` integer digits (zero
/// padded), the integer digits in groups of `group` from the right joined by
/// `separator` (none when 0): 1234.5 as `1,234.50`, or 30 as the station
/// `0+030.00` with 4 digits in groups of 3 joined by `+`.
fn format_number(x: f64, decimals: usize, digits: usize, group: usize, separator: &str, mark: &str) -> String {
    let fixed = format!("{:.*}", decimals, x.abs());
    let (whole, fraction) = fixed.split_once('.').unwrap_or((&fixed, ""));
    let whole = format!("{whole:0>digits$}");
    let mut grouped = String::new();
    for (k, digit) in whole.chars().enumerate() {
        if group > 0 && k > 0 && (whole.len() - k) % group == 0 {
            grouped.push_str(separator);
        }
        grouped.push(digit);
    }
    let negative = x < 0.0 && fixed.chars().any(|c| c.is_ascii_digit() && c != '0');
    let mut result = if negative { format!("-{grouped}") } else { grouped };
    if !fraction.is_empty() {
        result.push_str(mark);
        result.push_str(fraction);
    }
    result
}

/// Columns a Table node gives apart.
const TABLE_COLUMNS: usize = 8;

fn flatten(value: &Value, into: &mut Vec<Value>) {
    match value {
        Value::Array(items) => items.iter().for_each(|item| flatten(item, into)),
        other => into.push(other.clone()),
    }
}

fn map_point(a: &Value, f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<Value> {
    out(point(a).map_or(Value::Null, |p| point_value(f(p))))
}

fn rotate_xy(p: [f64; 3], c: [f64; 3], degrees: f64) -> [f64; 3] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    [c[0] + dx * cos - dy * sin, c[1] + dx * sin + dy * cos, p[2]]
}

/// Replaces the variables `a`..`d` in a formula with their values.
fn substitute(expression: &str, values: &[Value]) -> String {
    let mut result = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, result: &mut String| {
        match ["a", "b", "c", "d"].iter().position(|name| *name == word.as_str()) {
            Some(i) => result.push_str(&format!("({})", num(&values[i]).unwrap_or(0.0))),
            None => result.push_str(word),
        }
        word.clear();
    };
    for ch in expression.chars() {
        if ch.is_alphanumeric() || ch == '_' || (ch == '.' && !word.is_empty()) {
            word.push(ch);
        } else {
            flush(&mut word, &mut result);
            result.push(ch);
        }
    }
    flush(&mut word, &mut result);
    result
}

/// The curve a value stands for: a curve value, or whatever the host
/// resolves (a drawn object).
fn curve_of(value: &Value, host: &dyn Host) -> Option<kernel::space::PlanarCurve> {
    curve(value).or_else(|| host.curve(value))
}

/// A curve on the XY plane through height `z`, as a value.
fn flat_curve(z: f64, shape: kernel::geom2d::Curve) -> Value {
    curve_value(&kernel::space::PlanarCurve::new(plane_at(z), shape)).unwrap_or(Value::Null)
}

/// The parallel of `curve` at `distance`, positive to the left of its
/// direction; a polyline may come apart into several pieces.
fn offset(curve: &kernel::space::PlanarCurve, distance: f64) -> Value {
    use kernel::geom2d::Curve;
    let value = |shape: Curve| {
        curve_value(&kernel::space::PlanarCurve::new(curve.plane, shape)).unwrap_or(Value::Null)
    };
    match &curve.curve {
        Curve::Polyline(polyline) => {
            let [a, b] = [0, 1].map(|k| polyline.vertices[k % polyline.vertices.len()].position);
            let along = [b[0] - a[0], b[1] - a[1]];
            let length = along[0].hypot(along[1]);
            if length == 0.0 || distance == 0.0 {
                return Value::Null;
            }
            // A point on the wanted side of the first segment's middle.
            let side = [
                (a[0] + b[0]) / 2.0 - along[1] / length * distance,
                (a[1] + b[1]) / 2.0 + along[0] / length * distance,
            ];
            let mut pieces: Vec<Value> = kernel::geom2d::offset_polyline(polyline, distance, side)
                .into_iter()
                .map(|piece| value(Curve::Polyline(piece)))
                .collect();
            match pieces.len() {
                0 => Value::Null,
                1 => pieces.remove(0),
                _ => Value::Array(pieces),
            }
        }
        shape => kernel::geom2d::offset_curve(shape, distance, 1e-6).map_or(Value::Null, value),
    }
}

/// Every closed curve in `value`, a curve or a nested list of them.
fn closed_curves(value: &Value, host: &dyn Host, into: &mut Vec<kernel::space::PlanarCurve>) {
    match value {
        Value::Array(items) => items.iter().for_each(|item| closed_curves(item, host, into)),
        Value::Null => {}
        other => into.extend(curve_of(other, host).filter(|c| c.curve.is_closed())),
    }
}

/// The regions `keep` picks from the closed curves of `a` and of `b`, each
/// a list of its outline and its holes, and their net area.
fn regions(a: &[Value], host: &dyn Host, keep: fn(bool, bool) -> bool) -> Vec<Value> {
    let (mut first, mut second) = (Vec::new(), Vec::new());
    closed_curves(&a[0], host, &mut first);
    closed_curves(&a[1], host, &mut second);
    let Some(plane) = first.iter().chain(&second).map(|c| c.plane).next() else {
        return vec![Value::Null; 2];
    };
    let on_plane = |shapes: Vec<kernel::space::PlanarCurve>| -> Vec<kernel::geom2d::Curve> {
        shapes.into_iter().filter(|c| same_plane(&c.plane, &plane)).map(|c| c.curve).collect()
    };
    let regions = g::boolean(&on_plane(first), &on_plane(second), keep);
    let area: f64 = regions.iter().flatten().map(|shape| shape.enclosed_area()).sum();
    vec![
        Value::Array(regions.into_iter()
            .map(|region| Value::Array(region.into_iter()
                .filter_map(|shape| curve_value(&kernel::space::PlanarCurve::new(plane, shape)))
                .collect()))
            .collect()),
        number(area),
    ]
}

/// A point or vector input, the origin when unset.
fn at(value: &Value) -> [f64; 3] {
    point(value).unwrap_or([0.0; 3])
}

/// `geometry` carried by the map `(point, vector)`.
fn moved(geometry: &Value, map: (impl Fn([f64; 3]) -> [f64; 3], impl Fn([f64; 3]) -> [f64; 3])) -> Value {
    g::map_value(geometry, &g::Map { point: &map.0, vector: &map.1 })
}

/// A curve from a curve value or the host, back as a value on its plane.
fn edit(value: &Value, host: &dyn Host, f: impl Fn(&kernel::geom2d::Curve) -> Option<kernel::geom2d::Curve>) -> Value {
    curve_of(value, host)
        .and_then(|c| f(&c.curve).map(|shape| kernel::space::PlanarCurve::new(c.plane, shape)))
        .and_then(|c| curve_value(&c))
        .unwrap_or(Value::Null)
}

/// Frames at `count` even spacings along `c`: both ends for an open curve,
/// once round for a closed one.
fn frames_along(c: &kernel::space::PlanarCurve, count: usize) -> Vec<kernel::space::Plane> {
    let length = c.length();
    let steps = if c.curve.is_closed() { count } else { count.saturating_sub(1).max(1) };
    (0..count)
        .filter_map(|k| g::curve_frame(c, c.parameter_at_distance(length * k as f64 / steps as f64)))
        .collect()
}

fn same_plane(a: &kernel::space::Plane, b: &kernel::space::Plane) -> bool {
    let close = |p: [f64; 3], q: [f64; 3]| p.iter().zip(q).all(|(x, y)| (x - y).abs() <= 1e-9);
    close(a.origin, b.origin) && close(a.x_axis, b.x_axis) && close(a.y_axis, b.y_axis)
}

pub const LIBRARY: &[Spec] = &[
    // ── Input ────────────────────────────────────────────────────────────
    Spec { name: "Number", category: "Input", inputs: &[n("value", 0.0)], outputs: VALUE,
        eval: |a, _| out(num(&a[0]).map_or(Value::Null, number)) },
    Spec { name: "Number Slider", category: "Input",
        inputs: &[n("value", 5.0), n("min", 0.0), n("max", 10.0), n("step", 0.1)], outputs: VALUE,
        eval: |a, _| {
            let [value, min, max, step] = [0, 1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0));
            let snapped = if step > 0.0 { min + ((value - min) / step).round() * step } else { value };
            out(number(snapped.clamp(min.min(max), max.max(min))))
        } },
    Spec { name: "Integer Slider", category: "Input",
        inputs: &[n("value", 5.0), n("min", 0.0), n("max", 10.0), n("step", 1.0)], outputs: VALUE,
        eval: |a, _| {
            let [value, min, max, step] = [0, 1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0).round());
            let snapped = if step > 0.0 { min + ((value - min) / step).round() * step } else { value };
            out(number(snapped.clamp(min.min(max), max.max(min))))
        } },
    Spec { name: "Text", category: "Input", inputs: &[s("value", "")], outputs: VALUE,
        eval: |a, _| out(Value::String(text(&a[0]))) },
    Spec { name: "Boolean", category: "Input", inputs: &[b("value", false)], outputs: VALUE,
        eval: |a, _| out(boolean(&a[0]).map_or(Value::Null, Value::Bool)) },
    Spec { name: "Formula", category: "Input",
        inputs: &[s("expression", "a+b"), n("a", 0.0), n("b", 0.0), n("c", 0.0), n("d", 0.0)],
        outputs: RESULT,
        eval: |a, host| out(host.expression(&substitute(&text(&a[0]), &a[1..])).map_or(Value::Null, number)) },
    Spec { name: "Watch", category: "Input", inputs: &[l("in")], outputs: &["out"],
        eval: |a, _| out(a[0].clone()) },
    Spec { name: "Table", category: "Input", inputs: &[l("table"), b("header", true)],
        outputs: &["column 1", "column 2", "column 3", "column 4", "column 5", "column 6", "column 7", "column 8",
            "rows", "count"],
        eval: |a, _| {
            let mut rows = crate::value::table(&a[0]);
            if boolean(&a[1]).unwrap_or(true) && !rows.is_empty() {
                rows.remove(0);
            }
            let mut outputs: Vec<Value> = (0..TABLE_COLUMNS)
                .map(|k| Value::Array(rows.iter().map(|row| row.get(k).cloned().unwrap_or(Value::Null)).collect()))
                .collect();
            outputs.push(number(rows.len() as f64));
            outputs.insert(TABLE_COLUMNS, Value::Array(rows.into_iter().map(Value::Array).collect()));
            outputs
        } },
    // ── Math ─────────────────────────────────────────────────────────────
    Spec { name: "Add", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x + y) },
    Spec { name: "Subtract", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x - y) },
    Spec { name: "Multiply", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x * y) },
    Spec { name: "Divide", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x / y) },
    Spec { name: "Modulo", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x.rem_euclid(y)) },
    Spec { name: "Power", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, f64::powf) },
    Spec { name: "Minimum", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, f64::min) },
    Spec { name: "Maximum", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, f64::max) },
    Spec { name: "Negate", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| -x) },
    Spec { name: "Absolute", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::abs) },
    Spec { name: "Square Root", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::sqrt) },
    Spec { name: "Exponential", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::exp) },
    Spec { name: "Natural Log", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::ln) },
    Spec { name: "Log10", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::log10) },
    Spec { name: "Sine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.to_radians().sin()) },
    Spec { name: "Cosine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.to_radians().cos()) },
    Spec { name: "Tangent", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.to_radians().tan()) },
    Spec { name: "Arcsine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.asin().to_degrees()) },
    Spec { name: "Arccosine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.acos().to_degrees()) },
    Spec { name: "Arctangent", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.atan().to_degrees()) },
    Spec { name: "Arctangent2", category: "Math", inputs: &[n("y", 0.0), n("x", 1.0)], outputs: RESULT,
        eval: |a, _| f2(a, |y, x| y.atan2(x).to_degrees()) },
    Spec { name: "Round", category: "Math", inputs: &[n("x", 0.0), n("digits", 0.0)], outputs: RESULT,
        eval: |a, _| f2(a, |x, d| { let m = 10f64.powi(d as i32); (x * m).round() / m }) },
    Spec { name: "Floor", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::floor) },
    Spec { name: "Ceiling", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::ceil) },
    Spec { name: "Clamp", category: "Math", inputs: &[n("x", 0.0), n("min", 0.0), n("max", 1.0)], outputs: RESULT,
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(lo), Some(hi)) => number(x.clamp(lo.min(hi), hi.max(lo))),
            _ => Value::Null,
        }) },
    Spec { name: "Remap", category: "Math",
        inputs: &[n("x", 0.0), n("from min", 0.0), n("from max", 1.0), n("to min", 0.0), n("to max", 1.0)],
        outputs: RESULT,
        eval: |a, _| {
            let values: Option<Vec<f64>> = a.iter().map(num).collect();
            out(values.map_or(Value::Null, |v| number(v[3] + (v[0] - v[1]) / (v[2] - v[1]) * (v[4] - v[3]))))
        } },
    Spec { name: "To Radians", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::to_radians) },
    Spec { name: "To Degrees", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::to_degrees) },
    Spec { name: "Pi", category: "Math", inputs: &[], outputs: RESULT, eval: |_, _| out(number(std::f64::consts::PI)) },
    Spec { name: "E", category: "Math", inputs: &[], outputs: RESULT, eval: |_, _| out(number(std::f64::consts::E)) },
    Spec { name: "Random", category: "Math", inputs: &[n("seed", 1.0), n("count", 10.0)], outputs: &["list"],
        eval: |a, _| {
            let mut state = (num(&a[0]).unwrap_or(1.0) as u64) ^ 0x9E37_79B9_7F4A_7C15;
            let count = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::Array((0..count.min(MAX_ITEMS)).map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                number((state >> 11) as f64 / (1u64 << 53) as f64)
            }).collect()))
        } },
    // ── Logic ────────────────────────────────────────────────────────────
    Spec { name: "Equal", category: "Logic", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(Value::Bool(equal(&a[0], &a[1]))) },
    Spec { name: "Not Equal", category: "Logic", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(Value::Bool(!equal(&a[0], &a[1]))) },
    Spec { name: "Less", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_lt) },
    Spec { name: "Less Or Equal", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_le) },
    Spec { name: "Greater", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_gt) },
    Spec { name: "Greater Or Equal", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_ge) },
    Spec { name: "And", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: |a, _| logic(a, |x, y| x && y) },
    Spec { name: "Or", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: |a, _| logic(a, |x, y| x || y) },
    Spec { name: "Xor", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: |a, _| logic(a, |x, y| x != y) },
    Spec { name: "Not", category: "Logic", inputs: &[b("a", false)], outputs: RESULT,
        eval: |a, _| out(boolean(&a[0]).map_or(Value::Null, |x| Value::Bool(!x))) },
    Spec { name: "If", category: "Logic", inputs: &[b("test", true), v("then"), v("else")], outputs: RESULT,
        eval: |a, _| out(match boolean(&a[0]) {
            Some(true) => a[1].clone(),
            Some(false) => a[2].clone(),
            None => Value::Null,
        }) },
    Spec { name: "Choose", category: "Logic",
        inputs: &[n("index", 0.0), l("option 0"), l("option 1"), l("option 2"), l("option 3"), l("option 4")],
        outputs: &["value"],
        eval: |a, _| out(num(&a[0])
            .filter(|i| *i >= 0.0)
            .and_then(|i| a.get(1 + i as usize))
            .cloned()
            .unwrap_or(Value::Null)) },
    // ── List ─────────────────────────────────────────────────────────────
    Spec { name: "Range", category: "List",
        inputs: &[n("start", 0.0), n("end", 10.0), n("step", 1.0), b("include end", false)],
        outputs: &["list"],
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(start), Some(end), Some(step)) if step != 0.0 && (end - start) / step >= 0.0 => {
                let count = ((end - start) / step + 1e-9).floor() as usize + 1;
                let mut values = range(start, count, step);
                // A step that does not divide the span stops short of `end`.
                let last = start + step * (count - 1) as f64;
                if boolean(&a[3]) == Some(true)
                    && count < MAX_ITEMS
                    && (end - last).abs() > 1e-9 * end.abs().max(1.0)
                {
                    if let Value::Array(items) = &mut values {
                        items.push(number(end));
                    }
                }
                values
            }
            _ => Value::Array(Vec::new()),
        }) },
    Spec { name: "Sequence", category: "List", inputs: &[n("start", 0.0), n("count", 10.0), n("step", 1.0)],
        outputs: &["list"],
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(start), Some(count), Some(step)) => range(start, count.max(0.0) as usize, step),
            _ => Value::Array(Vec::new()),
        }) },
    Spec { name: "Create List", category: "List",
        inputs: &[l("item 0"), l("item 1"), l("item 2"), l("item 3"), l("item 4")], outputs: &["list"],
        eval: |a, _| out(Value::Array(a.iter().filter(|item| !item.is_null()).cloned().collect())) },
    Spec { name: "Count", category: "List", inputs: LIST, outputs: &["count"],
        eval: |a, _| out(number(list(&a[0]).len() as f64)) },
    Spec { name: "Get Item", category: "List", inputs: &[l("list"), n("index", 0.0)], outputs: &["item"],
        eval: |a, _| {
            let items = list(&a[0]);
            out(num(&a[1]).and_then(|i| index(items.len(), i)).map_or(Value::Null, |i| items[i].clone()))
        } },
    Spec { name: "First Item", category: "List", inputs: LIST, outputs: &["item"],
        eval: |a, _| out(list(&a[0]).first().cloned().unwrap_or(Value::Null)) },
    Spec { name: "Last Item", category: "List", inputs: LIST, outputs: &["item"],
        eval: |a, _| out(list(&a[0]).last().cloned().unwrap_or(Value::Null)) },
    Spec { name: "Reverse", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| { let mut items = list(&a[0]); items.reverse(); out(Value::Array(items)) } },
    Spec { name: "Sort", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| {
            let mut items = list(&a[0]);
            items.sort_by(|x, y| compare(x, y).unwrap_or(Ordering::Equal));
            out(Value::Array(items))
        } },
    Spec { name: "Sum", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| out(number(numbers(&a[0]).iter().sum())) },
    Spec { name: "Average", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| { let x = numbers(&a[0]); out(number(x.iter().sum::<f64>() / x.len() as f64)) } },
    Spec { name: "List Minimum", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| out(numbers(&a[0]).into_iter().reduce(f64::min).map_or(Value::Null, number)) },
    Spec { name: "List Maximum", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| out(numbers(&a[0]).into_iter().reduce(f64::max).map_or(Value::Null, number)) },
    Spec { name: "Flatten", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| { let mut items = Vec::new(); flatten(&a[0], &mut items); out(Value::Array(items)) } },
    Spec { name: "Slice", category: "List", inputs: &[l("list"), n("start", 0.0), n("end", -1.0)],
        outputs: &["list"],
        eval: |a, _| {
            let items = list(&a[0]);
            let len = items.len();
            let bound = |value: &Value, fallback: usize| match num(value) {
                Some(i) if i < 0.0 => (len as f64 + i + 1.0).max(0.0) as usize,
                Some(i) => (i as usize).min(len),
                None => fallback,
            };
            let (start, end) = (bound(&a[1], 0), bound(&a[2], len));
            out(Value::Array(items.get(start..end.max(start)).unwrap_or_default().to_vec()))
        } },
    Spec { name: "Join Lists", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: |a, _| out(Value::Array([list(&a[0]), list(&a[1])].concat())) },
    Spec { name: "Repeat Item", category: "List", inputs: &[l("item"), n("count", 3.0)], outputs: &["list"],
        eval: |a, _| {
            let count = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::Array(vec![a[0].clone(); count.min(MAX_ITEMS)]))
        } },
    Spec { name: "Filter By Mask", category: "List", inputs: &[l("list"), l("mask")], outputs: &["in", "out"],
        eval: |a, _| {
            let (mut kept, mut dropped) = (Vec::new(), Vec::new());
            for (item, keep) in list(&a[0]).into_iter().zip(list(&a[1])) {
                if boolean(&keep).unwrap_or(false) { kept.push(item) } else { dropped.push(item) }
            }
            vec![Value::Array(kept), Value::Array(dropped)]
        } },
    Spec { name: "Unique Items", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| {
            let mut items: Vec<Value> = Vec::new();
            for item in list(&a[0]) {
                if !items.iter().any(|seen| equal(seen, &item)) { items.push(item) }
            }
            out(Value::Array(items))
        } },
    Spec { name: "Index Of", category: "List", inputs: &[l("list"), l("item")], outputs: &["index"],
        eval: |a, _| out(number(list(&a[0]).iter().position(|x| equal(x, &a[1])).map_or(-1.0, |i| i as f64))) },
    Spec { name: "Contains", category: "List", inputs: &[l("list"), l("item")], outputs: RESULT,
        eval: |a, _| out(Value::Bool(list(&a[0]).iter().any(|x| equal(x, &a[1])))) },
    Spec { name: "Chop", category: "List", inputs: &[l("list"), n("size", 2.0)], outputs: &["list"],
        eval: |a, _| {
            let size = num(&a[1]).unwrap_or(1.0).max(1.0) as usize;
            out(Value::Array(list(&a[0]).chunks(size).map(|chunk| Value::Array(chunk.to_vec())).collect()))
        } },
    Spec { name: "Transpose", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| {
            let rows: Vec<Vec<Value>> = list(&a[0]).iter().map(list).collect();
            let width = rows.iter().map(Vec::len).min().unwrap_or(0);
            out(Value::Array((0..width).map(|k| Value::Array(rows.iter().map(|row| row[k].clone()).collect())).collect()))
        } },
    // Nested lists are the graph's trees: these reshape them.
    Spec { name: "Graft", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| out(Value::Array(list(&a[0]).into_iter().map(|item| Value::Array(vec![item])).collect())) },
    Spec { name: "Flatten Once", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| out(Value::Array(list(&a[0]).into_iter().flat_map(|item| list(&item)).collect())) },
    Spec { name: "List Depth", category: "List", inputs: LIST, outputs: &["depth"],
        eval: |a, _| {
            fn depth(value: &Value) -> usize {
                match value {
                    Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
                    _ => 0,
                }
            }
            out(number(depth(&a[0]) as f64))
        } },
    Spec { name: "Cross Reference", category: "List", inputs: &[l("a"), l("b")], outputs: &["a", "b"],
        eval: |a, _| {
            let (xs, ys) = (list(&a[0]), list(&a[1]));
            let pairs = xs.len().saturating_mul(ys.len()).min(MAX_ITEMS);
            let (mut left, mut right) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
            'all: for x in &xs {
                for y in &ys {
                    if left.len() == pairs {
                        break 'all;
                    }
                    left.push(x.clone());
                    right.push(y.clone());
                }
            }
            vec![Value::Array(left), Value::Array(right)]
        } },
    Spec { name: "Weave", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: |a, _| {
            let (xs, ys) = (list(&a[0]), list(&a[1]));
            let mut woven = Vec::with_capacity(xs.len() + ys.len());
            for k in 0..xs.len().max(ys.len()) {
                woven.extend(xs.get(k).cloned());
                woven.extend(ys.get(k).cloned());
            }
            out(Value::Array(woven))
        } },
    Spec { name: "Shift List", category: "List", inputs: &[l("list"), n("offset", 1.0), b("wrap", true)],
        outputs: &["list"],
        eval: |a, _| {
            let items = list(&a[0]);
            let len = items.len() as i64;
            let offset = num(&a[1]).unwrap_or(0.0) as i64;
            if len == 0 {
                return out(Value::Array(items));
            }
            out(Value::Array(if boolean(&a[2]).unwrap_or(true) {
                (0..len).map(|k| items[(k + offset).rem_euclid(len) as usize].clone()).collect()
            } else {
                (0..len).filter_map(|k| items.get((k + offset) as usize).filter(|_| k + offset >= 0).cloned()).collect()
            }))
        } },
    Spec { name: "Cull Pattern", category: "List", inputs: &[l("list"), l("pattern")], outputs: &["list"],
        eval: |a, _| {
            let pattern: Vec<bool> = list(&a[1]).iter().map(|keep| boolean(keep).unwrap_or(false)).collect();
            out(if pattern.is_empty() {
                a[0].clone()
            } else {
                Value::Array(list(&a[0]).into_iter().enumerate()
                    .filter(|(k, _)| pattern[k % pattern.len()])
                    .map(|(_, item)| item)
                    .collect())
            })
        } },
    // ── Text ─────────────────────────────────────────────────────────────
    Spec { name: "Concatenate", category: "Text", inputs: &[s("a", ""), s("b", "")], outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]) + &text(&a[1]))) },
    Spec { name: "Text Length", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(number(text(&a[0]).chars().count() as f64)) },
    Spec { name: "Upper Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]).to_uppercase())) },
    Spec { name: "Lower Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]).to_lowercase())) },
    Spec { name: "Split", category: "Text", inputs: &[s("text", ""), s("separator", ",")], outputs: &["list"],
        eval: |a, _| {
            let (source, separator) = (text(&a[0]), text(&a[1]));
            out(Value::Array(if separator.is_empty() {
                source.chars().map(|c| Value::String(c.to_string())).collect()
            } else {
                source.split(separator.as_str()).map(|part| Value::String(part.to_owned())).collect()
            }))
        } },
    Spec { name: "Join Text", category: "Text", inputs: &[l("list"), s("separator", ",")], outputs: RESULT,
        eval: |a, _| out(Value::String(list(&a[0]).iter().map(text).collect::<Vec<_>>().join(&text(&a[1])))) },
    Spec { name: "Replace", category: "Text", inputs: &[s("text", ""), s("search", ""), s("replacement", "")],
        outputs: RESULT,
        eval: |a, _| {
            let search = text(&a[1]);
            let source = text(&a[0]);
            out(Value::String(if search.is_empty() { source } else { source.replace(&search, &text(&a[2])) }))
        } },
    Spec { name: "Substring", category: "Text", inputs: &[s("text", ""), n("start", 0.0), n("length", 1.0)],
        outputs: RESULT,
        eval: |a, _| {
            let start = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            let length = num(&a[2]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::String(text(&a[0]).chars().skip(start).take(length).collect()))
        } },
    Spec { name: "Text Contains", category: "Text", inputs: &[s("text", ""), s("search", "")], outputs: RESULT,
        eval: |a, _| out(Value::Bool(text(&a[0]).contains(&text(&a[1])))) },
    Spec { name: "To Number", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(num(&a[0]).map_or(Value::Null, number)) },
    Spec { name: "To Text", category: "Text", inputs: &[l("value")], outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]))) },
    Spec { name: "Format Number", category: "Text",
        inputs: &[n("value", 0.0), n("decimals", 2.0), n("digits", 1.0), n("group size", 0.0),
            s("separator", ","), s("decimal mark", ".")],
        outputs: RESULT,
        eval: |a, _| out(match num(&a[0]) {
            Some(x) => Value::String(format_number(
                x,
                num(&a[1]).unwrap_or(0.0).clamp(0.0, 12.0) as usize,
                num(&a[2]).unwrap_or(1.0).clamp(1.0, 32.0) as usize,
                num(&a[3]).unwrap_or(0.0).clamp(0.0, 32.0) as usize,
                &text(&a[4]),
                &text(&a[5]),
            )),
            None => Value::Null,
        }) },
    // ── Geometry ─────────────────────────────────────────────────────────
    Spec { name: "Point", category: "Geometry", inputs: &[n("x", 0.0), n("y", 0.0), n("z", 0.0)], outputs: POINT,
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(y), Some(z)) => point_value([x, y, z]),
            _ => Value::Null,
        }) },
    Spec { name: "Point Components", category: "Geometry", inputs: &[v("point")], outputs: &["x", "y", "z"],
        eval: |a, _| match point(&a[0]) {
            Some(p) => p.map(number).to_vec(),
            None => vec![Value::Null; 3],
        } },
    Spec { name: "Distance", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number(p.iter().zip(q).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()),
            _ => Value::Null,
        }) },
    Spec { name: "Midpoint", category: "Geometry", inputs: PQ, outputs: POINT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => point_value([0, 1, 2].map(|i| (p[i] + q[i]) * 0.5)),
            _ => Value::Null,
        }) },
    Spec { name: "Interpolate", category: "Geometry", inputs: &[v("a"), v("b"), n("t", 0.5)], outputs: POINT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1]), num(&a[2])) {
            (Some(p), Some(q), Some(t)) => point_value([0, 1, 2].map(|i| p[i] + (q[i] - p[i]) * t)),
            _ => Value::Null,
        }) },
    Spec { name: "Angle", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number((q[1] - p[1]).atan2(q[0] - p[0]).to_degrees()),
            _ => Value::Null,
        }) },
    Spec { name: "Translate", category: "Geometry",
        inputs: &[v("point"), n("dx", 0.0), n("dy", 0.0), n("dz", 0.0)], outputs: POINT,
        eval: |a, _| {
            let d = [1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0));
            map_point(&a[0], |p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]])
        } },
    Spec { name: "Polar Point", category: "Geometry", inputs: &[v("point"), n("angle", 0.0), n("distance", 1.0)],
        outputs: POINT,
        eval: |a, _| {
            let (angle, distance) = (num(&a[1]).unwrap_or(0.0), num(&a[2]).unwrap_or(0.0));
            let (sin, cos) = angle.to_radians().sin_cos();
            map_point(&a[0], |p| [p[0] + distance * cos, p[1] + distance * sin, p[2]])
        } },
    Spec { name: "Rotate Point", category: "Geometry", inputs: &[v("point"), v("center"), n("angle", 90.0)],
        outputs: POINT,
        eval: |a, _| match point(&a[1]) {
            Some(center) => {
                let angle = num(&a[2]).unwrap_or(0.0);
                map_point(&a[0], |p| rotate_xy(p, center, angle))
            }
            None => out(Value::Null),
        } },
    Spec { name: "Scale Point", category: "Geometry", inputs: &[v("point"), v("center"), n("factor", 2.0)],
        outputs: POINT,
        eval: |a, _| match point(&a[1]) {
            Some(c) => {
                let k = num(&a[2]).unwrap_or(1.0);
                map_point(&a[0], |p| [0, 1, 2].map(|i| c[i] + (p[i] - c[i]) * k))
            }
            None => out(Value::Null),
        } },
    Spec { name: "Curve Length", category: "Geometry", inputs: CURVE, outputs: RESULT,
        eval: |a, host| out(curve_of(&a[0], host).map_or(Value::Null, |c| number(c.length()))) },
    Spec { name: "Curve Area", category: "Geometry", inputs: CURVE, outputs: RESULT,
        eval: |a, host| out(curve_of(&a[0], host).map_or(Value::Null, |c| number(c.curve.enclosed_area().abs()))) },
    Spec { name: "Curve Endpoints", category: "Geometry", inputs: CURVE, outputs: &["start", "end"],
        eval: |a, host| match curve_of(&a[0], host) {
            Some(c) => vec![point_value(c.point_at(0.0)), point_value(c.point_at(1.0))],
            None => vec![Value::Null; 2],
        } },
    Spec { name: "Point At Parameter", category: "Geometry", inputs: &[v("curve"), n("t", 0.5)], outputs: POINT,
        eval: |a, host| out(match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(t)) => point_value(c.point_at(t)),
            _ => Value::Null,
        }) },
    Spec { name: "Point At Distance", category: "Geometry", inputs: &[v("curve"), n("distance", 0.0)],
        outputs: POINT,
        eval: |a, host| out(match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(d)) => point_value(c.point_at_distance(d)),
            _ => Value::Null,
        }) },
    Spec { name: "Parameter At Point", category: "Geometry", inputs: &[v("curve"), v("point")], outputs: &["t"],
        eval: |a, host| out(match (curve_of(&a[0], host), point(&a[1])) {
            (Some(c), Some(p)) => c.parameter_at(p).map_or(Value::Null, number),
            _ => Value::Null,
        }) },
    Spec { name: "Divide Curve", category: "Geometry", inputs: &[v("curve"), n("count", 10.0)], outputs: &["points"],
        eval: |a, host| out(match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(count)) if count >= 1.0 => {
                let (count, length) = ((count as usize).min(MAX_ITEMS), c.length());
                Value::Array((0..=count).map(|k| point_value(c.point_at_distance(length * k as f64 / count as f64))).collect())
            }
            _ => Value::Null,
        }) },
    Spec { name: "Intersect", category: "Geometry", inputs: &[v("a"), v("b")], outputs: &["points"],
        eval: |a, host| out(match (curve_of(&a[0], host), curve_of(&a[1], host)) {
            // Curves on different planes meet at most in isolated points the
            // plane-curve intersection cannot see; they report none.
            (Some(p), Some(q)) if same_plane(&p.plane, &q.plane) => Value::Array(
                cross::intersect(&p.curve, &q.curve, Tolerance::new(1e-9))
                    .into_iter()
                    .map(|crossing| point_value(p.plane.point_at(crossing.point)))
                    .collect(),
            ),
            _ => Value::Array(Vec::new()),
        }) },
    // ── Curve ────────────────────────────────────────────────────────────
    // Curves made here are values: they flow to other nodes and only reach
    // the drawing through a node that draws them.
    Spec { name: "Line Curve", category: "Curve", inputs: &[v("start"), v("end")], outputs: &["curve"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => flat_curve(p[2], kernel::geom2d::Curve::Line(kernel::geom2d::Line {
                start: [p[0], p[1]],
                end: [q[0], q[1]],
            })),
            _ => Value::Null,
        }) },
    Spec { name: "Circle Curve", category: "Curve", inputs: &[v("center"), n("radius", 1.0)], outputs: &["curve"],
        eval: |a, _| out(match (point(&a[0]), num(&a[1])) {
            (Some(c), Some(r)) if r > 0.0 => flat_curve(c[2], kernel::geom2d::Curve::Circle(kernel::geom2d::Circle {
                centre: [c[0], c[1]],
                radius: r,
            })),
            _ => Value::Null,
        }) },
    Spec { name: "Arc Curve", category: "Curve",
        inputs: &[v("center"), n("radius", 1.0), n("from", 0.0), n("to", 90.0)], outputs: &["curve"],
        eval: |a, _| out(match (point(&a[0]), num(&a[1]), num(&a[2]), num(&a[3])) {
            (Some(c), Some(r), Some(from), Some(to)) if r > 0.0 => flat_curve(c[2], kernel::geom2d::Curve::Arc(kernel::geom2d::Arc {
                centre: [c[0], c[1]],
                radius: r,
                start_angle: from.to_radians(),
                end_angle: to.to_radians(),
            })),
            _ => Value::Null,
        }) },
    Spec { name: "Polyline Curve", category: "Curve", inputs: &[l("points"), b("closed", false)], outputs: &["curve"],
        eval: |a, _| {
            let points: Vec<[f64; 3]> = list(&a[0]).iter().filter_map(point).collect();
            out(if points.len() < 2 {
                Value::Null
            } else {
                flat_curve(points[0][2], kernel::geom2d::Curve::Polyline(kernel::geom2d::polyline::Polyline {
                    vertices: points
                        .iter()
                        .map(|p| kernel::geom2d::polyline::PolylineVertex::straight([p[0], p[1]]))
                        .collect(),
                    closed: boolean(&a[1]).unwrap_or(false),
                }))
            })
        } },
    Spec { name: "Offset Curve", category: "Curve", inputs: &[v("curve"), n("distance", 1.0)], outputs: &["curve"],
        eval: |a, host| out(match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(d)) => offset(&c, d),
            _ => Value::Null,
        }) },
    Spec { name: "Offset Both Sides", category: "Curve", inputs: &[v("curve"), n("width", 1.0)],
        outputs: &["outline", "left", "right"],
        eval: |a, host| match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(width)) if width > 0.0 => {
                let (left, right) = (offset(&c, width / 2.0), offset(&c, -width / 2.0));
                // A closed curve leaves a ring: its two sides are the outline.
                let outline = match (curve(&left), curve(&right)) {
                    (Some(_), Some(_)) if c.curve.is_closed() => Value::Array(vec![left.clone(), right.clone()]),
                    (Some(l), Some(r)) => g::outline(&l.curve, &r.curve)
                        .and_then(|shape| curve_value(&kernel::space::PlanarCurve::new(c.plane, shape)))
                        .unwrap_or(Value::Null),
                    _ => Value::Null,
                };
                vec![outline, left, right]
            }
            _ => vec![Value::Null; 3],
        } },
    Spec { name: "Region Union", category: "Curve", inputs: &[l("a"), l("b")], outputs: &["regions", "area"],
        eval: |a, host| regions(a, host, |a, b| a || b) },
    Spec { name: "Region Difference", category: "Curve", inputs: &[l("a"), l("b")], outputs: &["regions", "area"],
        eval: |a, host| regions(a, host, |a, b| a && !b) },
    Spec { name: "Region Intersection", category: "Curve", inputs: &[l("a"), l("b")], outputs: &["regions", "area"],
        eval: |a, host| regions(a, host, |a, b| a && b) },
    Spec { name: "Sub Curve", category: "Curve", inputs: &[v("curve"), n("from", 0.0), n("to", 1.0)],
        outputs: &["curve"],
        eval: |a, host| {
            let (from, to) = (num(&a[1]).unwrap_or(0.0), num(&a[2]).unwrap_or(0.0));
            out(edit(&a[0], host, |shape| g::sub_curve(shape, from, to)))
        } },
    Spec { name: "Split At Distances", category: "Curve", inputs: &[v("curve"), l("distances")],
        outputs: &["pieces", "even", "odd"],
        eval: |a, host| match curve_of(&a[0], host) {
            Some(c) => {
                let total = c.length();
                let mut cuts: Vec<f64> = list(&a[1]).iter().filter_map(num)
                    .filter(|d| *d > 1e-9 && *d < total - 1e-9)
                    .collect();
                cuts.sort_by(f64::total_cmp);
                cuts.dedup_by(|x, y| (*x - *y).abs() <= 1e-9);
                let bounds: Vec<f64> = std::iter::once(0.0).chain(cuts).chain(std::iter::once(total)).collect();
                let pieces: Vec<Value> = bounds.windows(2)
                    .map(|span| g::sub_curve(&c.curve, span[0], span[1])
                        .and_then(|shape| curve_value(&kernel::space::PlanarCurve::new(c.plane, shape)))
                        .unwrap_or(Value::Null))
                    .collect();
                let every = |first: usize| Value::Array(pieces.iter().skip(first).step_by(2).cloned().collect());
                let (even, odd) = (every(0), every(1));
                vec![Value::Array(pieces), even, odd]
            }
            None => vec![Value::Null; 3],
        } },
    // ── Vector ───────────────────────────────────────────────────────────
    // Vectors are written as points; planes as {"plane": …} (see value).
    Spec { name: "Vector", category: "Vector", inputs: &[n("x", 1.0), n("y", 0.0), n("z", 0.0)], outputs: &["vector"],
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(y), Some(z)) => point_value([x, y, z]),
            _ => Value::Null,
        }) },
    Spec { name: "Vector Between", category: "Vector", inputs: &[v("from"), v("to")], outputs: &["vector"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => point_value(g::sub(q, p)),
            _ => Value::Null,
        }) },
    Spec { name: "Vector Length", category: "Vector", inputs: &[v("vector")], outputs: RESULT,
        eval: |a, _| out(point(&a[0]).map_or(Value::Null, |v| number(g::length(v)))) },
    Spec { name: "Unit Vector", category: "Vector", inputs: &[v("vector")], outputs: &["vector"],
        eval: |a, _| out(point(&a[0]).and_then(g::unit).map_or(Value::Null, point_value)) },
    Spec { name: "Scale Vector", category: "Vector", inputs: &[v("vector"), n("factor", 2.0)], outputs: &["vector"],
        eval: |a, _| out(match (point(&a[0]), num(&a[1])) {
            (Some(v), Some(k)) => point_value(g::scale(v, k)),
            _ => Value::Null,
        }) },
    Spec { name: "Add Vectors", category: "Vector", inputs: PQ, outputs: &["vector"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => point_value(g::add(p, q)),
            _ => Value::Null,
        }) },
    Spec { name: "Dot Product", category: "Vector", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number(g::dot(p, q)),
            _ => Value::Null,
        }) },
    Spec { name: "Cross Product", category: "Vector", inputs: PQ, outputs: &["vector"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => point_value(g::cross(p, q)),
            _ => Value::Null,
        }) },
    Spec { name: "Plane", category: "Vector", inputs: &[v("origin"), v("x_axis"), v("y_axis")], outputs: &["plane"],
        eval: |a, _| out(g::frame(at(&a[0]), point(&a[1]).unwrap_or([1.0, 0.0, 0.0]), point(&a[2]).unwrap_or([0.0, 1.0, 0.0]))
            .map_or(Value::Null, |p| plane_value(&p))) },
    Spec { name: "XY Plane", category: "Vector", inputs: &[v("origin")], outputs: &["plane"],
        eval: |a, _| out(plane_value(&kernel::space::Plane { origin: at(&a[0]), ..kernel::space::Plane::XY })) },
    Spec { name: "Curve Frame", category: "Vector", inputs: &[v("curve"), n("t", 0.0)], outputs: &["plane"],
        eval: |a, host| out(match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(t)) => g::curve_frame(&c, t).map_or(Value::Null, |p| plane_value(&p)),
            _ => Value::Null,
        }) },
    Spec { name: "Frame At Distance", category: "Vector", inputs: &[v("curve"), n("distance", 0.0)],
        outputs: &["plane", "point", "tangent", "perpendicular", "angle"],
        eval: |a, host| match (curve_of(&a[0], host), num(&a[1])) {
            (Some(c), Some(d)) => match g::curve_frame(&c, c.parameter_at_distance(d)) {
                Some(p) => vec![
                    plane_value(&p),
                    point_value(p.origin),
                    point_value(p.x_axis),
                    point_value(p.y_axis),
                    number(p.x_axis[1].atan2(p.x_axis[0]).to_degrees()),
                ],
                None => vec![Value::Null; 5],
            },
            _ => vec![Value::Null; 5],
        } },
    Spec { name: "Plane Components", category: "Vector", inputs: &[v("plane")],
        outputs: &["origin", "x_axis", "y_axis", "normal"],
        eval: |a, _| match plane(&a[0]).and_then(|p| g::frame(p.origin, p.x_axis, p.y_axis)) {
            Some(p) => vec![
                point_value(p.origin),
                point_value(p.x_axis),
                point_value(p.y_axis),
                point_value(g::cross(p.x_axis, p.y_axis)),
            ],
            None => vec![Value::Null; 4],
        } },
    // ── Curve, built and edited ──────────────────────────────────────────
    Spec { name: "Line By Direction", category: "Curve", inputs: &[v("start"), v("direction"), n("length", 0.0)],
        outputs: &["curve"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1]).and_then(g::unit)) {
            (Some(p), Some(u)) => {
                // A zero or missing length takes the direction's own.
                let length = num(&a[2]).filter(|l| *l != 0.0)
                    .unwrap_or_else(|| point(&a[1]).map_or(0.0, g::length));
                let q = g::add(p, g::scale(u, length));
                flat_curve(p[2], kernel::geom2d::Curve::Line(kernel::geom2d::Line { start: [p[0], p[1]], end: [q[0], q[1]] }))
            }
            _ => Value::Null,
        }) },
    Spec { name: "Arc 3 Points", category: "Curve", inputs: &[v("a"), v("b"), v("c")], outputs: &["curve"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1]), point(&a[2])) {
            (Some(p), Some(q), Some(r)) => g::arc_through(p, q, r).map_or(Value::Null, |shape| flat_curve(p[2], shape)),
            _ => Value::Null,
        }) },
    Spec { name: "Circle 3 Points", category: "Curve", inputs: &[v("a"), v("b"), v("c")], outputs: &["curve"],
        eval: |a, _| out(match (point(&a[0]), point(&a[1]), point(&a[2])) {
            (Some(p), Some(q), Some(r)) => g::circle_through(p, q, r)
                .map_or(Value::Null, |(centre, radius)| flat_curve(p[2], g::circle(centre, radius))),
            _ => Value::Null,
        }) },
    Spec { name: "Rectangle", category: "Curve", inputs: &[v("corner"), n("width", 10.0), n("height", 5.0)],
        outputs: &["curve"],
        eval: |a, _| out(match (num(&a[1]), num(&a[2])) {
            (Some(w), Some(h)) if w != 0.0 && h != 0.0 => {
                let c = at(&a[0]);
                flat_curve(c[2], g::closed_polyline(&[[c[0], c[1]], [c[0] + w, c[1]], [c[0] + w, c[1] + h], [c[0], c[1] + h]]))
            }
            _ => Value::Null,
        }) },
    Spec { name: "Polygon", category: "Curve", inputs: &[v("center"), n("radius", 5.0), n("sides", 6.0)],
        outputs: &["curve"],
        eval: |a, _| out(match (num(&a[1]), num(&a[2])) {
            (Some(r), Some(sides)) if r > 0.0 && sides >= 3.0 => {
                let (c, sides) = (at(&a[0]), (sides as usize).min(MAX_ITEMS));
                let corners: Vec<[f64; 2]> = (0..sides)
                    .map(|k| {
                        let angle = std::f64::consts::TAU * k as f64 / sides as f64;
                        [c[0] + r * angle.cos(), c[1] + r * angle.sin()]
                    })
                    .collect();
                flat_curve(c[2], g::closed_polyline(&corners))
            }
            _ => Value::Null,
        }) },
    Spec { name: "Explode Curve", category: "Curve", inputs: CURVE, outputs: &["curves"],
        eval: |a, host| out(match curve_of(&a[0], host) {
            Some(c) => {
                let segments = c.curve.segments();
                let pieces = if segments.is_empty() { vec![c.curve.clone()] } else { segments };
                Value::Array(pieces.into_iter()
                    .filter_map(|shape| curve_value(&kernel::space::PlanarCurve::new(c.plane, shape)))
                    .collect())
            }
            None => Value::Null,
        }) },
    Spec { name: "Join Curves", category: "Curve", inputs: &[l("curves")], outputs: &["curves"],
        eval: |a, host| {
            let shapes: Vec<kernel::space::PlanarCurve> = list(&a[0]).iter().filter_map(|c| curve_of(c, host)).collect();
            out(Value::Array(g::join(&shapes, 1e-6).iter().filter_map(curve_value).collect()))
        } },
    Spec { name: "Reverse Curve", category: "Curve", inputs: CURVE, outputs: &["curve"],
        eval: |a, host| out(edit(&a[0], host, g::reverse)) },
    Spec { name: "Close Curve", category: "Curve", inputs: CURVE, outputs: &["curve"],
        eval: |a, host| out(edit(&a[0], host, g::close)) },
    Spec { name: "Extend Curve", category: "Curve", inputs: &[v("curve"), n("start", 0.0), n("end", 1.0)],
        outputs: &["curve"],
        eval: |a, host| {
            let (start, end) = (num(&a[1]).unwrap_or(0.0), num(&a[2]).unwrap_or(0.0));
            out(edit(&a[0], host, |shape| g::extend(shape, start, end)))
        } },
    Spec { name: "Fillet Corners", category: "Curve", inputs: &[v("curve"), n("radius", 1.0)], outputs: &["curve"],
        eval: |a, host| {
            let radius = num(&a[1]).unwrap_or(0.0);
            out(edit(&a[0], host, |shape| g::fillet(shape, radius)))
        } },
    Spec { name: "Closest Point", category: "Curve", inputs: &[v("curve"), v("point")],
        outputs: &["point", "t", "distance"],
        eval: |a, host| match (curve_of(&a[0], host), point(&a[1])) {
            (Some(c), Some(p)) => match c.plane.project(p) {
                Some(uv) => {
                    let found = kernel::geom2d::closest_point(&c.curve, uv);
                    let at = c.plane.point_at(found.point);
                    vec![point_value(at), number(found.t), number(g::length(g::sub(at, p)))]
                }
                None => vec![Value::Null; 3],
            },
            _ => vec![Value::Null; 3],
        } },
    // ── Transform ────────────────────────────────────────────────────────
    // Points, curves, planes and lists of them alike.
    Spec { name: "Move", category: "Transform", inputs: &[v("geometry"), v("vector")], outputs: &["geometry"],
        eval: |a, _| {
            let by = at(&a[1]);
            out(moved(&a[0], (move |p| g::add(p, by), |v| v)))
        } },
    Spec { name: "Rotate", category: "Transform", inputs: &[v("geometry"), v("center"), n("angle", 90.0)],
        outputs: &["geometry"],
        eval: |a, _| out(num(&a[2]).map_or(Value::Null, |angle| moved(&a[0], g::rotation(at(&a[1]), angle)))) },
    Spec { name: "Scale", category: "Transform", inputs: &[v("geometry"), v("center"), n("factor", 2.0)],
        outputs: &["geometry"],
        eval: |a, _| out(match num(&a[2]) {
            Some(k) if k != 0.0 => {
                let c = at(&a[1]);
                moved(&a[0], (move |p| g::add(c, g::scale(g::sub(p, c), k)), move |v| g::scale(v, k)))
            }
            _ => Value::Null,
        }) },
    Spec { name: "Mirror", category: "Transform", inputs: &[v("geometry"), v("a"), v("b")], outputs: &["geometry"],
        eval: |a, _| out(match (point(&a[1]), point(&a[2])) {
            (Some(p), Some(q)) => g::mirror(p, q).map_or(Value::Null, |map| moved(&a[0], map)),
            _ => Value::Null,
        }) },
    Spec { name: "Orient", category: "Transform", inputs: &[v("geometry"), v("from"), v("to")], outputs: &["geometry"],
        eval: |a, _| {
            let from = plane(&a[1]).unwrap_or(kernel::space::Plane::XY);
            out(match plane(&a[2]).and_then(|to| g::orientation(&from, &to)) {
                Some(map) => moved(&a[0], map),
                None => Value::Null,
            })
        } },
    Spec { name: "Linear Array", category: "Transform", inputs: &[v("geometry"), v("vector"), n("count", 3.0)],
        outputs: &["geometry"],
        eval: |a, _| {
            let (by, count) = (at(&a[1]), num(&a[2]).map_or(0, |c| (c.max(0.0) as usize).min(MAX_ITEMS)));
            out(Value::Array((0..count).map(|k| {
                let shift = g::scale(by, k as f64);
                moved(&a[0], (move |p| g::add(p, shift), |v| v))
            }).collect()))
        } },
    Spec { name: "Polar Array", category: "Transform",
        inputs: &[v("geometry"), v("center"), n("count", 6.0), n("angle", 360.0)], outputs: &["geometry"],
        eval: |a, _| {
            let count = num(&a[2]).map_or(0, |c| (c.max(0.0) as usize).min(MAX_ITEMS));
            let angle = num(&a[3]).unwrap_or(360.0);
            // A full turn spaces copies evenly round it; a part turn puts
            // the last copy at its end.
            let full = (angle.abs() - 360.0).abs() < 1e-9;
            let steps = if full { count } else { count.saturating_sub(1).max(1) };
            out(Value::Array((0..count)
                .map(|k| moved(&a[0], g::rotation(at(&a[1]), angle * k as f64 / steps as f64)))
                .collect()))
        } },
    Spec { name: "Curve Array", category: "Transform", inputs: &[v("geometry"), v("curve"), n("count", 5.0)],
        outputs: &["geometry"],
        eval: |a, host| out(match curve_of(&a[1], host) {
            Some(c) => {
                let count = num(&a[2]).map_or(0, |n| (n.max(0.0) as usize).min(MAX_ITEMS));
                Value::Array(frames_along(&c, count).iter()
                    .map(|to| g::orientation(&kernel::space::Plane::XY, to).map_or(Value::Null, |map| moved(&a[0], map)))
                    .collect())
            }
            None => Value::Null,
        }) },
];

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A host with no drawing: every curve must come from a value.
    struct NoDrawing;

    impl Host for NoDrawing {
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
        fn expression(&self, _: &str) -> Option<f64> {
            None
        }
    }

    fn run(name: &str, args: &[Value]) -> Vec<Value> {
        (spec(name).unwrap().eval)(args, &NoDrawing)
    }

    fn p(x: f64, y: f64) -> Value {
        point_value([x, y, 0.0])
    }

    #[test]
    fn curves_cut_at_distances_into_pieces_and_openings() {
        // An L: 10 along x, then 10 up.
        let path = run("Polyline Curve", &[json!([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0)]), json!(false)]).remove(0);
        let corner = run("Sub Curve", &[path.clone(), json!(8.0), json!(12.0)]).remove(0);
        assert!((length_of(&corner) - 4.0).abs() < 1e-9, "{corner}");
        let ends = run("Curve Endpoints", &[corner]);
        assert_eq!((point(&ends[0]), point(&ends[1])), (Some([8.0, 0.0, 0.0]), Some([10.0, 2.0, 0.0])));
        let straight = run("Sub Curve", &[path.clone(), json!(2.0), json!(5.0)]).remove(0);
        assert_eq!(straight["curve"], "line", "{straight}");

        let split = run("Split At Distances", &[path, json!([14.0, 3.0, 4.0, 13.0, 99.0])]);
        let lengths = |value: &Value| value.as_array().unwrap().iter().map(length_of).collect::<Vec<_>>();
        let close = |got: Vec<f64>, want: &[f64]| {
            got.len() == want.len() && got.iter().zip(want).all(|(x, y)| (x - y).abs() < 1e-9)
        };
        assert!(close(lengths(&split[0]), &[3.0, 1.0, 9.0, 1.0, 6.0]), "{}", split[0]);
        assert!(close(lengths(&split[1]), &[3.0, 9.0, 6.0]), "{}", split[1]);
        assert!(close(lengths(&split[2]), &[1.0, 1.0]), "{}", split[2]);

        // Half of a quarter arc is an eighth of a turn.
        let arc = run("Arc 3 Points", &[p(1.0, 0.0), p(0.5f64.sqrt(), 0.5f64.sqrt()), p(0.0, 1.0)]).remove(0);
        let half = run("Sub Curve", &[arc, json!(0.0), json!(std::f64::consts::FRAC_PI_4)]).remove(0);
        let end = point(&run("Curve Endpoints", &[half])[1]).unwrap();
        assert!((end[0] - 0.5f64.sqrt()).abs() < 1e-9 && (end[1] - 0.5f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn both_sides_of_an_axis_close_into_an_outline() {
        let axis = run("Line Curve", &[p(0.0, 0.0), p(10.0, 0.0)]).remove(0);
        let sides = run("Offset Both Sides", &[axis, json!(0.2)]);
        let area = |value: &Value| num(&run("Curve Area", &[value.clone()])[0]).unwrap();
        assert!((area(&sides[0]) - 2.0).abs() < 1e-9, "{}", sides[0]);
        assert_eq!(sides[1]["start"], json!([0.0, 0.1]));
        assert_eq!(sides[2]["start"], json!([0.0, -0.1]));
        let path = run("Polyline Curve", &[json!([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0)]), json!(false)]).remove(0);
        let outline = run("Offset Both Sides", &[path, json!(1.0)]).remove(0);
        assert!((area(&outline) - 20.0).abs() < 1e-6, "{outline}");
        let ring = run("Offset Both Sides", &[run("Circle Curve", &[p(0.0, 0.0), json!(5.0)]).remove(0), json!(1.0)]);
        assert_eq!(ring[0].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn regions_merge_cut_and_overlap() {
        let square = |x: f64, y: f64, size: f64| run("Rectangle", &[p(x, y), json!(size), json!(size)]).remove(0);
        let (a, b) = (square(0.0, 0.0, 10.0), square(5.0, 5.0, 10.0));
        let area = |name: &str, a: &Value, b: &Value| num(&run(name, &[a.clone(), b.clone()])[1]).unwrap();
        assert!((area("Region Union", &a, &b) - 175.0).abs() < 1e-6);
        assert!((area("Region Difference", &a, &b) - 75.0).abs() < 1e-6);
        assert!((area("Region Intersection", &a, &b) - 25.0).abs() < 1e-6);
        let union = run("Region Union", &[a.clone(), b]).remove(0);
        assert_eq!(union.as_array().unwrap().len(), 1, "one region: {union}");
        assert_eq!(union[0].as_array().unwrap().len(), 1, "no holes: {union}");
        // A hole inside: one region, its outline and a clockwise hole.
        let holed = run("Region Difference", &[a, square(2.0, 2.0, 2.0)]);
        assert_eq!(holed[0].as_array().unwrap().len(), 1);
        assert_eq!(holed[0][0].as_array().unwrap().len(), 2);
        assert!((num(&holed[1]).unwrap() - 96.0).abs() < 1e-6);
        let circle = run("Circle Curve", &[p(0.0, 0.0), json!(1.0)]).remove(0);
        let disc = num(&run("Region Union", &[circle, json!(null)])[1]).unwrap();
        assert!((disc - std::f64::consts::PI).abs() < 1e-2, "{disc}");
    }

    #[test]
    fn tables_give_columns_and_choose_picks_an_option() {
        let pasted = run("Table", &[json!("at\twidth\tkind\n2\t0.9\tdoor\n6\t1.5\twindow\n"), json!(true)]);
        assert_eq!(pasted[0], json!([2.0, 6.0]));
        assert_eq!(pasted[2], json!(["door", "window"]));
        assert_eq!(pasted[9], json!(2.0));
        let typed = run("Table", &[json!([["a", "b"], ["1", "x"]]), json!(false)]);
        assert_eq!(typed[0], json!(["a", 1.0]));
        let pick = run("Choose", &[json!(1.0), json!("simple"), json!([1, 2]), json!(null), json!(null), json!(null)]);
        assert_eq!(pick[0], json!([1, 2]));
        assert_eq!(run("Choose", &[json!(7.0), json!(1), json!(2), json!(3), json!(4), json!(5)])[0], Value::Null);
    }

    #[test]
    fn range_can_end_on_its_end() {
        let stations = |include: bool| run("Range", &[json!(0.0), json!(200.0), json!(30.0), json!(include)]);
        assert_eq!(stations(false)[0].as_array().unwrap().len(), 7);
        assert_eq!(stations(true)[0].as_array().unwrap().last(), Some(&json!(200.0)));
        let exact = run("Range", &[json!(0.0), json!(200.0), json!(20.0), json!(true)]);
        assert_eq!(exact[0].as_array().unwrap().len(), 11);
    }

    #[test]
    fn numbers_format_as_stations_and_thousands() {
        let format = |x: f64, decimals: f64, digits: f64, group: f64, separator: &str| {
            run("Format Number", &[json!(x), json!(decimals), json!(digits), json!(group), json!(separator), json!(".")])
                .remove(0)
        };
        assert_eq!(format(30.0, 2.0, 4.0, 3.0, "+"), json!("0+030.00"));
        assert_eq!(format(1234.5, 2.0, 4.0, 3.0, "+"), json!("1+234.50"));
        assert_eq!(format(1234567.891, 1.0, 1.0, 3.0, ","), json!("1,234,567.9"));
        assert_eq!(format(-7.0, 0.0, 3.0, 0.0, ","), json!("-007"));
        assert_eq!(format(-0.001, 2.0, 1.0, 0.0, ","), json!("0.00"));
    }

    #[test]
    fn a_frame_at_a_distance_follows_the_curve() {
        let path = run("Polyline Curve", &[json!([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0)]), json!(false)]).remove(0);
        let frame = run("Frame At Distance", &[path, json!(15.0)]);
        assert_eq!(point(&frame[1]), Some([10.0, 5.0, 0.0]));
        let tangent = point(&frame[2]).unwrap();
        assert!(tangent[0].abs() < 1e-9 && (tangent[1] - 1.0).abs() < 1e-9);
        let perpendicular = point(&frame[3]).unwrap();
        assert!((perpendicular[0] + 1.0).abs() < 1e-9);
        assert!((num(&frame[4]).unwrap() - 90.0).abs() < 1e-9);
    }

    #[test]
    fn curves_made_as_values_feed_the_curve_operations() {
        let line = run("Line Curve", &[p(0.0, 0.0), p(3.0, 4.0)]).remove(0);
        assert_eq!(line["curve"], "line");
        assert_eq!(run("Curve Length", &[line.clone()])[0], json!(5.0));
        let ends = run("Curve Endpoints", &[line]);
        assert_eq!(point(&ends[1]), Some([3.0, 4.0, 0.0]));

        let circle = run("Circle Curve", &[p(0.0, 0.0), json!(2.0)]).remove(0);
        let area = num(&run("Curve Area", &[circle])[0]).unwrap();
        assert!((area - 4.0 * std::f64::consts::PI).abs() < 1e-9);
        assert_eq!(run("Circle Curve", &[p(0.0, 0.0), json!(0.0)])[0], Value::Null);
    }

    #[test]
    fn offsets_go_left_for_positive_distances() {
        let line = run("Line Curve", &[p(0.0, 0.0), p(10.0, 0.0)]).remove(0);
        let left = run("Offset Curve", &[line.clone(), json!(2.0)]).remove(0);
        assert_eq!(left["start"], json!([0.0, 2.0]));
        let right = run("Offset Curve", &[line, json!(-2.0)]).remove(0);
        assert_eq!(right["end"], json!([10.0, -2.0]));

        // A counter-clockwise square grows outward to the right of travel.
        let square = run(
            "Polyline Curve",
            &[json!([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.0, 10.0)]), json!(true)],
        )
        .remove(0);
        let outer = run("Offset Curve", &[square.clone(), json!(-1.0)]).remove(0);
        let inner = run("Offset Curve", &[square, json!(1.0)]).remove(0);
        let area = |value: &Value| num(&run("Curve Area", &[value.clone()])[0]).unwrap();
        assert!(area(&outer) > 100.0 && area(&inner) < 100.0, "{outer} {inner}");
        assert!((area(&inner) - 64.0).abs() < 1e-6);
    }

    #[test]
    fn a_polyline_needs_two_points() {
        assert_eq!(run("Polyline Curve", &[json!([p(0.0, 0.0)]), json!(false)])[0], Value::Null);
    }

    fn length_of(value: &Value) -> f64 {
        num(&run("Curve Length", &[value.clone()])[0]).unwrap()
    }

    #[test]
    fn constructors_make_the_shapes_they_name() {
        let rectangle = run("Rectangle", &[p(1.0, 1.0), json!(4.0), json!(3.0)]).remove(0);
        assert!((length_of(&rectangle) - 14.0).abs() < 1e-9);
        let hexagon = run("Polygon", &[p(0.0, 0.0), json!(1.0), json!(6.0)]).remove(0);
        assert!((length_of(&hexagon) - 6.0).abs() < 1e-9);
        let arc = run("Arc 3 Points", &[p(1.0, 0.0), p(0.0, 1.0), p(-1.0, 0.0)]).remove(0);
        assert!((length_of(&arc) - std::f64::consts::PI).abs() < 1e-9);
        let line = run("Line By Direction", &[p(0.0, 0.0), p(0.0, 2.0), json!(5.0)]).remove(0);
        assert_eq!(line["end"], json!([0.0, 5.0]));
    }

    #[test]
    fn transforms_carry_points_curves_and_lists() {
        let line = run("Line Curve", &[p(0.0, 0.0), p(1.0, 0.0)]).remove(0);
        let moved = run("Move", &[json!([line.clone(), p(1.0, 1.0)]), point_value([0.0, 0.0, 2.0])]).remove(0);
        assert!(moved[0]["plane"].is_object(), "raised to z 2: {moved}");
        assert_eq!(point(&moved[1]), Some([1.0, 1.0, 2.0]));

        let copies = run("Linear Array", &[line.clone(), p(0.0, 3.0), json!(4.0)]).remove(0);
        assert_eq!(copies.as_array().map(Vec::len), Some(4));
        assert_eq!(copies[3]["start"], json!([0.0, 9.0]));

        let ring = run("Polar Array", &[p(5.0, 0.0), p(0.0, 0.0), json!(4.0), json!(360.0)]).remove(0);
        let last = point(&ring[3]).unwrap();
        assert!((last[0]).abs() < 1e-9 && (last[1] + 5.0).abs() < 1e-9, "{ring}");

        // Posts along a path, each turned to face along it.
        let path = run("Line Curve", &[p(0.0, 0.0), p(0.0, 10.0)]).remove(0);
        let posts = run("Curve Array", &[p(1.0, 0.0), path, json!(3.0)]).remove(0);
        let mid = point(&posts[1]).unwrap();
        assert!((mid[0]).abs() < 1e-9 && (mid[1] - 6.0).abs() < 1e-9, "{posts}");
    }

    #[test]
    fn planes_and_closest_points() {
        let wall = run("Plane", &[p(0.0, 0.0), p(1.0, 0.0), point_value([0.0, 0.0, 1.0])]).remove(0);
        let parts = run("Plane Components", &[wall]);
        assert_eq!(point(&parts[3]), Some([0.0, -1.0, 0.0]));
        let line = run("Line Curve", &[p(0.0, 0.0), p(10.0, 0.0)]).remove(0);
        let near = run("Closest Point", &[line, p(4.0, 3.0)]);
        assert_eq!(point(&near[0]), Some([4.0, 0.0, 0.0]));
        assert_eq!(near[2], json!(3.0));
    }

    #[test]
    fn joined_and_filleted_outlines() {
        let line = |a: (f64, f64), b: (f64, f64)| run("Line Curve", &[p(a.0, a.1), p(b.0, b.1)]).remove(0);
        let joined = run(
            "Join Curves",
            &[json!([line((0.0, 0.0), (4.0, 0.0)), line((4.0, 4.0), (4.0, 0.0)), line((4.0, 4.0), (0.0, 0.0))])],
        )
        .remove(0);
        assert_eq!(joined.as_array().map(Vec::len), Some(1));
        assert_eq!(joined[0]["closed"], true);
        let pieces = run("Explode Curve", &[joined[0].clone()]).remove(0);
        assert_eq!(pieces.as_array().map(Vec::len), Some(3));
        let round = run("Fillet Corners", &[joined[0].clone(), json!(0.5)]).remove(0);
        assert!(length_of(&round) < length_of(&joined[0]));
    }

    #[test]
    fn nested_lists_reshape() {
        let items = json!([1, 2, 3]);
        assert_eq!(run("Graft", &[items.clone()])[0], json!([[1], [2], [3]]));
        assert_eq!(run("Flatten Once", &[json!([[1, [2]], [3]])])[0], json!([1, [2], 3]));
        assert_eq!(run("List Depth", &[json!([[1], [2, [3]]])])[0], json!(3.0));
        let crossed = run("Cross Reference", &[json!(["a", "b"]), json!([1, 2])]);
        assert_eq!(crossed[0], json!(["a", "a", "b", "b"]));
        assert_eq!(crossed[1], json!([1, 2, 1, 2]));
        assert_eq!(run("Weave", &[json!([1, 3, 5]), json!([2, 4])])[0], json!([1, 2, 3, 4, 5]));
        assert_eq!(run("Shift List", &[items.clone(), json!(1), json!(true)])[0], json!([2, 3, 1]));
        assert_eq!(run("Shift List", &[items.clone(), json!(1), json!(false)])[0], json!([2, 3]));
        assert_eq!(run("Cull Pattern", &[json!([1, 2, 3, 4]), json!([true, false])])[0], json!([1, 3]));
    }
}
