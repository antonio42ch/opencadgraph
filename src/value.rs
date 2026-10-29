//! Port values.
//!
//! A value is plain JSON: numbers, text, booleans, `null`, lists (arrays),
//! points and vectors (`{"x", "y", "z"}` objects), curves (see [`curve`])
//! and planes (see [`plane`]). Object ports carry whatever the host puts
//! there. Every node reads its inputs through the lenient coercions below,
//! so a property shown as text (`"10.0000"`) feeds a number input.

use kernel::geom2d::polyline::{Polyline, PolylineVertex};
use kernel::space::{PlanarCurve, Plane};
use serde_json::{json, Value};

/// A number, from a number, a numeric string or a boolean.
pub fn num(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// A boolean, from a boolean, a number (non-zero) or `true`/`yes`/`1` text.
pub fn boolean(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|x| x != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Display text: strings as they are, `null` empty, whole numbers without a
/// decimal point, anything else as JSON.
pub fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Number(n) => match n.as_f64() {
            Some(x) if x.fract() == 0.0 && x.abs() < 1e15 => format!("{}", x as i64),
            _ => n.to_string(),
        },
        other => other.to_string(),
    }
}

/// A number value; NaN and infinities become `null`.
pub fn number(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// A point, from `{"x", "y"[, "z"]}`.
pub fn point(value: &Value) -> Option<[f64; 3]> {
    let object = value.as_object()?;
    let axis = |key: &str| object.get(key).and_then(num);
    Some([axis("x")?, axis("y")?, axis("z").unwrap_or(0.0)])
}

pub fn point_value(p: [f64; 3]) -> Value {
    json!({ "x": number(p[0]), "y": number(p[1]), "z": number(p[2]) })
}

/// The items of a list; a single value is a one-item list, `null` none.
pub fn list(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items.clone(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    }
}

/// Runs `f` once per element when a `scalar` input receives a list.
///
/// Lists on scalar inputs are walked in step, stopping at the shortest; other
/// inputs are passed whole to every call. Nested lists recurse, so a list of
/// lists maps element by element too. Returns `outputs` values, each a list
/// when lacing happened.
pub fn lace(
    inputs: &[Value],
    scalar: &[bool],
    outputs: usize,
    f: &mut dyn FnMut(&[Value]) -> Vec<Value>,
) -> Vec<Value> {
    let shortest = inputs
        .iter()
        .zip(scalar)
        .filter_map(|(value, scalar)| match value {
            Value::Array(items) if *scalar => Some(items.len()),
            _ => None,
        })
        .min();
    let Some(len) = shortest else {
        let mut result = f(inputs);
        result.resize(outputs, Value::Null);
        return result;
    };
    let mut columns = vec![Vec::with_capacity(len); outputs];
    for k in 0..len {
        let row: Vec<Value> = inputs
            .iter()
            .zip(scalar)
            .map(|(value, scalar)| match value {
                Value::Array(items) if *scalar => items[k].clone(),
                other => other.clone(),
            })
            .collect();
        for (column, value) in columns.iter_mut().zip(lace(&row, scalar, outputs, f)) {
            column.push(value);
        }
    }
    columns.into_iter().map(Value::Array).collect()
}

/// A curve value: `{"curve": kind, …}` with its shape in plane coordinates
/// and an optional `plane` (`{"origin", "x_axis", "y_axis"}`, each `[x, y,
/// z]`; the world XY plane when absent). Kinds and their fields, angles in
/// degrees:
///
/// - `line`: `start`, `end` (`[x, y]`)
/// - `circle`: `center`, `radius`
/// - `arc`: `center`, `radius`, `from`, `to` (counter-clockwise)
/// - `polyline`: `points`, optional `bulges` (one per point), `closed`
/// - `ray`: `origin`, `direction`; `xline`: `base`, `direction`
///
/// Graph nodes make and transform curves as values like this; only a node
/// that draws them puts them in the drawing.
pub fn curve(value: &Value) -> Option<PlanarCurve> {
    use kernel::geom2d::{Arc, Circle, Curve, Line, Ray, XLine};
    let object = value.as_object()?;
    let pair = |key: &str| -> Option<[f64; 2]> {
        let items = object.get(key)?.as_array()?;
        Some([num(items.first()?)?, num(items.get(1)?)?])
    };
    let scalar = |key: &str| object.get(key).and_then(num);
    let shape = match object.get("curve")?.as_str()? {
        "line" => Curve::Line(Line {
            start: pair("start")?,
            end: pair("end")?,
        }),
        "circle" => Curve::Circle(Circle {
            centre: pair("center")?,
            radius: scalar("radius").filter(|r| *r > 0.0)?,
        }),
        "arc" => Curve::Arc(Arc {
            centre: pair("center")?,
            radius: scalar("radius").filter(|r| *r > 0.0)?,
            start_angle: scalar("from")?.to_radians(),
            end_angle: scalar("to")?.to_radians(),
        }),
        "polyline" => {
            let points = object.get("points")?.as_array()?;
            let bulges = object.get("bulges").and_then(Value::as_array);
            let vertices = points
                .iter()
                .enumerate()
                .map(|(k, point)| {
                    let items = point.as_array()?;
                    Some(PolylineVertex {
                        position: [num(items.first()?)?, num(items.get(1)?)?],
                        bulge: bulges
                            .and_then(|bulges| bulges.get(k))
                            .and_then(num)
                            .unwrap_or(0.0),
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            if vertices.len() < 2 {
                return None;
            }
            Curve::Polyline(Polyline {
                vertices,
                closed: object.get("closed").and_then(boolean).unwrap_or(false),
            })
        }
        "ray" => Curve::Ray(Ray {
            origin: pair("origin")?,
            direction: pair("direction")?,
        }),
        "xline" => Curve::XLine(XLine {
            base: pair("base")?,
            direction: pair("direction")?,
        }),
        _ => return None,
    };
    let plane = match object.get("plane") {
        Some(plane) => frame(plane)?,
        None => Plane::XY,
    };
    Some(PlanarCurve::new(plane, shape))
}

/// The value of `curve`, as [`curve`] reads it; `None` for ellipses and
/// splines, which have no value form yet.
pub fn curve_value(curve: &PlanarCurve) -> Option<Value> {
    use kernel::geom2d::Curve;
    let pair = |p: [f64; 2]| json!([number(p[0]), number(p[1])]);
    let mut value = match &curve.curve {
        Curve::Line(line) => {
            json!({ "curve": "line", "start": pair(line.start), "end": pair(line.end) })
        }
        Curve::Circle(circle) => json!({
            "curve": "circle",
            "center": pair(circle.centre),
            "radius": number(circle.radius),
        }),
        Curve::Arc(arc) => json!({
            "curve": "arc",
            "center": pair(arc.centre),
            "radius": number(arc.radius),
            "from": number(arc.start_angle.to_degrees()),
            "to": number(arc.end_angle.to_degrees()),
        }),
        Curve::Polyline(polyline) => {
            let mut value = json!({
                "curve": "polyline",
                "points": polyline.vertices.iter().map(|v| pair(v.position)).collect::<Vec<_>>(),
                "closed": polyline.closed,
            });
            if polyline.vertices.iter().any(|v| v.bulge != 0.0) {
                value["bulges"] = json!(polyline
                    .vertices
                    .iter()
                    .map(|v| number(v.bulge))
                    .collect::<Vec<_>>());
            }
            value
        }
        Curve::Ray(ray) => {
            json!({ "curve": "ray", "origin": pair(ray.origin), "direction": pair(ray.direction) })
        }
        Curve::XLine(line) => {
            json!({ "curve": "xline", "base": pair(line.base), "direction": pair(line.direction) })
        }
        Curve::Ellipse(_) | Curve::Nurbs(_) => return None,
    };
    if curve.plane != Plane::XY {
        value["plane"] = frame_value(&curve.plane);
    }
    Some(value)
}

/// `{"origin", "x_axis", "y_axis"}`, each `[x, y, z]`.
fn frame(value: &Value) -> Option<Plane> {
    let axis = |key: &str| -> Option<[f64; 3]> {
        let items = value.get(key)?.as_array()?;
        Some([
            num(items.first()?)?,
            num(items.get(1)?)?,
            num(items.get(2)?)?,
        ])
    };
    Some(Plane {
        origin: axis("origin")?,
        x_axis: axis("x_axis")?,
        y_axis: axis("y_axis")?,
    })
}

fn frame_value(plane: &Plane) -> Value {
    let triple = |p: [f64; 3]| json!([number(p[0]), number(p[1]), number(p[2])]);
    json!({
        "origin": triple(plane.origin),
        "x_axis": triple(plane.x_axis),
        "y_axis": triple(plane.y_axis),
    })
}

/// A plane value: `{"plane": {"origin", "x_axis", "y_axis"}}`, the frame
/// geometry is placed in and oriented between.
pub fn plane(value: &Value) -> Option<Plane> {
    frame(value.get("plane")?)
}

pub fn plane_value(plane: &Plane) -> Value {
    json!({ "plane": frame_value(plane) })
}

/// The plane parallel to world XY through height `z`, which curves built
/// from points lie on.
pub fn plane_at(z: f64) -> Plane {
    Plane {
        origin: [0.0, 0.0, z],
        ..Plane::XY
    }
}

/// A table's rows: a list of lists, or text with a row per line and cells
/// split by tabs, semicolons or commas (the first of them the text holds).
/// Cells that read as numbers become numbers.
pub fn table(value: &Value) -> Vec<Vec<Value>> {
    let cell = |item: &Value| match item {
        Value::String(text) => text
            .trim()
            .parse::<f64>()
            .map_or_else(|_| Value::String(text.trim().to_owned()), number),
        other => other.clone(),
    };
    match value {
        Value::Array(rows) => rows
            .iter()
            .map(|row| match row {
                Value::Array(cells) => cells.iter().map(cell).collect(),
                other => vec![cell(other)],
            })
            .collect(),
        Value::String(text) => {
            let separator = ['\t', ';', ','].into_iter().find(|c| text.contains(*c)).unwrap_or(',');
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| line.split(separator).map(|item| cell(&Value::String(item.to_owned()))).collect())
                .collect()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_round_trip_through_their_values() {
        for value in [
            json!({"curve":"line","start":[0.0,0.0],"end":[10.0,0.0]}),
            json!({"curve":"circle","center":[1.0,2.0],"radius":3.0}),
            json!({"curve":"arc","center":[0.0,0.0],"radius":2.0,"from":0.0,"to":90.0}),
            json!({"curve":"polyline","points":[[0.0,0.0],[5.0,0.0],[5.0,5.0]],"closed":true}),
            json!({"curve":"line","start":[0.0,0.0],"end":[1.0,0.0],"plane":{"origin":[0.0,0.0,4.0],"x_axis":[1.0,0.0,0.0],"y_axis":[0.0,1.0,0.0]}}),
        ] {
            let curve = curve(&value).unwrap_or_else(|| panic!("{value}"));
            assert_eq!(curve_value(&curve).unwrap(), value);
        }
        let arc =
            curve(&json!({"curve":"arc","center":[0,0],"radius":2,"from":0,"to":90})).unwrap();
        assert!((arc.length() - std::f64::consts::PI).abs() < 1e-9);
        assert!(curve(&json!({"curve":"circle","center":[0,0],"radius":-1})).is_none());
        assert!(curve(&json!({"x":1,"y":2})).is_none());
    }
}
