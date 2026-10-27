//! Vectors, planes, transforms and curve edits over geometry values.
//!
//! Every transform here works the same way on every kind of value: it is a
//! map of space (where points go, and where directions go), applied to a
//! point directly and to a curve or plane through its frame. A curve's
//! frame comes out of the map and is then *settled*: when it is still
//! parallel to world XY the map folds into a 2D transform of the shape on an
//! XY plane, so a turned or mirrored circle stays a circle a drawing can
//! hold; otherwise the curve keeps the new frame.

use kernel::geom2d::polyline::{Polyline, PolylineVertex};
use kernel::geom2d::vec::Vec2;
use kernel::geom2d::{Arc, Circle, Curve, Line, Transform};
use kernel::space::{PlanarCurve, Plane};
use serde_json::Value;

use crate::value::{curve, curve_value, plane, plane_at, plane_value, point, point_value};

pub type V3 = [f64; 3];

const EPSILON: f64 = 1e-9;

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: V3, k: f64) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// `a` at length one, or `None` when it has no direction.
pub fn unit(a: V3) -> Option<V3> {
    let l = length(a);
    (l > EPSILON).then(|| scale(a, 1.0 / l))
}

/// The plane through `origin` with `x` as its X axis and `y` turned
/// square to it in their common plane; `None` when they are parallel.
pub fn frame(origin: V3, x: V3, y: V3) -> Option<Plane> {
    let x_axis = unit(x)?;
    let y_axis = unit(sub(y, scale(x_axis, dot(y, x_axis))))?;
    Some(Plane {
        origin,
        x_axis,
        y_axis,
    })
}

/// A plane's unit axes and normal.
fn axes(plane: &Plane) -> Option<(V3, V3, V3)> {
    let frame = frame(plane.origin, plane.x_axis, plane.y_axis)?;
    Some((
        frame.x_axis,
        frame.y_axis,
        cross(frame.x_axis, frame.y_axis),
    ))
}

/// A map of space: where points go and where directions go.
pub struct Map<'a> {
    pub point: &'a dyn Fn(V3) -> V3,
    pub vector: &'a dyn Fn(V3) -> V3,
}

/// `value` carried through `map`: points, curves, planes and lists of them.
/// Anything else, and any curve the map cannot carry, becomes `null`.
pub fn map_value(value: &Value, map: &Map) -> Value {
    if let Value::Array(items) = value {
        return Value::Array(items.iter().map(|item| map_value(item, map)).collect());
    }
    if let Some(shape) = curve(value) {
        return map_curve(&shape, map)
            .and_then(|shape| curve_value(&shape))
            .unwrap_or(Value::Null);
    }
    if let Some(p) = point(value) {
        return point_value((map.point)(p));
    }
    if let Some(frame_in) = plane(value) {
        return frame(
            (map.point)(frame_in.origin),
            (map.vector)(frame_in.x_axis),
            (map.vector)(frame_in.y_axis),
        )
        .map_or(Value::Null, |frame| plane_value(&frame));
    }
    Value::Null
}

pub fn map_curve(shape: &PlanarCurve, map: &Map) -> Option<PlanarCurve> {
    settle(
        (map.point)(shape.plane.origin),
        (map.vector)(shape.plane.x_axis),
        (map.vector)(shape.plane.y_axis),
        &shape.curve,
    )
}

/// `curve`, whose plane coordinates `(u, v)` now land at `origin + u x + v
/// y`, back on a plane with unit axes. Square axes are required: a shear has
/// no curve to give.
fn settle(origin: V3, x: V3, y: V3, shape: &Curve) -> Option<PlanarCurve> {
    let (kx, ky) = (length(x), length(y));
    if kx < EPSILON || ky < EPSILON || dot(x, y).abs() > EPSILON * kx * ky {
        return None;
    }
    let normal = cross(x, y);
    if normal[0].abs() <= EPSILON * kx * ky && normal[1].abs() <= EPSILON * kx * ky {
        // Still parallel to world XY: fold the whole map into the shape.
        let by = Transform {
            x_axis: Vec2::new(x[0], x[1]),
            y_axis: Vec2::new(y[0], y[1]),
            origin: Vec2::new(origin[0], origin[1]),
        };
        return Some(PlanarCurve::new(
            plane_at(origin[2]),
            shape.transformed(&by)?,
        ));
    }
    let shape = if (kx - 1.0).abs() < EPSILON && (ky - 1.0).abs() < EPSILON {
        shape.clone()
    } else {
        shape.transformed(&Transform::scale(kx, ky))?
    };
    Some(PlanarCurve::new(
        Plane {
            origin,
            x_axis: scale(x, 1.0 / kx),
            y_axis: scale(y, 1.0 / ky),
        },
        shape,
    ))
}

/// Turning by `degrees` about the vertical through `center`.
pub fn rotation(center: V3, degrees: f64) -> (impl Fn(V3) -> V3, impl Fn(V3) -> V3) {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let turn = move |v: V3| [v[0] * cos - v[1] * sin, v[0] * sin + v[1] * cos, v[2]];
    (move |p: V3| add(center, turn(sub(p, center))), turn)
}

/// Reflection in the vertical plane through `a` and `b`.
pub fn mirror(a: V3, b: V3) -> Option<(impl Fn(V3) -> V3, impl Fn(V3) -> V3)> {
    let along = unit([b[0] - a[0], b[1] - a[1], 0.0])?;
    let reflect = move |v: V3| {
        let d = v[0] * along[0] + v[1] * along[1];
        [2.0 * d * along[0] - v[0], 2.0 * d * along[1] - v[1], v[2]]
    };
    Some((move |p: V3| add(a, reflect(sub(p, a))), reflect))
}

/// Carrying geometry from frame `from` to frame `to`: what sat at some
/// coordinates in `from` sits at the same coordinates in `to`.
pub fn orientation(from: &Plane, to: &Plane) -> Option<(impl Fn(V3) -> V3, impl Fn(V3) -> V3)> {
    let (fx, fy, fz) = axes(from)?;
    let (tx, ty, tz) = axes(to)?;
    let vector = move |v: V3| {
        add(
            add(scale(tx, dot(v, fx)), scale(ty, dot(v, fy))),
            scale(tz, dot(v, fz)),
        )
    };
    let (from_origin, to_origin) = (from.origin, to.origin);
    Some((
        move |p: V3| add(to_origin, vector(sub(p, from_origin))),
        vector,
    ))
}

/// The frame at parameter `t` of `shape`: its point, its direction of travel
/// as X, and Y square to that in the curve's plane.
pub fn curve_frame(shape: &PlanarCurve, t: f64) -> Option<Plane> {
    let normal = shape.normal()?;
    let x = unit(shape.tangent_at(t))?;
    frame(shape.point_at(t), x, cross(normal, x))
}

/// One straight or circular piece of a chain, in plane coordinates.
#[derive(Clone, Copy, Debug)]
struct Piece {
    start: [f64; 2],
    end: [f64; 2],
    bulge: f64,
}

impl Piece {
    fn reversed(self) -> Self {
        Self {
            start: self.end,
            end: self.start,
            bulge: -self.bulge,
        }
    }
}

fn near(a: [f64; 2], b: [f64; 2], tolerance: f64) -> bool {
    (a[0] - b[0]).hypot(a[1] - b[1]) <= tolerance
}

/// A curve as the pieces a polyline is made of; `None` for circles,
/// ellipses, splines and unbounded curves.
fn pieces(shape: &Curve) -> Option<Vec<Piece>> {
    match shape {
        Curve::Line(line) => Some(vec![Piece {
            start: line.start,
            end: line.end,
            bulge: 0.0,
        }]),
        Curve::Arc(arc) => Some(vec![Piece {
            start: shape.point_at(0.0),
            end: shape.point_at(1.0),
            bulge: (arc.sweep() / 4.0).tan(),
        }]),
        Curve::Polyline(polyline) => {
            let count = polyline.vertices.len();
            let segments = if polyline.closed { count } else { count - 1 };
            Some(
                (0..segments)
                    .map(|k| Piece {
                        start: polyline.vertices[k].position,
                        end: polyline.vertices[(k + 1) % count].position,
                        bulge: polyline.vertices[k].bulge,
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// The polyline through `chain`, closed when it ends where it began.
fn polyline(chain: &[Piece], tolerance: f64) -> Curve {
    let closed = chain.len() > 1 && near(chain[0].start, chain[chain.len() - 1].end, tolerance);
    let mut vertices: Vec<PolylineVertex> = chain
        .iter()
        .map(|piece| PolylineVertex {
            position: piece.start,
            bulge: piece.bulge,
        })
        .collect();
    if !closed {
        vertices.push(PolylineVertex::straight(chain[chain.len() - 1].end));
    }
    Curve::Polyline(Polyline { vertices, closed })
}

/// `shapes` chained end to end into polylines, turning pieces round where
/// that joins them. Curves no polyline can hold (circles, splines) and
/// curves on another plane than the first are passed through as they are.
pub fn join(shapes: &[PlanarCurve], tolerance: f64) -> Vec<PlanarCurve> {
    let Some(first) = shapes.first() else {
        return Vec::new();
    };
    let plane = first.plane;
    let mut loose: Vec<Piece> = Vec::new();
    let mut passed: Vec<PlanarCurve> = Vec::new();
    for shape in shapes {
        match pieces(&shape.curve) {
            Some(found) if shape.plane == plane => loose.extend(found),
            _ => passed.push(shape.clone()),
        }
    }
    let mut chains: Vec<Vec<Piece>> = Vec::new();
    while !loose.is_empty() {
        let mut chain = vec![loose.remove(0)];
        loop {
            let end = chain[chain.len() - 1].end;
            let start = chain[0].start;
            if chain.len() > 1 && near(end, start, tolerance) {
                break;
            }
            if let Some(k) = loose
                .iter()
                .position(|piece| near(piece.start, end, tolerance))
            {
                chain.push(loose.remove(k));
            } else if let Some(k) = loose
                .iter()
                .position(|piece| near(piece.end, end, tolerance))
            {
                chain.push(loose.remove(k).reversed());
            } else if let Some(k) = loose
                .iter()
                .position(|piece| near(piece.end, start, tolerance))
            {
                chain.insert(0, loose.remove(k));
            } else if let Some(k) = loose
                .iter()
                .position(|piece| near(piece.start, start, tolerance))
            {
                chain.insert(0, loose.remove(k).reversed());
            } else {
                break;
            }
        }
        chains.push(chain);
    }
    chains
        .iter()
        .map(|chain| PlanarCurve::new(plane, polyline(chain, tolerance)))
        .chain(passed)
        .collect()
}

/// A piece as a curve of its own, to measure and walk along.
fn piece_curve(piece: &Piece) -> Curve {
    Curve::Polyline(Polyline {
        vertices: vec![
            PolylineVertex {
                position: piece.start,
                bulge: piece.bulge,
            },
            PolylineVertex::straight(piece.end),
        ],
        closed: false,
    })
}

/// The part of `piece` from fraction `u0` to `u1` of its length.
fn piece_part(piece: &Piece, u0: f64, u1: f64) -> Piece {
    let shape = piece_curve(piece);
    let length = shape.length();
    let sweep = 4.0 * piece.bulge.atan();
    Piece {
        start: shape.point_at_distance(length * u0),
        end: shape.point_at_distance(length * u1),
        bulge: (sweep * (u1 - u0) / 4.0).tan(),
    }
}

/// The part of `shape` between arc lengths `from` and `to` along it, in
/// either order, clamped to the curve; `None` when nothing is left.
/// Circles start at angle 0; unbounded curves, ellipses and splines give
/// `None`.
pub fn sub_curve(shape: &Curve, from: f64, to: f64) -> Option<Curve> {
    let total = shape.length();
    if !total.is_finite() {
        return None;
    }
    let (from, to) = (from.min(to).max(0.0), from.max(to).min(total));
    if to - from <= EPSILON {
        return None;
    }
    match shape {
        Curve::Line(_) => Some(Curve::Line(Line {
            start: shape.point_at_distance(from),
            end: shape.point_at_distance(to),
        })),
        Curve::Arc(arc) => Some(Curve::Arc(Arc {
            start_angle: arc.start_angle + from / arc.radius,
            end_angle: arc.start_angle + to / arc.radius,
            ..*arc
        })),
        Curve::Circle(circle) => Some(Curve::Arc(Arc {
            centre: circle.centre,
            radius: circle.radius,
            start_angle: from / circle.radius,
            end_angle: to / circle.radius,
        })),
        _ => {
            let mut chain = Vec::new();
            let mut start = 0.0;
            for piece in pieces(shape)? {
                let length = piece_curve(&piece).length();
                let end = start + length;
                let (a, b) = (from.max(start), to.min(end));
                if length > EPSILON && b - a > EPSILON {
                    chain.push(piece_part(&piece, (a - start) / length, (b - start) / length));
                }
                start = end;
            }
            match chain.as_slice() {
                [] => None,
                [piece] if piece.bulge == 0.0 => Some(Curve::Line(Line {
                    start: piece.start,
                    end: piece.end,
                })),
                _ => Some(polyline(&chain, EPSILON)),
            }
        }
    }
}

/// The closed outline between `left` and `right`, two parallels of one
/// open curve running its way: along `left`, across its end, back along
/// `right` and across its start.
pub fn outline(left: &Curve, right: &Curve) -> Option<Curve> {
    let left = pieces(left)?;
    let right = pieces(right)?;
    let (first, last) = (left.first()?, left.last()?);
    let (back_first, back_last) = (right.first()?, right.last()?);
    let mut chain = left.clone();
    chain.push(Piece {
        start: last.end,
        end: back_last.end,
        bulge: 0.0,
    });
    chain.extend(right.iter().rev().map(|piece| piece.reversed()));
    chain.push(Piece {
        start: back_first.start,
        end: first.start,
        bulge: 0.0,
    });
    Some(polyline(&chain, EPSILON))
}

/// `shape` run the other way. A circle has no stored direction and comes
/// back as it is; an arc, which always runs counter-clockwise, comes back as
/// the one-segment polyline that runs it clockwise.
pub fn reverse(shape: &Curve) -> Option<Curve> {
    match shape {
        Curve::Circle(_) => Some(shape.clone()),
        Curve::Line(line) => Some(Curve::Line(Line {
            start: line.end,
            end: line.start,
        })),
        Curve::Arc(_) | Curve::Polyline(_) => {
            let mut chain: Vec<Piece> = pieces(shape)?
                .into_iter()
                .rev()
                .map(Piece::reversed)
                .collect();
            let closed = matches!(shape, Curve::Polyline(polyline) if polyline.closed);
            if closed {
                // Keep the start where it was.
                chain.rotate_right(1);
            }
            Some(polyline(&chain, 0.0))
        }
        _ => None,
    }
}

/// A polyline closed back to its start. Circles are closed already; a line
/// or an arc closed on itself would be degenerate, so they give `None`.
pub fn close(shape: &Curve) -> Option<Curve> {
    match shape {
        Curve::Polyline(polyline) => Some(Curve::Polyline(Polyline {
            vertices: polyline.vertices.clone(),
            closed: true,
        })),
        Curve::Circle(_) => Some(shape.clone()),
        _ => None,
    }
}

/// `shape` made longer by `start` before its start and `end` past its end
/// (negative values shorten it): lines along themselves, arcs around their
/// circle, open polylines through a straight first or last segment.
pub fn extend(shape: &Curve, start: f64, end: f64) -> Option<Curve> {
    let push = |from: [f64; 2], to: [f64; 2], by: f64| -> Option<[f64; 2]> {
        let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
        let l = dx.hypot(dy);
        (l > EPSILON).then(|| [to[0] + dx / l * by, to[1] + dy / l * by])
    };
    match shape {
        Curve::Line(line) => Some(Curve::Line(Line {
            start: push(line.end, line.start, start)?,
            end: push(line.start, line.end, end)?,
        })),
        Curve::Arc(arc) => Some(Curve::Arc(Arc {
            start_angle: arc.start_angle - start / arc.radius,
            end_angle: arc.end_angle + end / arc.radius,
            ..*arc
        })),
        Curve::Polyline(polyline) if !polyline.closed && polyline.vertices.len() >= 2 => {
            let mut vertices = polyline.vertices.clone();
            let n = vertices.len();
            if start != 0.0 {
                if vertices[0].bulge != 0.0 {
                    return None;
                }
                vertices[0].position = push(vertices[1].position, vertices[0].position, start)?;
            }
            if end != 0.0 {
                if vertices[n - 2].bulge != 0.0 {
                    return None;
                }
                vertices[n - 1].position =
                    push(vertices[n - 2].position, vertices[n - 1].position, end)?;
            }
            Some(Curve::Polyline(Polyline {
                vertices,
                closed: false,
            }))
        }
        _ => None,
    }
}

/// A polyline with every corner between two straight segments rounded at
/// `radius`. Corners too tight for the radius, and corners touching an arc,
/// stay sharp.
pub fn fillet(shape: &Curve, radius: f64) -> Option<Curve> {
    let Curve::Polyline(source) = shape else {
        return None;
    };
    if radius <= 0.0 {
        return Some(shape.clone());
    }
    let n = source.vertices.len();
    let mut vertices: Vec<PolylineVertex> = Vec::with_capacity(2 * n);
    for k in 0..n {
        let vertex = source.vertices[k];
        let interior = source.closed || (k > 0 && k + 1 < n);
        let prev = source.vertices[(k + n - 1) % n];
        let next = source.vertices[(k + 1) % n];
        let straight = prev.bulge == 0.0 && vertex.bulge == 0.0;
        let rounded = (interior && straight)
            .then(|| corner(prev.position, vertex.position, next.position, radius))
            .flatten();
        match rounded {
            Some((into, out, bulge)) => {
                vertices.push(PolylineVertex {
                    position: into,
                    bulge,
                });
                vertices.push(PolylineVertex::straight(out));
            }
            None => vertices.push(vertex),
        }
    }
    Some(Curve::Polyline(Polyline {
        vertices,
        closed: source.closed,
    }))
}

/// The tangent points and bulge of a `radius` arc rounding the corner at
/// `at` between the segments from `from` and to `to`.
fn corner(
    from: [f64; 2],
    at: [f64; 2],
    to: [f64; 2],
    radius: f64,
) -> Option<([f64; 2], [f64; 2], f64)> {
    let a = [from[0] - at[0], from[1] - at[1]];
    let b = [to[0] - at[0], to[1] - at[1]];
    let (la, lb) = (a[0].hypot(a[1]), b[0].hypot(b[1]));
    if la < EPSILON || lb < EPSILON {
        return None;
    }
    let (ua, ub) = ([a[0] / la, a[1] / la], [b[0] / lb, b[1] / lb]);
    let between = (ua[0] * ub[0] + ua[1] * ub[1]).clamp(-1.0, 1.0).acos();
    if between < 1e-6 || (std::f64::consts::PI - between) < 1e-6 {
        return None;
    }
    let reach = radius / (between / 2.0).tan();
    if reach > la || reach > lb {
        return None;
    }
    let into = [at[0] + ua[0] * reach, at[1] + ua[1] * reach];
    let out = [at[0] + ub[0] * reach, at[1] + ub[1] * reach];
    // Turning left (counter-clockwise) bulges positive.
    let turn = (at[0] - from[0]) * (to[1] - at[1]) - (at[1] - from[1]) * (to[0] - at[0]);
    let bulge = ((std::f64::consts::PI - between) / 4.0).tan() * turn.signum();
    Some((into, out, bulge))
}

/// The circle through three points, as its centre and radius in their XY.
pub fn circle_through(a: V3, b: V3, c: V3) -> Option<([f64; 2], f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < EPSILON {
        return None;
    }
    let sq = |p: V3| p[0] * p[0] + p[1] * p[1];
    let x = (sq(a) * (b[1] - c[1]) + sq(b) * (c[1] - a[1]) + sq(c) * (a[1] - b[1])) / d;
    let y = (sq(a) * (c[0] - b[0]) + sq(b) * (a[0] - c[0]) + sq(c) * (b[0] - a[0])) / d;
    Some(([x, y], (a[0] - x).hypot(a[1] - y)))
}

/// The arc from `a` through `b` to `c`.
pub fn arc_through(a: V3, b: V3, c: V3) -> Option<Curve> {
    let (centre, radius) = circle_through(a, b, c)?;
    let angle = |p: V3| (p[1] - centre[1]).atan2(p[0] - centre[0]);
    let (from, via, to) = (angle(a), angle(b), angle(c));
    let tau = std::f64::consts::TAU;
    let ccw = |start: f64, end: f64| (end - start).rem_euclid(tau);
    // Counter-clockwise from a to c when b lies on that way round,
    // otherwise counter-clockwise from c to a.
    let (start_angle, end_angle) = if ccw(from, via) <= ccw(from, to) {
        (from, to)
    } else {
        (to, from)
    };
    Some(Curve::Arc(Arc {
        centre,
        radius,
        start_angle,
        end_angle,
    }))
}

/// A closed polyline through `corners`.
pub fn closed_polyline(corners: &[[f64; 2]]) -> Curve {
    Curve::Polyline(Polyline {
        vertices: corners
            .iter()
            .map(|p| PolylineVertex::straight(*p))
            .collect(),
        closed: true,
    })
}

/// A circle as a curve, for the constructors.
pub fn circle(centre: [f64; 2], radius: f64) -> Curve {
    Curve::Circle(Circle { centre, radius })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flat(shape: Curve) -> PlanarCurve {
        PlanarCurve::new(Plane::XY, shape)
    }

    #[test]
    fn a_turned_or_mirrored_circle_stays_a_circle_on_xy() {
        let value = curve_value(&flat(circle([5.0, 0.0], 1.0))).unwrap();
        let (point, vector) = rotation([0.0; 3], 90.0);
        let turned = map_value(
            &value,
            &Map {
                point: &point,
                vector: &vector,
            },
        );
        assert_eq!(turned["curve"], "circle");
        assert!(turned.get("plane").is_none(), "{turned}");
        let centre = &turned["center"];
        assert!(
            centre[0].as_f64().unwrap().abs() < 1e-9
                && (centre[1].as_f64().unwrap() - 5.0).abs() < 1e-9
        );

        let (point, vector) = mirror([0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        let arc = curve_value(&flat(
            arc_through([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]).unwrap(),
        ))
        .unwrap();
        let mirrored = map_value(
            &arc,
            &Map {
                point: &point,
                vector: &vector,
            },
        );
        let mirrored = curve(&mirrored).unwrap();
        assert!((mirrored.length() - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn orienting_onto_a_wall_plane_keeps_the_shape() {
        let wall = frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let (point, vector) = orientation(&Plane::XY, &wall).unwrap();
        let square = curve_value(&flat(closed_polyline(&[
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 3.0],
            [0.0, 3.0],
        ])))
        .unwrap();
        let standing = curve(&map_value(
            &square,
            &Map {
                point: &point,
                vector: &vector,
            },
        ))
        .unwrap();
        assert!((standing.curve.enclosed_area().abs() - 6.0).abs() < 1e-9);
        let top = (0..=8)
            .map(|k| standing.point_at(k as f64 / 8.0)[2])
            .fold(0.0, f64::max);
        assert!((top - 3.0).abs() < 1e-9, "{top}");
        assert_eq!(
            map_value(
                &json!({"x": 1, "y": 2}),
                &Map {
                    point: &point,
                    vector: &vector
                }
            )["z"],
            2.0
        );
    }

    #[test]
    fn pieces_join_into_one_closed_polyline() {
        let line = |a: [f64; 2], b: [f64; 2]| flat(Curve::Line(Line { start: a, end: b }));
        // Out of order, and one of them backwards.
        let joined = join(
            &[
                line([10.0, 0.0], [10.0, 10.0]),
                line([0.0, 0.0], [10.0, 0.0]),
                line([0.0, 10.0], [10.0, 10.0]),
                line([0.0, 10.0], [0.0, 0.0]),
            ],
            1e-6,
        );
        assert_eq!(joined.len(), 1);
        assert!(joined[0].curve.is_closed());
        assert!((joined[0].curve.enclosed_area().abs() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn fillets_round_every_square_corner() {
        let square = closed_polyline(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        let round = fillet(&square, 1.0).unwrap();
        let lost = 100.0 - round.enclosed_area().abs();
        assert!((lost - (4.0 - std::f64::consts::PI)).abs() < 1e-6, "{lost}");
    }

    #[test]
    fn edits_keep_lengths_right() {
        let line = Curve::Line(Line {
            start: [0.0, 0.0],
            end: [4.0, 0.0],
        });
        let longer = extend(&line, 1.0, 2.0).unwrap();
        assert!((flat(longer).length() - 7.0).abs() < 1e-9);
        let back = reverse(&line).unwrap();
        assert_eq!(back.point_at(0.0), [4.0, 0.0]);
        let arc = arc_through([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]).unwrap();
        let reversed = flat(reverse(&arc).unwrap());
        assert!((reversed.length() - std::f64::consts::PI).abs() < 1e-9);
        assert!((reversed.point_at(0.0)[0] + 1.0).abs() < 1e-9);
    }
}
