# opencadgraph

A node graph engine for CAD applications.

Nodes are library operations (input, math, logic, list, text, geometry,
curve) or
host objects — drawing objects whose property rows are the node's ports. The
engine owns ordering, list lacing and value coercion; the application
implements [`Host`](src/lib.rs) to create, edit and read its objects and to
supply curves for the geometry nodes (built on
[opencadkernel](https://github.com/HakanSeven12/opencadkernel)).

Values are JSON: numbers, text, booleans, lists, points as
`{"x", "y", "z"}`, and curves as `{"curve": "line" | "circle" | "arc" |
"polyline" | "ray" | "xline", …}` (see `value::curve`). Curve nodes make and
offset curves as values, so intermediate geometry never has to be drawn; a
geometry node reads a curve from a value or, failing that, from the host. A
list reaching a scalar input runs the node once per element; a list reaching
an object property makes one object per element.

License: MPL-2.0
