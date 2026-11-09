Minimal compile-time stand-ins for the parts of the COMSOL Java API
(`com.comsol.model`) that `axionfem.comsol.write_java` emits. They let
`tests/test_comsol_export.py` check with `javac` that a generated `Model.java` is
valid Java and only calls methods of the expected shapes, without a COMSOL
installation. They do nothing at run time and are not the COMSOL API: with
COMSOL, compile with `comsolcompile` instead (docs/comsol_export.md).

The physics classes (`Physics*`, `Material*`, `Study*`, `SolverSequence*`, `Results`,
`EvaluationGroup*`, `EntitySelection`) cover what `--physics fullwave` emits. Their class
names are arbitrary (the generated code only chains calls); only the call shapes matter.
