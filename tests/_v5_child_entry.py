"""Test-only CLI child entry (Round-9 fix 2/3).

Children launched by the sharded/protocol tests execute through this entry so
that, BEFORE any experiment code runs, the harness verifies the child really
imports the synthetic world's own closure and the agreed small-window schema:

- the imported forecasting.full_v5.schema module comes from the world cwd,
- the effective SEGMENTS/FIT_CUTOFFS equal the windows the parent installed
  (no reliance on in-memory monkeypatch propagation),
- the recomputed SCHEMA_HASH equals the parent-side hash,
- numpy/scipy/pandas/lightgbm/yaml versions match the parent process.

On completion (or failure) the child dumps every module it actually imported
from the world (including src/dualfield) plus dependency versions to a
snapshot JSON, so evidence binds the true import closure, not assumptions.
"""
import hashlib
import json
import os
import runpy
import sys


def _verify(world, expected):
    problems = []
    import forecasting.full_v5.schema as schema
    schema_file = os.path.realpath(schema.__file__)
    if not schema_file.startswith(os.path.realpath(world) + os.sep):
        problems.append(f"schema imported from outside the world: {schema_file}")
    if {k: list(v) for k, v in schema.SEGMENTS.items()} != expected["segments"]:
        problems.append(f"effective SEGMENTS != agreed window: {dict(schema.SEGMENTS)}")
    if dict(schema.FIT_CUTOFFS) != expected["fit_cutoffs"]:
        problems.append(f"effective FIT_CUTOFFS != agreed: {dict(schema.FIT_CUTOFFS)}")
    if schema.SCHEMA_HASH != expected["schema_hash"]:
        problems.append(f"SCHEMA_HASH mismatch: {schema.SCHEMA_HASH} != {expected['schema_hash']}")
    for module_name, want in sorted(expected["dep_versions"].items()):
        got = __import__(module_name).__version__
        if got != want:
            problems.append(f"{module_name} {got} != parent {want}")
    return problems, schema, schema_file


def _dump_snapshot(path, world, dep_names):
    import forecasting.full_v5.schema as schema  # noqa: F401  (already imported)
    world_real = os.path.realpath(world)
    modules, deps = {}, {}
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        real = os.path.realpath(file)
        if real.startswith(world_real + os.sep):
            modules[name] = os.path.relpath(real, world_real)
    for dep in dep_names:
        deps[dep] = __import__(dep).__version__
    payload = dict(pid=os.getpid(), world=world_real, imported_from_world=modules,
                   dep_versions=deps, argv=list(sys.argv))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)


def main():
    world = os.getcwd()
    sys.path.insert(0, world)  # the world's own bound closure is the import root
    expected = json.loads(os.environ["V5_CHILD_EXPECT"])
    problems, _schema, schema_file = _verify(world, expected)
    if problems:
        for problem in problems:
            print(f"CHILD-VERIFY-FAIL: {problem}", file=sys.stderr)
        return 3
    print(f"CHILD-VERIFY-OK schema={os.path.relpath(schema_file, world)} "
          f"segments={ {k: list(v) for k, v in _schema.SEGMENTS.items()} }")
    sys.argv = ["experiment"] + expected["cli_args"]
    code = 0
    try:
        runpy.run_module("forecasting.full_v5.experiment", run_name="__main__", alter_sys=False)
    except SystemExit as exit_exc:
        code = exit_exc.code if isinstance(exit_exc.code, int) else (0 if exit_exc.code is None else 1)
    except BaseException:
        code = 1
        _dump_snapshot(os.environ["V5_CHILD_SNAPSHOT"], world, expected["dep_versions"])
        raise
    _dump_snapshot(os.environ["V5_CHILD_SNAPSHOT"], world, expected["dep_versions"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
