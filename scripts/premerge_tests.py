"""The quick pre-merge subset: check tests/premerge.txt against the suite, and run it.

    python scripts/premerge_tests.py check                 # the list is sound and every listed test exists (seconds)
    python scripts/premerge_tests.py run [pytest args...]  # check, then run the listed tests (scripts/slurm/tests_premerge.slurm)
    python scripts/premerge_tests.py ids                   # print every listed node ID, one per line
    python scripts/premerge_tests.py ci3d                  # print the 3D part of the list (CI's linux-mpi job runs it)

tests/premerge.txt names pytest node IDs, one per line (a whole file, a test, or one parametrization). A line
`exclude tests/test_x.py` records a test file that is deliberately not represented. Text after `#` is a comment.

`check` fails, before anything runs, when
- a listed file does not exist, or a listed test is not collected (renamed, removed, or a parametrization changed);
  a listed file that skips as a whole at collection because this build lacks a feature (the fluids module of an MPI
  build, say) is reported, not fatal, since its tests cannot be checked here;
- a line is repeated, or a test is listed together with its whole file;
- a test file under tests/ is neither listed nor excluded (a new file: add a representative or an exclude line);
- an excluded file is also listed;
- a test file builds a 3D design (THREE_D below: `"symmetry": "3d"`, `to_3d(`, a 3D benchmark module) but never
  mentions `engine_3d` (the skip guard for builds without the 3D engine, such as the wheels and CI's sequential
  job), and is not named in NO_SOLVE_3D or MIXED_3D below with a reason. The same check runs in
  tests/test_packaging.py, without a compiled core.

`ci3d` selects the entries whose file mentions `engine_3d`, plus the files in MIXED_3D: the part of the list that a
build without the 3D engine (the wheels, CI's sequential job) cannot run. CI's MPI job runs these on pull requests
(.github/workflows/ci.yml).

`run` sets AXF_SLOW=1 (and AXF_MPI=1 unless it is set), runs the listed tests and writes a JUnit XML file. It fails if
any test failed, and also if a test named on its own line was skipped, or every test of a listed file: a skipped test
covers nothing, so either the environment is wrong (no SLURM allocation, too few MPI slots, a missing build feature)
or the test does not belong in the list. Skips inside a listed file (an optional package such as schemdraw) are
reported, not fatal.
"""
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = Path(os.environ.get("AXF_PREMERGE_LIST", ROOT / "tests" / "premerge.txt")).resolve()

# a 3D design document, a 3D solve, or a benchmark module that builds 3D designs
THREE_D = re.compile(r"""["']symmetry["']\s*:\s*["']3d["']|\bto_3d\(|\brun_3d\b|Model3D|stage1_twoport|"""
                     r"""\bimport (fullwave_indicator|dzdw_check|fw3d_robin_validation|fullwave3d_background)\b""")
# 3D test files without a file-wide engine_3d guard, gated per test (AXF_SLOW, AXF_MPI); `ci3d` includes them
MIXED_3D = {
    "tests/test_cavity3d.py": "3D eigenmodes behind AXF_SLOW; its spec and closed-form tests are pure Python",
    "tests/test_srf.py": "its 3D cell is behind AXF_SLOW",
}
# files that build 3D design documents but never solve them (or need a build CI does not have)
NO_SOLVE_3D = {
    "tests/test_convergence.py": "3D documents for the history and spec checks only",
    "tests/test_scan_reach.py": "a 3D design with a background, validated but not solved",
    "tests/test_schema.py": "validates a 3D document",
    "tests/test_widget.py": "a 3D scene for the viewer",
    "tests/test_mesh3d_degenerate_element.py": "meshes only (axf-mesher is in every build), behind AXF_SLOW",
    "tests/test_gpu.py": "needs a CUDA build and a visible GPU",
    "tests/test_axi_background_exterior.py": "axisymmetric; uses fullwave3d_background's grid helpers only",
    "tests/test_multiport_fem_spec.py": "validates the two-port frame's specification, not solved",
    "tests/test_mpi_output_parser.py": "parses faked launcher output; Model3D is a stub, nothing is solved",
}


def _text(rel):
    return (ROOT / rel).read_text("utf-8")


def guard_errors():
    """Test files that build 3D designs without an engine_3d guard or a stated reason (tests/test_packaging.py)."""
    errors = []
    for name in list(MIXED_3D) + list(NO_SOLVE_3D):
        if not (ROOT / name).is_file():
            errors.append(f"{name} is named in premerge_tests.py's MIXED_3D or NO_SOLVE_3D but does not exist")
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        rel = f.relative_to(ROOT).as_posix()
        text = f.read_text("utf-8")
        if THREE_D.search(text) and "engine_3d" not in text and rel not in MIXED_3D and rel not in NO_SOLVE_3D:
            errors.append(f"{rel} builds a 3D design but has no engine_3d skip guard: add pytestmark = "
                          f"pytest.mark.skipif(not af.build_info()['engine_3d'], ...) (or name it in "
                          f"scripts/premerge_tests.py's MIXED_3D / NO_SOLVE_3D, with the reason)")
    return errors


def ci3d(ids):
    """The entries CI's linux-mpi job runs on pushes and pull requests: files that mention engine_3d, and the mixed
    3D files."""
    files = {i.split("::", 1)[0] for i in ids}
    needs = {f for f in files if f in MIXED_3D or ((ROOT / f).is_file() and "engine_3d" in _text(f))}
    return [i for i in ids if i.split("::", 1)[0] in needs]


def fail(lines):
    print("PREMERGE CHECK FAILED", file=sys.stderr)
    for line in lines:
        print("  " + line, file=sys.stderr)
    print(f"  (fix {os.path.relpath(LIST, ROOT)}; see its header)", file=sys.stderr)
    sys.exit(2)


def parse(path=LIST):
    """(node IDs, excluded files), with the line number of each entry."""
    ids = []
    excluded = []
    for n, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        if words[0] == "exclude" and len(words) == 2:
            excluded.append((words[1], n))
        elif len(words) == 1 and words[0].startswith("tests/"):
            ids.append((words[0], n))
        else:
            fail([f"line {n}: '{raw.strip()}' is neither a node ID (tests/...) nor 'exclude tests/<file>.py'"])
    return ids, excluded


def check_list(ids, excluded):
    errors = []
    seen = {}
    whole_files = {i for i, _ in ids if "::" not in i}
    for i, n in ids:
        if i in seen:
            errors.append(f"line {n}: {i} repeats line {seen[i]}")
        seen[i] = n
        f = i.split("::", 1)[0]
        if not (ROOT / f).is_file():
            errors.append(f"line {n}: {f} does not exist")
        elif "::" in i and f in whole_files:
            errors.append(f"line {n}: {i} is listed with its whole file {f}")
    listed_files = {i.split("::", 1)[0] for i, _ in ids}
    for f, n in excluded:
        if not (ROOT / f).is_file():
            errors.append(f"line {n}: excluded {f} does not exist")
        if f in listed_files:
            errors.append(f"line {n}: {f} is excluded but also listed")
    known = listed_files | {f for f, _ in excluded}
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        rel = f.relative_to(ROOT).as_posix()
        if rel not in known:
            errors.append(f"{rel} is neither listed nor excluded: add a representative test, or "
                          f"'exclude {rel}  # why'")
    errors += guard_errors()
    if errors:
        fail(errors)


_MODULE_SKIP = re.compile(r"^SKIPPED \[\d+\] (\S+?):\d+: (.*)$")


def collect(ids):
    """Collect the listed IDs; pytest refuses an ID it cannot find. Returns the collected node IDs and
    {listed entry: reason} for the entries whose file skipped as a whole at collection (a missing build feature)."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-rs", "-p", "no:cacheprovider",
           *[i for i, _ in ids]]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        errors = []
        for line in out.splitlines():
            if "not found:" in line:
                rel = os.path.relpath(line.split("not found:", 1)[1].strip(), ROOT)
                hits = [f"line {n}: {i} is not collected (renamed or removed? a changed parametrization?)"
                        for i, n in ids if i == rel or i.startswith(rel + "[")]
                errors += hits or [line]
            elif "ERROR" in line or "Error" in line:
                errors.append(line)
        fail(["pytest could not collect the list:", *(errors or out.splitlines()[-20:])])
    collected = [line for line in p.stdout.splitlines() if "::" in line and not line.startswith(" ")]
    skipped_files = {}
    for line in out.splitlines():
        m = _MODULE_SKIP.match(line.strip())
        if m:
            skipped_files[os.path.relpath(os.path.join(ROOT, m.group(1)), ROOT)] = m.group(2)
    module_skipped = {}
    for i, n in ids:
        if not any(c == i or c.startswith(i + "::") or c.startswith(i + "[") for c in collected):
            rel = i.split("::")[0].split("[")[0]
            if rel in skipped_files:
                module_skipped[i] = f"line {n}: {i} skipped at collection ({skipped_files[rel]})"
                continue
            fail([f"line {n}: {i} collects no test"])
    return collected, module_skipped


def junit_outcomes(path, collected):
    """{node ID: outcome} from a JUnit XML file (passed, failed, error, skipped: reason), keyed by the collected IDs."""
    by_case = {}
    for node in collected:
        parts = node.split("::")
        cls = ".".join([parts[0][:-3].replace("/", "."), *parts[1:-1]])
        by_case[(cls, parts[-1])] = node
    out = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        node = by_case.get(key, "::".join(key))
        if case.find("failure") is not None:
            out[node] = "failed"
        elif case.find("error") is not None:
            out[node] = "error"
        elif case.find("skipped") is not None:
            out[node] = "skipped: " + case.find("skipped").get("message", "")
        else:
            out[node] = "passed"
    return out


def coverage_lost(ids, outcomes):
    """Skips that lose coverage: any test under an entry that names a test, or a listed file whose every test skipped."""
    lost = []
    for i, n in ids:
        under = {k: v for k, v in outcomes.items() if k == i or k.startswith(i + "::") or k.startswith(i + "[")}
        skipped = {k: v for k, v in under.items() if v.startswith("skipped")}
        if "::" in i:
            lost += [f"line {n}: {k}  ({v})" for k, v in sorted(skipped.items())]
        elif under and len(skipped) == len(under):
            lost.append(f"line {n}: every test of {i} was skipped")
    return lost


def premerge_basetemp(env: dict) -> Path:
    """pytest's temporary directory for the pre-merge subset, never inside the source tree (some tests must not
    create their temporary directories inside a git repository). $TMPDIR (tests_premerge.slurm sets one outside the
    source tree) or the system temporary directory; if that is itself inside the source tree, /tmp."""
    import tempfile
    root = Path(env.get("TMPDIR") or tempfile.gettempdir())
    if root.resolve().is_relative_to(ROOT.resolve()):
        root = Path("/tmp")
    return root / f"axf_premerge_{env.get('SLURM_JOB_ID', 'local')}_{os.getpid()}" / "tmp"


def main(argv):
    if not argv or argv[0] not in ("check", "run", "ids", "ci3d"):
        print(__doc__)
        return 2
    ids, excluded = parse()
    if argv[0] in ("ids", "ci3d"):
        errors = guard_errors()
        if errors:
            fail(errors)
        chosen = [i for i, _ in ids]
        print("\n".join(ci3d(chosen) if argv[0] == "ci3d" else chosen))
        return 0
    check_list(ids, excluded)
    collected, module_skipped = collect(ids)
    print(f"premerge list: {len(ids)} entries, {len(excluded)} excluded files, {len(collected)} tests collected")
    if module_skipped:
        print(f"{len(module_skipped)} listed entries skip as a whole in this build and were not checked:")
        for line in module_skipped.values():
            print("  " + line)
    if argv[0] == "check":
        return 0
    if module_skipped:
        print("PREMERGE FAILED: listed files skip as a whole in this build, so they cover nothing")
        return 1

    env = dict(os.environ)
    env["AXF_SLOW"] = "1"
    env.setdefault("AXF_MPI", "1")
    out_dir = Path(env.get("AXF_PREMERGE_OUT", ROOT / "runs" / f"premerge_{env.get('SLURM_JOB_ID', 'local')}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    junit = out_dir / "junit.xml"
    basetemp = premerge_basetemp(env)
    basetemp.parent.mkdir(parents=True, exist_ok=True)   # pytest needs the parent to exist
    print(f"pytest basetemp: {basetemp} (outside the worktree)")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={basetemp}",
           f"--junitxml={junit}", "--durations=25", "-rfEs", *[i for i, _ in ids], *argv[1:]]
    t0 = time.time()
    code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    wall = time.time() - t0
    if not junit.is_file():
        print(f"PREMERGE FAILED: pytest exited {code} and wrote no {junit}")
        return code or 1
    outcomes = junit_outcomes(junit, collected)
    skipped = {k: v for k, v in outcomes.items() if v.startswith("skipped")}
    failed = {k: v for k, v in outcomes.items() if v in ("failed", "error")}
    print(f"premerge: {len(outcomes)} tests, {len(failed)} failed, {len(skipped)} skipped, "
          f"pytest wall {wall / 60:.1f} min; JUnit XML {junit}")
    lost = coverage_lost(ids, outcomes)
    if lost:
        print("PREMERGE FAILED: listed tests were skipped, so they covered nothing:")
        for line in lost:
            print("  " + line)
        return code or 1
    if code != 0:
        print(f"PREMERGE FAILED: pytest exited {code}")
        return code
    print("PREMERGE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
