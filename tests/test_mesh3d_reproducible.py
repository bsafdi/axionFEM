"""Order-2 3D meshes are reproducible run to run, where gmsh's elastic pass and HighOrder patch optimizer run.

gmsh orders both by the addresses of the mesh objects: the patch optimizer keeps each patch in a std::set of element
pointers, and the elastic pass sorts the surface triangles it smooths by address. Those addresses depended on the heap
that mesh generation left behind, which differs from run to run even on one thread with ASLR off, so the same input gave
different curved meshes (the 5 mm capacitor of the tree-cotree runs: 743 vertices and 8426 mid-edge nodes moved by up to
50 um, C by 2.4e-5). axf-mesher therefore meshes and raises the order in a forked child and runs both steps in the
parent on the handed-over mesh (mesh info "high_order_isolation": "isolated"), and re-executes itself once with address
randomization off ("address_randomization": "off"; Linux).

That parent alone still depends on its caller: the strings it allocates from the output path (the path, + ".handoff",
+ ".handoff.json", the mesher's intermediate files) decide which free chunks gmsh's tetrahedra reuse. SHELL_COARSE gave
four meshes for mesh directories 60, 66, 72 and 78 characters long, one for each number (0-3) of those strings in
glibc's 112-byte chunks. A pytest tmp_path under /tmp/pytest-of-<user> puts "a" and "second_run" in two of them; a
tmp_path of 139 and 148 characters puts both in none. Inputs with STEP solids or the surface dump also depended on the
executable's location, the environment's size, LD_PRELOAD, where stdout goes and the node. The curved steps therefore
run in a process with a fixed command line, environment, executable path, stdout buffer and CPU count, in a private
work directory ("high_order_canonical": "on", "high_order_executable": "in memory"; Linux).

- The test's design: a coarse curved_probe_shell through `axf mesh` (105 invalid curved tets after the elastic pass, so
  the patch optimizer runs). The control, AXF_MESHER_HO_ISOLATE=0, gave 3 meshes in 3 runs with address randomization
  on, but with it off (the default) it can give one mesh twice. So the hash comparison alone may not catch a
  regression to one process; the "high_order_isolation" == "isolated" assertion does.
- shell_coarse: a coarser mesh of the same geometry (11,659 tets, mesher input written here), on which the elastic
  pass smooths curved surfaces ("Smoothing surface" in gmsh's log); with the elastic pass in the meshing process,
  every run gave a new mesh there (4 of 4).
- The same shell as a STEP solid, and with the surface dump.

Each case is meshed in directories of the lengths above, given relative to tmp_path (the run's current directory) so
that the lengths do not depend on where pytest puts it, and once more in an absolute path, from another current
directory, with a different environment (OpenMP placement, locale, malloc tunables, a long variable) and with what the
curved-mesh process would otherwise take from its caller (OTHER_CALLER: a copy of the mesher elsewhere, an explicit
default switch, no LD_LIBRARY_PATH, LD_PRELOAD, stdout to a pipe). All meshes of a case must be byte-identical; the
hashes are compared before the mesh info, so that a non-reproducible mesher fails on its meshes. Measured: one mesh in
48 of 48 runs of each case. With address randomization on, 1 run in 146 differed (the test's design) and 2 in 81
(shell_coarse, elastic pass already in the parent). Set AXF_SLOW=1.
"""
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D meshing (slow); set AXF_SLOW=1")

from axionfem import _paths

DESIGN = Path(__file__).resolve().parents[1] / "benchmarks" / "designs" / "curved_probe_shell.json"
# Coarser than the design's own 20 mm / 8 mm mesh: 43k tets, 105 of them invalid after the elastic pass, 0 after the
# patch optimizer, 35-50 s per mesh.
MESH = {"size": 0.06, "surface_size": 0.02, "domain_radius": 0.2}
# shell_coarse: the same shell at size 0.1, surface
# size 0.05 and 4 points per curvature radian, no infinity shell; 11,659 tets, 18 invalid after the elastic pass.
SHELL = {"attr": 2, "name": "shell", "solid": {"revolve": {"angles": [0.0, 360.0], "axis": [0.0, 0.0, 1.0],
         "center": [0.0, 0.0, 0.0], "profile": {"frame": {"fillet": 0.0055, "r": [0.05, 0.15], "thickness": 0.005,
                                                          "z": [-0.05, 0.05]}}}}}
SHELL_COARSE = {"air_attr": 1, "dim": 3, "order": 2, "outer_attr": 901, "shell_attr": 898, "symmetry": [], "threads": 1,
                "domain_radius": 0.2, "infinity_shell": 0, "size": 0.1, "surface_size": 0.05, "curvature_points": 4,
                "refine_regions": ["shell"], "regions": [SHELL], "verbose": 1}
# switches that change meshes; the tests run the defaults
MESHER_ENV = ("AXF_MESHER_HO", "AXF_MESHER_HO_ISOLATE", "AXF_MESHER_OPTIMIZE", "AXF_MESHER_CURVATURE_FLOOR",
              "AXF_MESHER_ASLR", "AXF_MESHER_SURFACE_DUMP", "PETSC_OPTIONS")
# lengths of the mesh directories (relative paths): 0, 1, 2 and 3 of the output path strings in 112-byte chunks
DIR_LENGTHS = (60, 66, 72, 78)
# what a caller's environment may differ in, none of which may change a curved mesh
OTHER_ENV = {"OMP_NUM_THREADS": "4", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close", "LANG": "C.UTF-8",
             "GLIBC_TUNABLES": "glibc.malloc.tcache_count=0", "AXF_TEST_PADDING": "x" * 200}
LINUX = sys.platform.startswith("linux")
WARNING = "WARNING: this curved (order-2) mesh is not reproducible"


def _env(other=False):
    env = dict(os.environ)
    for k in MESHER_ENV:
        env.pop(k, None)
    if other:
        env.update(OTHER_ENV)
    return env


def _loader_library_path(exe, library_path):
    """The directories of `library_path`, in its order, that the dynamic loader needs to start `exe` without
    LD_LIBRARY_PATH (libraries that are neither on the executable's RPATH nor in the system directories, e.g. an MPI
    library's network stack), found with ldd; "" when it needs none, or off Linux, or without ldd."""
    if not (LINUX and exe and library_path and shutil.which("ldd")):
        return ""
    dirs = [d for d in library_path.split(os.pathsep) if d]
    keep = []
    for _ in range(len(dirs) + 1):
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
        if keep:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(keep)
        out = subprocess.run(["ldd", str(exe)], capture_output=True, text=True, env=env).stdout
        missing = [line.split()[0] for line in out.splitlines() if "not found" in line]
        found = [next((d for d in dirs if d not in keep and os.path.exists(os.path.join(d, name))), None)
                 for name in missing]
        found = [d for d in dict.fromkeys(found) if d]
        if not found:
            break
        keep += found
    return os.pathsep.join(keep)


def _other_caller(tmp_path, env, unset_library_path=True):
    """What the curved-mesh process would otherwise take from its caller, changed: the environment's size
    (a default switch set), LD_LIBRARY_PATH (unset; axf itself needs it) and LD_PRELOAD (a small library), and a copy
    of the mesher in a directory of another length. Returns the mesher to run. Where the mesher cannot start without
    some of the caller's LD_LIBRARY_PATH, only those directories are passed on: the variable still differs from the
    caller's, and the mesher runs."""
    env["AXF_MESHER_HO_ISOLATE"] = "1"
    if unset_library_path:
        needed = _loader_library_path(_paths.mesher_path(), env.pop("LD_LIBRARY_PATH", ""))
        if needed:
            env["LD_LIBRARY_PATH"] = needed
    util = ctypes.util.find_library("util")
    if util:
        env["LD_PRELOAD"] = util
    copy = tmp_path / "a_copy_of_the_mesher_in_a_directory_with_a_long_name" / "axf-mesher"
    if not copy.exists():
        copy.parent.mkdir()
        shutil.copy2(_paths.mesher_path(), copy)
    probe = subprocess.run([str(copy)], capture_output=True, text=True, env=env)
    return str(copy) if probe.returncode == 2 else _paths.mesher_path()  # 2: usage, so it starts


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reasons(wd, info):
    """Why the curved mesh in wd was not made reproducibly (empty if it was)."""
    why = []
    leftovers = sorted(f.name for f in wd.iterdir() if "handoff" in f.name or ".axf-mesher" in f.name)
    if leftovers:
        why.append(f"axf-mesher left {leftovers}")
    if info.get("high_order_isolation") != "isolated":
        why.append(f"high_order_isolation is {info.get('high_order_isolation')!r}")
    if LINUX:
        if info.get("address_randomization") != "off":
            why.append(f"address_randomization is {info.get('address_randomization')!r}")
        if info.get("high_order_canonical") != "on":
            why.append(f"high_order_canonical is {info.get('high_order_canonical')!r}")
        if info.get("high_order_executable") != "in memory":
            why.append(f"high_order_executable is {info.get('high_order_executable')!r}")
        if info.get("high_order_threads_before_fork") != 1:
            why.append(f"{info.get('high_order_threads_before_fork')} threads before the fork")
    return why


def _compare(runs):
    """All meshes of a case equal (checked first), then every run made reproducibly."""
    hashes = [r["sha"] for r in runs]
    assert len(set(hashes)) == 1, [(r["name"], r["sha"][:12], r["info"].get("high_order_passes")) for r in runs]
    for r in runs:
        assert not r["why"], f"the curved mesh in {r['wd']} was not made reproducibly: " + "; ".join(r["why"])
        if LINUX:  # elsewhere every curved mesh warns that there is no curved-mesh process
            assert WARNING not in r["log"], r["log"][-2000:]


def _dirs(tmp_path):
    """(name, current directory, variation): relative mesh directories of DIR_LENGTHS (current directory tmp_path),
    then an absolute one from another current directory with another environment, then one with another caller."""
    names = [("d%d" % n).ljust(n, "x") for n in DIR_LENGTHS]
    return ([(name, tmp_path, None) for name in names] +
            [(str(tmp_path / "second_run"), tmp_path / "elsewhere", "env"),
             (str(tmp_path / "third_run"), tmp_path, "caller")])


def _axf_mesh(tmp_path, design, name, cwd, variation):
    axf = _paths.axf_path()
    if not axf:
        pytest.skip("axf executable not found")
    cwd.mkdir(exist_ok=True)
    env = _env(variation == "env")
    if variation == "caller":
        env["AXF_MESHER"] = _other_caller(tmp_path, env, unset_library_path=False)
    p = subprocess.run([axf, "mesh", str(design), "-o", name, "-v", "0"], capture_output=True, text=True, env=env,
                       cwd=cwd)
    assert p.returncode == 0, p.stderr[-2000:]
    wd = cwd / name
    info = json.loads((wd / "mesh3d_info.json").read_text())
    log = p.stderr + (wd / "mesh3d_mesher.log").read_text()
    return {"name": name, "wd": wd, "sha": _sha(wd / "mesh3d.msh"), "info": info, "why": _reasons(wd, info), "log": log}


def _mesher(tmp_path, doc, name, cwd, variation, extra_env=None):
    mesher = _paths.mesher_path()
    if not mesher:
        pytest.skip("axf-mesher not found")
    cwd.mkdir(exist_ok=True)
    wd = cwd / name
    wd.mkdir()
    doc = dict(doc, output=name + "/mesh3d.msh", info=name + "/mesh3d_info.json")
    (wd / "mesh3d_in.json").write_text(json.dumps(doc, indent=1))
    env = _env(variation == "env")
    env.update(extra_env or {})
    if variation == "caller":  # also stdout to a pipe, not a file
        mesher = _other_caller(tmp_path, env)
        p = subprocess.run([mesher, name + "/mesh3d_in.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=env, cwd=cwd)
        (wd / "mesher.log").write_bytes(p.stdout)
    else:
        with open(wd / "mesher.log", "w") as log:
            p = subprocess.run([mesher, name + "/mesh3d_in.json"], stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=cwd)
    text = (wd / "mesher.log").read_text()
    assert p.returncode == 0, text[-2000:]
    info = json.loads((wd / "mesh3d_info.json").read_text())
    return {"name": name, "wd": wd, "sha": _sha(wd / "mesh3d.msh"), "info": info, "why": _reasons(wd, info), "log": text}


def _step_file(tmp_path):
    """The shell of SHELL_COARSE as a STEP file (axf-mesher's geometry export)."""
    src = tmp_path / "step_src"
    src.mkdir(exist_ok=True)
    step = src / "shell.step"
    if not step.exists():
        doc = dict(SHELL_COARSE, output=str(src / "unused.msh"), info=str(src / "info.json"), verbose=0,
                   step_output=str(step), geometry_only=True)
        (src / "in.json").write_text(json.dumps(doc))
        p = subprocess.run([_paths.mesher_path(), str(src / "in.json")], capture_output=True, text=True, env=_env())
        assert p.returncode == 0 and step.exists(), p.stderr[-2000:]
    return step


def test_curved_mesh_with_patch_optimizer_is_reproducible(tmp_path):
    design = json.loads(DESIGN.read_text())
    design["mesh"] = dict(design["mesh"], **MESH)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design, indent=1))
    dirs = _dirs(tmp_path)
    runs = [_axf_mesh(tmp_path, path, name, cwd, var) for name, cwd, var in (dirs[0], dirs[2], dirs[-2], dirs[-1])]
    counts = runs[0]["info"]["high_order_counts"]
    assert counts["invalid_after_elastic"] > 0, counts  # else the patch optimizer did not run and nothing is tested
    assert runs[0]["info"]["high_order_patch_optimizer"] == "isolated", runs[0]["info"]
    _compare(runs)


def test_curved_mesh_with_surface_smoothing_is_reproducible(tmp_path):
    runs = [_mesher(tmp_path, SHELL_COARSE, name, cwd, var) for name, cwd, var in _dirs(tmp_path)]
    assert "Smoothing surface" in runs[0]["log"]  # else the elastic pass's address-ordered surface step did not run
    assert runs[0]["info"]["high_order_counts"]["invalid_after_elastic"] > 0, runs[0]["info"]["high_order_counts"]
    _compare(runs)


def test_curved_mesh_of_a_step_solid_is_reproducible(tmp_path):
    """The shell imported from a STEP file, a case that depended on the caller and the node."""
    step = _step_file(tmp_path)
    doc = dict(SHELL_COARSE, verbose=0, regions=[dict(SHELL, solid={"step": {"file": str(step)}})])
    dirs = _dirs(tmp_path)
    runs = [_mesher(tmp_path, doc, name, cwd, var) for name, cwd, var in (dirs[0], dirs[2], dirs[-1])]
    _compare(runs)


def test_surface_dump_leaves_the_curved_mesh_alone(tmp_path):
    """The dump is written by a short-lived child: with it, the curved mesh is the one made without it, whatever the
    caller."""
    dirs = _dirs(tmp_path)
    plain = _mesher(tmp_path, dict(SHELL_COARSE, verbose=0), *dirs[0])
    runs = [plain]
    for name, cwd, var in (dirs[2], dirs[-1]):
        dump = cwd / (Path(name).name + "_faces.json")
        runs.append(_mesher(tmp_path, dict(SHELL_COARSE, verbose=0), name, cwd, var,
                            extra_env={"AXF_MESHER_SURFACE_DUMP": str(dump)}))
        assert json.loads(dump.read_text()), dump
    _compare(runs)


def test_curved_mesh_in_one_process_says_it_is_not_reproducible(tmp_path):
    """AXF_MESHER_HO_ISOLATE=0 (one process): the mesh is made, and the mesher says loudly that it is not reproducible."""
    mesher = _paths.mesher_path()
    if not mesher:
        pytest.skip("axf-mesher not found")
    wd = tmp_path / "one_process"
    wd.mkdir()
    doc = dict(SHELL_COARSE, output=str(wd / "mesh3d.msh"), info=str(wd / "mesh3d_info.json"), verbose=0)
    (wd / "mesh3d_in.json").write_text(json.dumps(doc, indent=1))
    p = subprocess.run([mesher, str(wd / "mesh3d_in.json")], capture_output=True, text=True,
                       env=dict(_env(), AXF_MESHER_HO_ISOLATE="0"))
    assert p.returncode == 0, p.stderr[-2000:]
    info = json.loads((wd / "mesh3d_info.json").read_text())
    assert info["high_order_isolation"] == "off: AXF_MESHER_HO_ISOLATE=0", info
    assert WARNING in p.stderr and "AXF_MESHER_HO_ISOLATE=0" in p.stderr, p.stderr[-2000:]
