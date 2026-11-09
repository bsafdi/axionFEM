"""axf-mesher's curved-mesh process: its files, its lock and its errors.

A curved 3D mesh is made by a second axf-mesher process in a private work directory <output>.axf-mesher.XXXXXX, in
which every file of the input has a fixed name linked to the user's file; <output>.axf-mesher.lock keeps two meshers
off one output. The tests cover what this arrangement can break: a relative STEP path in "domain_solid" must be
linked; two meshers on one output must not share a work directory or delete files that axf-mesher did not make; errors
must name the user's files, not the fixed names; the lock must not follow a symbolic link and overwrite its target; a
path that is not UTF-8, closed standard descriptors and long output names must work; a stopped run must clean up.
The fast cases mesh QUICK, a box in the domain sphere (about 1 s), or fail before any meshing; the others need
AXF_SLOW=1 (about 2 min).
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from axionfem import _paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mesh3d_reproducible import SHELL, SHELL_COARSE, _env  # noqa: E402

slow = pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D meshing (slow); set AXF_SLOW=1")
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the curved-mesh process exists on Linux only")
# a quick curved (order-2) mesh: a box in the domain sphere, whose curved surface the elastic pass smooths
QUICK = {"air_attr": 1, "dim": 3, "order": 2, "outer_attr": 901, "shell_attr": 898, "symmetry": [], "threads": 1,
         "domain_radius": 0.2, "infinity_shell": 0, "size": 0.1, "surface_size": 0.1, "verbose": 0,
         "regions": [{"attr": 2, "name": "box", "solid": {"box": {"min": [-0.05, -0.05, -0.05],
                                                                  "max": [0.05, 0.05, 0.05]}}}]}


def _leftovers(d):
    return sorted(str(f.name) for f in Path(d).iterdir() if ".axf-mesher" in str(f.name) or "handoff" in str(f.name))


def _mesher():
    m = _paths.mesher_path()
    if not m:
        pytest.skip("axf-mesher not found")
    return m


def _write(wd, doc, name="in.json"):
    wd.mkdir(parents=True, exist_ok=True)
    (wd / name).write_text(json.dumps(doc, indent=1))
    return wd / name


def _run(inp, cwd, env=None):
    return subprocess.run([_mesher(), str(inp)], capture_output=True, text=True, env=env or _env(), cwd=cwd)


@linux
def test_a_missing_step_file_is_named_as_the_user_wrote_it(tmp_path):
    doc = dict(SHELL_COARSE, verbose=0, output="m/mesh3d.msh", info="m/info.json",
               regions=[dict(SHELL, solid={"step": {"file": "no_such_shell.step"}})])
    inp = _write(tmp_path, doc)
    (tmp_path / "m").mkdir()
    p = _run(inp, tmp_path)
    assert p.returncode == 1, p.stderr
    assert "no_such_shell.step" in p.stderr and "input_0" not in p.stderr, p.stderr
    assert "no_such_shell.step" in json.loads((tmp_path / "m" / "info.json").read_text())["error"]
    assert not [f.name for f in (tmp_path / "m").iterdir() if "axf-mesher" in f.name]


@linux
def test_a_read_only_output_directory_is_named(tmp_path):
    out = tmp_path / "ro"
    out.mkdir()
    doc = dict(SHELL_COARSE, verbose=0, output=str(out / "mesh3d.msh"), info=str(tmp_path / "info.json"))
    inp = _write(tmp_path, doc)
    out.chmod(0o555)
    try:
        if os.access(out, os.W_OK):
            pytest.skip("the directory stays writable (root?)")
        p = _run(inp, tmp_path)
    finally:
        out.chmod(0o755)
    assert p.returncode == 1, p.stderr
    assert str(out) in p.stderr and "writable" in p.stderr, p.stderr


@slow
@linux
def test_relative_step_path_in_domain_solid(tmp_path):
    """"domain_solid" STEP paths are relative to where axf-mesher runs, as before the curved-mesh process."""
    src = tmp_path / "src"
    doc = dict(SHELL_COARSE, verbose=0, output=str(src / "unused.msh"), info=str(src / "info.json"),
               step_output=str(tmp_path / "dom.step"), geometry_only=True)
    p = _run(_write(src, doc), src)
    assert p.returncode == 0, p.stderr[-2000:]
    wd = tmp_path / "m"
    wd.mkdir()
    doc = dict(SHELL_COARSE, verbose=0, domain_solid={"step": {"file": "dom.step"}}, output="m/mesh3d.msh",
               info="m/mesh3d_info.json")
    p = _run(_write(tmp_path, doc, "in_dom.json"), tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    info = json.loads((wd / "mesh3d_info.json").read_text())
    assert info["high_order_canonical"] == "on", info


@slow
@linux
def test_two_meshers_on_one_output_and_files_they_did_not_make(tmp_path):
    """A second axf-mesher on the same output stops with a message while the first meshes; neither touches a
    file or directory that it did not make, including one named like a work directory."""
    wd = tmp_path / "m"
    wd.mkdir()
    keep = [wd / "mesh_notes.txt", wd / "input_data.csv", wd / "mesh3d.msh.axf-mesher" / "mesh_notes.txt",
            wd / "mesh3d.msh.axf-mesher.old" / "input_0.step"]
    for f in keep:
        f.parent.mkdir(exist_ok=True)
        f.write_text("not axf-mesher's")
    doc = dict(SHELL_COARSE, verbose=0, output=str(wd / "mesh3d.msh"), info=str(wd / "mesh3d_info.json"))
    inp = _write(tmp_path, doc)
    first = subprocess.Popen([_mesher(), str(inp)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=_env(), cwd=tmp_path)
    lock = wd / "mesh3d.msh.axf-mesher.lock"
    for _ in range(600):
        if lock.exists() or first.poll() is not None:
            break
        time.sleep(0.1)
    assert lock.exists(), "the first mesher took no lock"
    second = _run(_write(tmp_path, dict(doc, info=str(wd / "second_info.json")), "in2.json"), tmp_path)
    out, _ = first.communicate(timeout=1800)
    assert first.returncode == 0, out[-2000:]
    assert second.returncode == 1, second.stderr
    assert "another axf-mesher (host " in second.stderr and str(wd / "mesh3d.msh") in second.stderr, second.stderr
    for f in keep:
        assert f.read_text() == "not axf-mesher's", f
    made = sorted(f.name for f in wd.iterdir())
    assert made == sorted(["mesh3d.msh", "mesh3d_info.json", "second_info.json", "mesh_notes.txt", "input_data.csv",
                           "mesh3d.msh.axf-mesher", "mesh3d.msh.axf-mesher.old"]), made


@slow
def test_mesher_warnings_reach_the_results(tmp_path, monkeypatch):
    """A curved mesh that is not reproducible says so in what `axf mesh` returns, not only in its log."""
    from axionfem import _core
    from test_3d import _COIL, _WIND, _doc3
    doc = _doc3([_COIL], _WIND)
    doc["mesh"] = dict(doc["mesh"], order=2)
    monkeypatch.setenv("AXF_MESHER_HO_ISOLATE", "0")
    out = json.loads(_core.mesh(json.dumps(doc), str(tmp_path / "m"), 0))
    assert any("not reproducible" in w for w in out.get("warnings", [])), out


@linux
def test_a_symbolic_link_at_the_lock_path_is_refused_and_its_target_kept(tmp_path):
    """The lock must not follow a link there and overwrite its target with a pid."""
    wd = tmp_path / "m"
    wd.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("a file of the user's\n")
    (wd / "mesh3d.msh.axf-mesher.lock").symlink_to(victim)
    doc = dict(QUICK, output=str(wd / "mesh3d.msh"), info=str(wd / "info.json"))
    p = _run(_write(tmp_path, doc), tmp_path)
    assert p.returncode == 1, p.stderr
    assert "symbolic link" in p.stderr, p.stderr
    assert victim.read_text() == "a file of the user's\n"
    assert (wd / "mesh3d.msh.axf-mesher.lock").is_symlink()
    assert not [f for f in _leftovers(wd) if f != "mesh3d.msh.axf-mesher.lock"], _leftovers(wd)


@linux
def test_a_working_directory_that_is_not_utf8(tmp_path):
    """A path that is not UTF-8 must not abort the mesher (in json::dump) or leave its files."""
    cwd = os.path.join(os.fsencode(tmp_path), b"caf\xe9")
    os.mkdir(cwd)
    os.mkdir(os.path.join(cwd, b"m"))
    doc = dict(QUICK, output="m/mesh3d.msh", info="m/info.json")
    with open(os.path.join(cwd, b"in.json"), "w") as f:
        json.dump(doc, f)
    p = subprocess.run([_mesher(), "in.json"], capture_output=True, text=True, errors="replace", env=_env(), cwd=cwd)
    assert p.returncode == 0, p.stderr[-2000:]
    with open(os.path.join(cwd, b"m", b"info.json")) as f:
        info = json.load(f)
    assert info["high_order_canonical"] == "on", info
    assert not _leftovers(os.fsdecode(os.path.join(cwd, b"m")))
    # an error naming a file in that directory: reported, with nothing left behind
    doc = dict(QUICK, output="m/mesh3d.msh", info="m/info2.json",
               regions=[dict(QUICK["regions"][0], solid={"step": {"file": "missing.step"}})])
    with open(os.path.join(cwd, b"in2.json"), "w") as f:
        json.dump(doc, f)
    p = subprocess.run([_mesher(), "in2.json"], capture_output=True, text=True, errors="replace", env=_env(), cwd=cwd)
    assert p.returncode == 1 and "missing.step" in p.stderr, p.stderr[-2000:]
    with open(os.path.join(cwd, b"m", b"info2.json")) as f:
        assert "missing.step" in json.load(f)["error"]


@linux
def test_closed_standard_descriptors(tmp_path):
    """With fds 0-2 closed, the copy in memory could land on fd 2 and stop the mesher; it must not."""
    wd = tmp_path / "m"
    wd.mkdir()
    doc = dict(QUICK, output=str(wd / "mesh3d.msh"), info=str(wd / "info.json"))
    inp = _write(tmp_path, doc)
    p = subprocess.run(["sh", "-c", 'exec "$0" "$1" <&- >&- 2>&-', _mesher(), str(inp)], env=_env(), cwd=tmp_path)
    assert p.returncode == 0
    info = json.loads((wd / "info.json").read_text())
    assert info["ok"] and info["high_order_executable"] == "in memory", info
    assert not _leftovers(wd), _leftovers(wd)


@linux
def test_a_long_output_name(tmp_path):
    """An output name of 238-254 characters must leave room for the lock and work names."""
    wd = tmp_path / "m"
    wd.mkdir()
    name = "a" * 236 + ".msh"
    doc = dict(QUICK, output=str(wd / name), info=str(wd / "info.json"))
    p = _run(_write(tmp_path, doc), tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert (wd / name).exists()
    assert not _leftovers(wd), _leftovers(wd)


@slow
@linux
def test_a_stopped_run_cleans_up(tmp_path):
    """SIGTERM to axf-mesher stops the curved-mesh process; the work directory and the lock
    file go, and axf-mesher ends by that signal."""
    wd = tmp_path / "m"
    wd.mkdir()
    doc = dict(SHELL_COARSE, verbose=0, output=str(wd / "mesh3d.msh"), info=str(wd / "info.json"))
    p = subprocess.Popen([_mesher(), str(_write(tmp_path, doc))], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=_env(), cwd=tmp_path, start_new_session=True)
    for _ in range(600):
        if [f for f in wd.iterdir() if f.is_dir()] or p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is None, "the mesher ended before it could be stopped"
    time.sleep(1)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=120)
    assert p.returncode == -signal.SIGTERM, (p.returncode, out[-2000:])
    assert not _leftovers(wd), _leftovers(wd)
    assert "stopped by signal" in json.loads((wd / "info.json").read_text())["error"]
