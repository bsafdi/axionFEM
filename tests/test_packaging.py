"""Tests for packaging: versions, pins, the build matrix and the 3D stub.

These run without the compiled core (they read pyproject.toml, CMakeLists.txt,
the scripts and the docs).  The few that need ``axionfem`` to import are skipped
when it is not installed; the ones that describe the *sequential* configuration
only assert something when the build really has no 3D engine, so the same file
is meaningful in a cluster build and in a wheel.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (the wheels support it): the backport, which pytest itself requires there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CMAKELISTS = ROOT / "CMakeLists.txt"
CORE_HPP = ROOT / "cpp" / "include" / "axionfem" / "core.hpp"
INSTALL_DOC = ROOT / "docs" / "install.md"


def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text("utf-8"))


# ----------------------------------------------------------------- versions
def test_version_is_the_same_in_python_cmake_and_cpp():
    py = pyproject()["project"]["version"]
    cm = re.search(r"project\(axionFEM VERSION ([0-9.]+)", CMAKELISTS.read_text("utf-8"))
    assert cm, "CMakeLists.txt has no project(axionFEM VERSION ...)"
    cpp = re.search(r'kVersion = "([0-9.]+)"', CORE_HPP.read_text("utf-8"))
    assert cpp, "core.hpp has no kVersion"
    assert py == cm.group(1) == cpp.group(1), (py, cm.group(1), cpp.group(1))


def test_third_party_pins_agree_across_the_build_scripts():
    """scripts/build_deps_wheel.sh duplicates the pins; they must not drift."""
    def pins(path: Path, keys):
        text = (ROOT / path).read_text("utf-8")
        out = {}
        for k in keys:
            m = re.search(rf"^{k}=([0-9.]+)$", text, re.M)
            assert m, f"{path}: no {k}="
            out[k] = m.group(1)
        return out

    shared = ["MFEM_VER", "METIS_VER", "EIGEN_VER", "JSON_VER"]
    base = pins(Path("scripts/fetch_deps.sh"), shared)
    wheel = pins(Path("scripts/build_deps_wheel.sh"), shared)
    assert base == wheel, f"pins drifted: fetch_deps {base} vs build_deps_wheel {wheel}"
    gpu = pins(Path("scripts/build_deps_gpu.sh"), ["MFEM_VER", "HYPRE_VER", "METIS_VER"])
    full = pins(Path("scripts/fetch_deps.sh"), ["MFEM_VER", "HYPRE_VER", "METIS_VER"])
    assert gpu == full, f"pins drifted: fetch_deps {full} vs build_deps_gpu {gpu}"


# ----------------------------------------------------------------- pyproject
def test_build_backend_and_requirements():
    bs = pyproject()["build-system"]
    assert bs["build-backend"] == "scikit_build_core.build"
    req = " ".join(bs["requires"])
    for pkg in ("scikit-build-core", "nanobind", "gmsh"):
        assert pkg in req, f"{pkg} missing from build-system.requires"


def test_gmsh_minor_version_is_pinned_consistently():
    """axf-mesher links libgmsh by SONAME, so the wheel's gmsh, the [mesh] extra, the CI installs, the SDK and the
    auditwheel exclude must all name one minor version (docs/install.md, "Why the [mesh] extra")."""
    pp = pyproject()
    build = [r for r in pp["build-system"]["requires"] if r.startswith("gmsh")]
    extras = pp["project"]["optional-dependencies"]
    mesh = [r for r in extras["mesh"] if r.startswith("gmsh")]
    full = [r for r in extras["all"] if r.startswith("gmsh")]
    assert len(build) == 1 and build == mesh == full, (build, mesh, full)
    m = re.fullmatch(r"gmsh>=(\d+)\.(\d+),<(\d+)\.(\d+)", build[0])
    assert m, f"gmsh must be pinned to one minor version (gmsh>=X.Y,<X.Y+1), not {build[0]!r}"
    major, minor = int(m.group(1)), int(m.group(2))
    assert (int(m.group(3)), int(m.group(4))) == (major, minor + 1), build[0]
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    pins = re.findall(r'"(gmsh[<>=][^"]*)"', ci)
    assert pins and set(pins) == {build[0]}, f"ci.yml must install the same gmsh pin: {pins}"
    for script in ("scripts/fetch_deps.sh", "scripts/build_deps.sh"):
        sdk = re.search(r"^GMSH_VER=(\d+)\.(\d+)\.", (ROOT / script).read_text("utf-8"), re.M)
        assert sdk and (int(sdk.group(1)), int(sdk.group(2))) == (major, minor), script
    repair = pp["tool"]["cibuildwheel"]["linux"]["repair-wheel-command"]
    assert f"--exclude libgmsh.so.{major}.{minor} " in repair, repair


def test_runtime_dependencies_are_minimal():
    p = pyproject()["project"]
    # A laptop install must not drag in MPI, a compiler or GPL code.
    deps = " ".join(p["dependencies"]).lower()
    for forbidden in ("mpi4py", "gmsh", "cmake", "scikit-build"):
        assert forbidden not in deps, f"{forbidden} must not be a runtime dependency"
    extras = p["optional-dependencies"]
    assert "gmsh" in " ".join(extras["mesh"]), "the `mesh` extra must install gmsh"
    assert "anywidget" in " ".join(extras["notebook"])
    assert p["requires-python"].startswith(">=3.10")


def test_console_scripts():
    s = pyproject()["project"]["scripts"]
    assert s["axf"] == "axionfem._cli:main"
    assert (ROOT / "python" / "axionfem" / "_cli.py").is_file()


def test_scikit_build_settings():
    sk = pyproject()["tool"]["scikit-build"]
    assert sk["wheel"]["packages"] == ["python/axionfem"]
    assert "LICENSE" in sk["wheel"]["license-files"]
    # every library the wheel bundles or compiles in has its licence text in the wheel (third_party/licenses)
    assert "third_party/licenses/*" in sk["wheel"]["license-files"]
    lic = {p.name for p in (ROOT / "third_party" / "licenses").iterdir()}
    for need in ("MFEM-LICENSE.txt", "METIS-LICENSE.txt", "Apache-2.0.txt", "Eigen-MPL-2.0.txt",
                 "nlohmann-json-LICENSE.txt", "nanobind-LICENSE.txt", "stb-LICENSE.txt", "hypre-LICENSE-MIT.txt"):
        assert need in lic, need
    cm = (ROOT / "CMakeLists.txt").read_text()
    for lib, text in (("libmfem", "MFEM"), ("libmetis", "METIS"), ("libHYPRE", "hypre")):
        assert lib in cm and any(n.startswith(text) for n in lic), lib
    d = sk["cmake"]["define"]
    # AUTO, so that the serial MFEM of the wheel build selects the 3D stub itself.
    assert d["AXF_MPI"] == "AUTO" and d["AXF_OPENMP"] == "AUTO"
    assert d["AXF_BUNDLE_LIBS"] == "ON" and d["AXF_BUILD_TESTS"] == "OFF"


def test_cibuildwheel_covers_the_required_platforms():
    cw = pyproject()["tool"]["cibuildwheel"]
    assert "scripts/build_deps_wheel.sh" in cw["before-all"]
    assert cw["skip"].find("musllinux") >= 0
    for tag in ("cp310", "cp311", "cp312"):
        assert tag in cw["build"]
    lin = cw["linux"]
    assert lin["archs"] == ["x86_64"]
    assert lin["manylinux-x86_64-image"] == "manylinux_2_28"
    # a platform before-all replaces the global one: it must still build the serial stack
    assert "scripts/build_deps_wheel.sh" in lin.get("before-all", cw["before-all"])
    assert "auditwheel repair" in lin["repair-wheel-command"]
    # gmsh (GPL) is never grafted into our wheel; it comes from the gmsh wheel (docs/install.md)
    assert "--exclude libgmsh" in lin["repair-wheel-command"]
    mac = cw["macos"]
    assert "delocate-wheel" in mac["repair-wheel-command"]
    assert "--exclude libgmsh" in mac["repair-wheel-command"]
    assert mac["environment"]["MACOSX_DEPLOYMENT_TARGET"]


# ----------------------------------------------------------------- CMake matrix
@pytest.mark.parametrize("opt", ["AXF_MPI", "AXF_OPENMP", "AXF_GPU", "AXF_BUNDLE_LIBS",
                                 "AXF_BUILD_MESHER", "AXF_GMSH_ROOT"])
def test_cmake_options_exist_and_are_documented(opt):
    assert re.search(rf"\b{opt}\b", CMAKELISTS.read_text("utf-8")), f"{opt} not in CMakeLists.txt"
    assert opt in INSTALL_DOC.read_text("utf-8"), f"{opt} not documented in docs/install.md"


def test_install_doc_has_the_build_matrix_and_licensing():
    doc = INSTALL_DOC.read_text("utf-8")
    for heading in ("build matrix", "GPU builds", "Licensing"):
        assert heading.lower() in doc.lower(), f"docs/install.md has no '{heading}' section"
    for word in ("manylinux", "delocate", "auditwheel", "cibuildwheel", "AXF_DEVICE"):
        assert word in doc, f"docs/install.md does not mention {word}"


def test_only_model3d_uses_mpi_or_hypre():
    """The premise of the sequential build: the 3D engine files are the only ones to replace with a stub."""
    pattern = re.compile(r"\b(MPI_[A-Za-z_]+|ParMesh|ParFiniteElementSpace|ParGridFunction"
                         r"|ParBilinearForm|ParLinearForm|Hypre[A-Z][A-Za-z]*|HYPRE_[A-Za-z]+)\b")
    offenders = []
    for p in sorted((ROOT / "cpp").rglob("*.cpp")) + sorted((ROOT / "cpp").rglob("*.hpp")):
        if p.name in ("model3d.cpp", "model3d_fullwave.cpp", "model3d_electro.cpp", "model3d_mech.cpp",
                      "model3d_fw_indicator.cpp", "fw_error_indicator.hpp", "model3d_impl.hpp",
                      "model3d_surface_losses.cpp", "surface_losses3d.hpp",
                      "model3d_nompi.cpp"):
            continue
        # the fluids module is ParMesh-based and CMake builds it (and its tests) only with AXF_MPI,
        # so the sequential build never compiles these files
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(("cpp/src/fluids/", "cpp/include/axionfem/fluids/", "cpp/tests/test_fluids_")):
            continue
        # the chemistry 3D current distribution is compiled in every build but wrapped whole in
        # #ifndef AXF_NO_MPI (its binding too), so the sequential build compiles it to nothing
        if rel in ("cpp/src/chem/current_distribution_3d.cpp", "cpp/include/axionfem/chem/current_distribution_3d.hpp"):
            assert "#ifndef AXF_NO_MPI" in p.read_text("utf-8"), f"{rel} lost its AXF_NO_MPI guard"
            continue
        if pattern.search(p.read_text("utf-8")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, (
        "these files now use MPI or hypre, so the sequential build (and the wheels) "
        f"would not compile: {offenders}. Guard them with #ifndef AXF_NO_MPI or move "
        "them behind the 3D engine."
    )


def test_the_stub_defines_everything_model3d_declares():
    header = (ROOT / "cpp" / "include" / "axionfem" / "model3d.hpp").read_text("utf-8")
    stub = (ROOT / "cpp" / "src" / "model3d_nompi.cpp").read_text("utf-8")
    body = header.split("class Model3D {", 1)[1].split("\n};", 1)[0]
    declared = set()
    for line in body.splitlines():
        s = line.strip()
        # Members defined inline in the header (results(), design()) need no stub,
        # and a line with '{' is either such a body or a nested struct.
        if not s or s.startswith("//") or "{" in s:
            continue
        m = re.match(r"(?:explicit\s+)?[\w:<>,&*\s]*?(~?\w+)\s*\(", s)
        if m and m.group(1) not in ("if", "for", "while", "return"):
            declared.add(m.group(1))
    assert {"Model3D", "~Model3D", "run", "sample", "scene", "write_paraview"} <= declared, declared
    missing = [n for n in declared if f"Model3D::{n}(" not in stub]
    assert not missing, (
        f"cpp/src/model3d_nompi.cpp does not define {missing}; the no-MPI build would not link. "
        "Every public member of Model3D needs a stub."
    )
    assert "json run_any(" in stub, "the stub must also provide run_any"


def test_the_wheel_scripts_are_runnable():
    for rel in ("scripts/build_deps_wheel.sh", "scripts/build_deps_gpu.sh"):
        p = ROOT / rel
        assert p.is_file(), rel
        assert p.read_text("utf-8").startswith("#!/usr/bin/env bash"), rel


# ----------------------------------------------------------------- CI workflows
def test_workflows_exist():
    wf = ROOT / ".github" / "workflows"
    ci, wheels = wf / "ci.yml", wf / "wheels.yml"
    assert ci.is_file() and wheels.is_file()
    ci_text, wh_text = ci.read_text("utf-8"), wheels.read_text("utf-8")
    assert "build_deps" in ci_text and "pytest" in ci_text
    assert "cibuildwheel" in wh_text
    # one job per native architecture: no cross-compiling of the MFEM dependency
    for runner in ("ubuntu-24.04", "macos-15-intel", "macos-14"):
        assert runner in wh_text, f"wheels.yml has no {runner} job"



# ----------------------------------------------------------------- the install guide and the containers
def _guide_block(label: str) -> str:
    """The first ```bash block after the paragraph **label** of docs/guides/install.md."""
    text = (ROOT / "docs" / "guides" / "install.md").read_text("utf-8")
    m = re.search(r"^\*\*" + re.escape(label) + r"\*\*\s*\n+```bash\n(.*?)\n```", text, re.S | re.M)
    assert m, f"no bash block under **{label}** in docs/guides/install.md"
    return m.group(1)


def _installed_packages(script: str, tool: str, args: dict | None = None) -> set:
    """The package names of every `<tool> install` command in a shell script or Containerfile RUN line, with
    $VARS expanded from ``args`` (the Containerfile's ARG defaults) and options dropped."""
    script = script.replace("\\\n", " ")
    pkgs = set()
    for m in re.finditer(rf"\b{tool}(?:-get)? install\b([^&\n]*)", script):
        for word in m.group(1).split():
            if word.startswith("$"):
                pkgs.update((args or {}).get(word[1:], "").split())
            elif not word.startswith("-"):
                pkgs.add(word)
    return pkgs


def _containerfile(name: str) -> tuple[str, dict]:
    text = (ROOT / "docker" / name).read_text("utf-8")
    args = dict(re.findall(r'^ARG (\w+)="?([^"\n]*)"?$', text, re.M))
    return text, args


def test_the_containerfiles_install_exactly_the_guides_prerequisites():
    """docker/ is how the guide's commands are tested: its images must contain what the guide lists, no more."""
    for label, name, tool in (("Debian/Ubuntu", "ubuntu-24.04.Containerfile", "apt"),
                              ("RHEL 9 and its rebuilds (Rocky, Alma)", "rockylinux-9.Containerfile", "dnf")):
        guide = _installed_packages(_guide_block(label), tool)
        text, args = _containerfile(name)
        assert args.get("EXTRA_PACKAGES", "") == "", f"docker/{name}: EXTRA_PACKAGES must default to nothing"
        assert _installed_packages(text, tool, args) == guide, f"docker/{name} and the guide's **{label}** differ"
        assert "python3-dev" in guide or "python3.12-devel" in guide

def test_3d_test_files_skip_without_the_3d_engine():
    """A test file that builds a 3D design must skip on a build without the 3D engine (the wheels, CI's sequential
    job): it mentions engine_3d, or scripts/premerge_tests.py names it with a reason. A file without the guard
    fails CI's sequential job."""
    spec = importlib.util.spec_from_file_location("premerge_tests", ROOT / "scripts" / "premerge_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.guard_errors() == []


# ----------------------------------------------------------------- the built module
def _af():
    """The installed package, or skip: the tests above need no compiled core."""
    return pytest.importorskip("axionfem", reason="the compiled core is not importable here")


def test_build_info_describes_the_build():
    af = _af()
    info = af.build_info()
    for key in ("version", "mfem", "device", "gpu_backend", "mpi", "engine_3d", "openmp", "threads"):
        assert key in info, key
    assert info["version"] == af.__version__ == pyproject()["project"]["version"]
    assert isinstance(info["mpi"], bool) and isinstance(info["engine_3d"], bool)
    assert info["mpi"] == info["engine_3d"], "the 3D engine is exactly the MPI build"
    assert info["device"] == "cpu", "AXF_DEVICE must default to the CPU"
    assert info["gpu_backend"] in ("", "cuda", "hip")
    assert info["threads"] >= 1


def test_build_provenance_comes_from_one_place():
    """build_info()'s commit keys come from the header cmake/axf_gitinfo.cmake writes on every build: the module's
    git_sha and source_commit and libaxionfem's own git_sha are one value, and no build path is compiled in (the
    library's build_dir is where the loaded libaxionfem is, looked up at run time)."""
    af = _af()
    info = af.build_info()
    lib = info["library"]
    assert info["source_commit"] == info["git_sha"] == lib["git_sha"], (info["source_commit"], info["git_sha"], lib)
    assert info["source_dirty"] in ("true", "false", "unknown"), info["source_dirty"]
    assert isinstance(info["git_dirty"], bool) and lib["git_dirty"] == info["git_dirty"]
    if info["git_sha"] == "unknown":   # not built from a git checkout (a source archive)
        assert info["git_dirty"] and info["source_dirty"] == "unknown"
    elif info["git_dirty"]:            # the compiled paths are part of the wider source scope
        assert info["source_dirty"] == "true"
    if lib["build_dir"]:
        assert Path(lib["build_dir"]).is_absolute() and list(Path(lib["build_dir"]).glob("libaxionfem*")), lib


def test_a_3d_design_is_refused_when_the_engine_is_absent(tmp_path):
    af = _af()
    if af.build_info()["engine_3d"]:
        pytest.skip("this build has the 3D engine")
    from axionfem.convert import to_3d

    d3 = to_3d(af.template("toroid"))
    with pytest.raises(af.AxionFEMError) as e:
        d3.solve(tmp_path / "s3", verbose=0)
    msg = str(e.value)
    for needed in ("3D engine is not available", "AXF_MPI=OFF", "docs/guides/install.md", "MPI source build"):
        assert needed in msg, f"the error should say {needed!r}: {msg}"


def test_the_mesher_and_the_viewer_are_reachable():
    _af()
    from axionfem import _paths

    report = _paths.prepare_environment()
    assert report["mesher"], "axf-mesher was not found: " + _paths.mesher_hint()
    from axionfem.viewer import viewer_dir

    vd = viewer_dir()
    assert (vd / "template.html").is_file() and (vd / "src" / "viewer.js").is_file()


def test_the_draw_extra_and_import_without_schemdraw():
    """Circuit drawings are optional (axionfem[draw]); importing axionfem, its readout modules and
    axionfem.draw must not need schemdraw, and .draw() without it must say how to install it."""
    import subprocess
    import sys
    extras = pyproject()["project"]["optional-dependencies"]
    assert any(x.startswith("schemdraw") for x in extras["draw"])
    assert any(x.startswith("schemdraw") for x in extras["all"])
    assert "schemdraw" not in " ".join(pyproject()["project"]["dependencies"])
    # The import half needs the compiled core, like the other tests of the built module: the
    # "packaging metadata" CI job has none and checks only the metadata above.
    _af()
    code = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "schemdraw" or name.startswith("schemdraw."):
            raise ImportError("schemdraw blocked")
sys.meta_path.insert(0, Block())
import axionfem, axionfem.circuit as ck, axionfem.multiport, axionfem.squid, axionfem.draw
assert "schemdraw" not in sys.modules
r = ck.series_readout(L_p_H=1e-6, Q=1e6, f_Hz=1e6, T_K=0.03, L_in_H=2e-9)
try:
    r.draw()
except ImportError as e:
    assert "axionfem[draw]" in str(e), e
    print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    # the last line: some system libraries print notices on stdout when they are loaded (Cray's libsci, for one)
    lines = out.stdout.strip().splitlines()
    assert out.returncode == 0 and lines and lines[-1] == "ok", (out.stdout, out.stderr)
