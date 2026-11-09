"""Physics regression tests with independent references.

Each test compares against a closed-form or independently computed value, or
guards against a known failure mode: the persistent-circuit sign, order-4 axis
evaluation, solver status reporting, iron in a toroidal field and sheets in the
G problem, sharp-corner flags, the CORE template's iron caps, and the gridded
background-field interpolation.
"""
import json
import math

import numpy as np
import pytest
from scipy import integrate, special

import axionfem as af

MU0 = 1.25663706212e-6  # H/m, CODATA 2018 (same value as the C++ core)


def mu0_exact():
    return MU0


# ---------------------------------------------------------------- coaxial coils (Maxwell)
def maxwell_M(a, b, d):
    """Mutual inductance of two coaxial filamentary loops (radii a, b, separation d)."""
    k2 = 4 * a * b / ((a + b) ** 2 + d ** 2)
    k = math.sqrt(k2)
    return mu0_exact() * math.sqrt(a * b) * ((2 / k - k) * special.ellipk(k2) - (2 / k) * special.ellipe(k2))


def coil_M(r1, z1, s1, r2, z2, s2, n=8):
    """Average of maxwell_M over two square cross-sections (uniform current), Gauss-Legendre n^4."""
    x, w = np.polynomial.legendre.leggauss(n)
    tot = 0.0
    for i, (xa, wa) in enumerate(zip(x, w)):
        for xb, wb in zip(x, w):
            ra, za = r1 + 0.5 * s1 * xa, z1 + 0.5 * s1 * xb
            for xc, wc in zip(x, w):
                for xd, wd in zip(x, w):
                    rb, zb = r2 + 0.5 * s2 * xc, z2 + 0.5 * s2 * xd
                    tot += wa * wb * wc * wd * maxwell_M(ra, rb, zb - za)
    return tot / 16.0


@pytest.mark.parametrize("r2,dz", [(0.30, 0.20), (0.80, 0.10)])
def test_mutual_inductance_vs_maxwell(tmp_path, r2, dz):
    s1, s2 = 0.02, 0.01
    doc = {"schema": "axionfem.design/1", "name": "mutual",
           "regions": [{"name": "c1", "shape": {"rect": {"r": [0.5 - s1 / 2, 0.5 + s1 / 2], "z": [-s1 / 2, s1 / 2]}},
                        "material": "winding"},
                       {"name": "c2", "shape": {"rect": {"r": [r2 - s2 / 2, r2 + s2 / 2], "z": [dz - s2 / 2, dz + s2 / 2]}},
                        "material": "winding"}],
           "windings": [{"region": "c1", "type": "azimuthal", "ampere_turns": 1.0},
                        {"region": "c2", "type": "azimuthal", "ampere_turns": 0.0}],
           "mesh": {"order": 3, "size": 0.06, "surface_size": 0.004}}
    r = af.Design(doc).solve(tmp_path / "m", verbose=0)
    M_fe = r.data["windings"]["c2"]["flux_per_turn_Wb"]  # coil 1 carries 1 A-turn
    M_ref = coil_M(0.5, 0.0, s1, r2, dz, s2)
    assert M_fe == pytest.approx(M_ref, rel=1e-6)


# ---------------------------------------------------------------- CORE pickup inductance (exact)
def core_Lp_exact(Rc=1.0, tw=0.02, cl=0.05, w=0.01, b=2.5, h=2.0, f=0.05, gn=0.01):
    """Sharp-lipped CORE: interior (rounded rectangle, corner radius f - w) plus the neck profile."""
    a = Rc + tw + cl + w
    rho = f - w
    dA = integrate.quad(lambda r: (rho - math.sqrt(max(rho ** 2 - (a + rho - r) ** 2, 0.0))) / r, a, a + rho,
                        epsabs=1e-15, epsrel=1e-13)[0]
    dB = integrate.quad(lambda r: (rho - math.sqrt(max(rho ** 2 - (r - (b - rho)) ** 2, 0.0))) / r, b - rho, b,
                        epsabs=1e-15, epsrel=1e-13)[0]
    interior = h * math.log(b / a) - 2 * dA - 2 * dB
    r0, r1 = a - w, a
    neck = gn * integrate.quad(lambda r: ((r * r - r0 * r0) / (r1 * r1 - r0 * r0)) ** 2 / r, r0, r1,
                               epsabs=1e-16, epsrel=1e-13)[0]
    return mu0_exact() / (2 * math.pi) * (interior + neck)


def test_core_pickup_inductance_exact(tmp_path):
    r = af.template("core", fl=0).solve(tmp_path / "c", verbose=0)
    assert r.L_p == pytest.approx(core_Lp_exact(), rel=1e-7)


# ---------------------------------------------------------------- persistent circuits
def _coil(doc_extra=None, flux=0.0, iron=False):
    regions = [{"name": "pc", "shape": {"rect": {"r": [0.5, 0.52], "z": [-0.05, 0.05]}}, "material": "winding"}]
    if iron:
        regions.append({"name": "core", "shape": {"rect": {"r": [0.0, 0.3], "z": [-0.2, 0.2]}}, "material": "iron"})
    doc = {"schema": "axionfem.design/1", "name": "pc",
           "regions": regions,
           "windings": [{"region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop"}],
           "circuits": [{"name": "loop", "type": "persistent", "flux_linkage": flux}],
           "mesh": {"order": 3, "size": 0.08, "surface_size": 0.01}}
    doc.update(doc_extra or {})
    return af.Design(doc)


def _circuit(res):
    return res.data["static"]["poloidal"]["solver"]["circuits"][0]


def test_persistent_nonzero_linkage(tmp_path):
    # self-inductance from a powered single coil (100 turns, 1 A)
    self_doc = _coil().doc
    self_doc.pop("circuits")
    self_doc["windings"] = [{"region": "pc", "type": "azimuthal", "ampere_turns": 100.0}]
    rs = af.Design(self_doc).solve(tmp_path / "s", verbose=0)
    L = 100 * rs.data["windings"]["pc"]["flux_per_turn_Wb"]  # per ampere of coil current
    rp = _coil(flux=1e-3).solve(tmp_path / "p", verbose=0)
    c = _circuit(rp)
    assert c["flux_linkage_Wb"] == pytest.approx(1e-3, rel=1e-8)
    assert c["current_A"] == pytest.approx(1e-3 / L, rel=1e-6)
    assert c["current_A"] > 0
    assert not rp.data["warnings"]


def test_persistent_with_iron_converges(tmp_path):
    rp = _coil(flux=0.05, iron=True).solve(tmp_path / "pi", verbose=0)
    c = _circuit(rp)
    sv = rp.data["static"]["poloidal"]["solver"]
    assert sv["converged"]
    assert c["flux_linkage_Wb"] == pytest.approx(0.05, rel=1e-7)
    assert c["current_A"] > 0
    assert not rp.data["warnings"]
    # the iron force is a Maxwell-stress estimate, and the force closure says so
    assert any("nonlinear iron" in x for x in rp.forces["closure"].get("caveats", [])), rp.forces["closure"]


def test_persistent_loop_screens_background(tmp_path):
    B0 = 0.01
    rp = _coil(doc_extra={"background": {"type": "uniform", "Bz": B0}}, flux=0.0).solve(tmp_path / "bg", verbose=0)
    c = _circuit(rp)
    assert abs(c["flux_linkage_Wb"]) < 1e-9
    assert c["current_A"] < 0  # opposes the +z background
    # a design with a winding has W_mag and a cost, which leave the background magnet out
    assert any("W_mag and cost exclude the background magnet" in w for w in rp.data["warnings"]), rp.data["warnings"]
    Bz = rp.sample([0.0], [0.0])["Bz"][0]
    assert 0 < Bz < B0


# ---------------------------------------------------------------- axis evaluation at order 4
def test_order4_axis_field_finite(tmp_path):
    r = af.template("core", mesh={"order": 4}).solve(tmp_path / "o4", verbose=0)
    assert r.data["fields"]["B_max_T"] < 1.0
    b = r.benchmark("optimistic")
    assert b["B_max_at_req_T"] < 100.0


# ---------------------------------------------------------------- toroidal field: iron and sheets
def test_iron_in_toroidal_field_saturates(tmp_path):
    d = af.template("toroid")
    d.doc["regions"].append({"name": "fe", "shape": {"rect": {"r": [1.3, 1.4], "z": [-0.1, 0.1]}},
                             "material": "iron"})
    r = d.solve(tmp_path / "tfe", verbose=0)
    NI = 1e6
    H = NI / (2 * math.pi * 1.3)  # H = G/r at the block's inner face (enclosed current NI)
    mu0 = mu0_exact()
    Bs, mur = 2.0, 1000.0
    B_ref = Bs + mu0 * (H - Bs / (mu0 * mur))
    assert r.data["fields"]["by_region_T"]["fe"] == pytest.approx(B_ref, rel=1e-6)
    assert r.data["linear"] is False


_IRON_CORE_SCRIPT = r"""
import resource
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))   # before any library starts threads
import json, sys
import axionfem as af
d = af.template("toroid", R=1.0, H=1.0, NI=1e6, y="exp(-0.5)",
                mesh={"order": 1, "size": 0.05, "surface_size": 0.05})
d.doc["regions"].insert(0, {"name": "core", "shape": {"rect": {"r": ["y*R", "R"], "z": ["-H/2", "H/2"]}},
                            "material": "iron", "mesh_size": 0.006})
r = d.solve(sys.argv[1], verbose=0)
# peak RSS of this process image: VmHWM is reset at exec, whereas ru_maxrss keeps the forking parent's peak
hwm = [int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM:")][0]
print(json.dumps({"B_core": r.data["fields"]["by_region_T"]["core"], "L_p": r.L_p,
                  "elements": json.load(open(sys.argv[1] + "/mesh_info.json"))["elements_2d"],
                  "peak_rss_kB": hwm}))
"""


@pytest.mark.skipif(__import__("sys").platform != "linux", reason="RLIMIT_AS is enforced on Linux only")
def test_iron_filled_toroid_memory_is_linear(tmp_path):
    """Iron filling the field box of a magnetic-core toroid.

    If the secant loop of the static toroidal field gave every iron element a closure holding its own copy of the
    per-element G vector, or the RF permeability closure (per-element G and B-H laws) were copied into the role of every
    iron element of the pickup and full-wave problems, memory would grow like (iron elements) x (all elements).
    Here 25 390 of 38 031 elements are iron, so such copies would need ~7.7 GB for the first copy set alone; the shared
    data needs ~0.15 GB. The solve runs in a subprocess under a 3 GB address-space limit (set in the child, with thread
    counts and malloc arenas pinned, because reserved address space grows with the thread count even when resident
    memory does not), and its peak resident memory must stay below 1.5 GiB.
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "OMP_STACKSIZE"}
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MALLOC_ARENA_MAX="2", AXF_DEVICE="cpu")
    wd = str(tmp_path / "fe_core")
    p = subprocess.run([sys.executable, "-c", _IRON_CORE_SCRIPT, wd], capture_output=True, text=True, env=env,
                       timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["elements"] > 30000
    assert out["peak_rss_kB"] < 1.5 * 1024**2
    # inner face of the box: H = NI/(2 pi y R), B on the knee law above saturation
    NI, y = 1e6, math.exp(-0.5)
    H = NI / (2 * math.pi * y)
    mu0 = mu0_exact()
    B_ref = 2.0 + mu0 * (H - 2.0 / (mu0 * 1000.0))
    assert out["B_core"] == pytest.approx(B_ref, rel=1e-6)


def test_sheet_with_poloidal_pickup_rejected():
    d = af.template("core")
    d.doc["curves"] = [{"name": "s", "path": {"segment": [[1.5, -0.5], [1.5, 0.5]]}, "material": "nb"}]
    with pytest.raises(Exception, match="sheet"):
        d.solve("/tmp/axf_should_not_run", verbose=0)


# ---------------------------------------------------------------- sharp corners
def test_sharp_neck_lips_flagged(tmp_path):
    sharp = af.template("core", fl=0).solve(tmp_path / "sh", verbose=0)
    round_ = af.template("core").solve(tmp_path / "rd", verbose=0)
    corners = sharp.data["fields"]["superconductor_sharp_corners"]
    assert len(corners) == 4 and all(abs(c["interior_angle_deg"] - 90) < 1 for c in corners)
    assert round_.data["fields"]["superconductor_sharp_corners"] == []
    loc = round_.data["fields"]["superconductor_surface_location"]["cavity"]
    assert set(loc) == {"r_m", "z_m"}


# ---------------------------------------------------------------- CORE iron caps
def test_core_iron_caps_exist(tmp_path):
    r0 = af.template("core").solve(tmp_path / "n", verbose=0)
    r1 = af.template("core", iron_t=0.1).solve(tmp_path / "i", verbose=0)
    assert r1.data["geometry"]["iron_mass_kg"] > 1e3
    assert r1.Psi > 1.2 * r0.Psi
    # the switch also works through Design.set on an existing design
    r2 = af.template("core").set(iron_t=0.1).solve(tmp_path / "s", verbose=0)
    assert r2.Psi == pytest.approx(r1.Psi, rel=1e-12)


# ---------------------------------------------------------------- background-field interpolation
def _loop_field(r, z, a=1.0, I=1e5):
    r = np.maximum(r, 1e-12)
    m = 4 * a * r / ((a + r) ** 2 + z ** 2)
    K, E = special.ellipk(m), special.ellipe(m)
    s = np.sqrt((a + r) ** 2 + z ** 2)
    d = (a - r) ** 2 + z ** 2
    Bz = MU0 * I / (2 * np.pi) / s * (K + (a * a - r * r - z * z) / d * E)
    Br = MU0 * I / (2 * np.pi * r) * z / s * (-K + (a * a + r * r + z * z) / d * E)
    return Br, Bz


@pytest.mark.parametrize("h,tol", [(0.05, 3e-4), (0.025, 4e-5)])
def test_background_hermite_interpolation(tmp_path, h, tol):
    """Gridded current-loop field: bicubic Hermite psi(r^2, z) gives O(h^3) fields, including near the axis."""
    rg = np.arange(0, 0.8 + 1e-9, h)
    zg = np.arange(-0.5, 0.5 + 1e-9, h)
    R, Z = np.meshgrid(rg, zg)
    Br, Bz = _loop_field(R, Z)
    Br[:, 0] = 0.0
    doc = {"schema": "axionfem.design/1", "name": "bg",
           "regions": [{"name": "dummy", "shape": {"rect": {"r": [2.0, 2.1], "z": [-0.05, 0.05]}}, "material": "vacuum"}],
           "background": {"type": "grid", "r": rg.tolist(), "z": zg.tolist(), "Br": Br.ravel().tolist(),
                          "Bz_grid": Bz.ravel().tolist()},
           "mesh": {"order": 2, "size": 0.3}}
    res = af.Design(doc).solve(tmp_path / "bg", verbose=0)
    rng = np.random.default_rng(1)
    pr, pz = rng.uniform(0.0, 0.7, 200), rng.uniform(-0.4, 0.4, 200)
    s = res.sample(pr.tolist(), pz.tolist())
    br, bz = _loop_field(pr, pz)
    err = np.hypot(np.array(s["Br"]) - br, np.array(s["Bz"]) - bz).max() / np.hypot(br, bz).max()
    assert err < tol
