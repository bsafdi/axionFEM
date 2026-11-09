"""High-frequency torsional mobility: the multi-branch waveguide (A) and the wall-resolved FEM path (B).

docs/physics/torsion_highfreq.md; validation page docs/validation/torsion_highfreq.md (benchmarks/torsion_highfreq.py).
- A: orthonormal Bessel radial shapes and the T(0, 2) cutoff of a 0.5 m steel tube with a 4 cm wall; the T(0,1)
  branch equals the damped single-branch integral; eta -> 0 gives the lossless group-velocity formula; the sampled
  axial kernel equals the closed forms; the finite-tube modal sum tends to A at high modal overlap;
- B: loads on regions, surfaces and curves are applied exactly (force and torque = closed form); B equals the exact
  finite-tube modal sum for volume, inner and outer tractions; backends and frequency-parallel solves agree; the band
  block; the mesh-resolution guard refuses and warns; request errors name the entry.
"""
import math

import numpy as np
import pytest
from scipy import integrate

import axionfem as af
from axionfem.torsion_waveguide import Gaussian, Rect, Ring, Sampled, Tube, TubeLoad

E, NU, RHO = 200e9, 0.3, 7900.0
MU = E / (2 * (1 + NU))
CT = math.sqrt(MU / RHO)


def _gauss(sigma, z0=0.0):
    return f"exp(-((z - ({z0!r}))^2)/(2*{sigma!r}^2))"


def _tube_design(a, t, L, h, order=2, curve=False):
    regions = [{"name": "tube", "shape": {"rect": {"r": [a, a + t], "z": [-L / 2, L / 2]}}, "material": "steel",
                "mesh_size": h},
               {"name": "bore", "shape": {"rect": {"r": [0.0, a], "z": [-L / 2, L / 2]}}, "material": "vacuum",
                "mesh_size": a / 4},
               {"name": "skin", "shape": {"rect": {"r": [a + t, a + 1.2 * t], "z": [-L / 2, L / 2]}},
                "material": "vacuum", "mesh_size": 0.2 * t},
               {"name": "coil", "shape": {"rect": {"r": [2.0, 2.01], "z": [-0.01, 0.01]}}, "material": "winding"}]
    doc = {"schema": "axionfem.design/1", "name": "hf_tube",
           "materials": {"steel": {"kind": "linear", "density": RHO,
                                   "mechanical": {"E_Pa": E, "nu": NU, "alpha": 0.0}}},
           "regions": regions, "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
           "mesh": {"order": order, "size": 0.2, "domain_radius": 5.0}}
    if curve:
        doc["curves"] = [{"name": "liner", "path": {"segment": [[a, -L / 2], [a, L / 2]]}, "material": "copper"}]
    return af.Design(doc)


@pytest.fixture(scope="module")
def small_tube_p3(tmp_path_factory):
    a, t, L = 0.5, 0.01, 0.1
    res = _tube_design(a, t, L, 0.001, order=3).solve(tmp_path_factory.mktemp("hf3"), verbose=0)
    return res, a, t, L


@pytest.fixture(scope="module")
def small_tube(tmp_path_factory):
    a, t, L = 0.5, 0.01, 0.1
    res = _tube_design(a, t, L, 0.001, curve=True).solve(tmp_path_factory.mktemp("hf"), verbose=0)
    return res, a, t, L


# ------------------------------------------------------------------------------------------------ A
def test_radial_shapes_are_orthonormal_and_the_cutoff_matches_the_reference():
    tb = Tube(0.5, 0.04, RHO, E=E, nu=NU)
    assert abs(tb.cutoffs_Hz(2)[1] / 39.049e3 - 1) < 1e-4          # T(0,2) at 39.05 kHz, computed independently
    r = np.linspace(0.5, 0.54, 8001)
    w = np.full(r.size, r[1] - r[0])
    w[0] = w[-1] = w[0] / 2
    S = np.array([tb.mode_shape(i, r) for i in range(6)])
    G = (S * w * r) @ S.T
    assert np.max(np.abs(G - np.eye(6))) < 1e-6
    rod = Tube(0.0, 0.05, RHO, mu=MU)                                  # J_2(kappa b) = 0
    assert abs(rod.kappa(2)[1] * 0.05 / 5.135622301840683 - 1) < 1e-10


@pytest.mark.parametrize("profile,P2", [
    (Gaussian(0.1), lambda k: 2 * math.pi * 0.01 * np.exp(-(k * 0.1) ** 2)),
    (Rect(0.4), lambda k: (0.4 * np.sinc(k * 0.4 / 2 / math.pi)) ** 2)])
def test_T01_branch_equals_the_damped_single_branch_integral(profile, P2):
    """Re Y_d = T^2/(pi J mu) int_0^inf |P|^2 w eta k^2/((k^2 - k0^2)^2 + eta^2 k^4) dk."""
    a, t, eta = 1.0, 0.01, 0.03
    I2 = ((a + t) ** 3 - a ** 3) / 3
    J = 2 * math.pi * ((a + t) ** 4 - a ** 4) / 4
    T = 2 * math.pi * I2 / t
    tb = Tube(a, t, RHO, mu=MU)
    for f in (2.5e3, 6e3):
        w = 2 * math.pi * f
        k0 = w / CT
        fun = lambda k: float(P2(np.array([k]))[0]) * w * eta * k * k / ((k * k - k0 * k0) ** 2 + (eta * k * k) ** 2)  # noqa
        pts = [k0 * (1 - 20 * eta), k0, k0 * (1 + 20 * eta)]
        v = integrate.quad(fun, 0, 3000, points=pts, limit=4000, epsabs=0, epsrel=1e-11)[0]
        v += integrate.quad(fun, 3000, np.inf, limit=200)[0]
        ref = T ** 2 / (math.pi * J * MU) * v
        got = tb.mobility([f], TubeLoad(profile, volume=1 / t), eta, n_branches=1).Y.real[0]
        assert abs(got / ref - 1) < 1e-8


def test_lossless_limit_and_sampled_kernel():
    tb = Tube(0.5, 0.04, RHO, E=E, nu=NU)
    f = np.array([25e3, 60e3, 100e3])
    for ld in (TubeLoad(Gaussian(0.005), inner=1.0), TubeLoad(Gaussian(0.005), volume=1.0)):
        damped = tb.mobility(f, ld, 1e-7).Y.real
        lossless, _ = tb.mobility_lossless(f, ld)
        assert np.max(np.abs(damped / lossless - 1)) < 1e-4
        one = tb.mobility(f, ld, 0.05)
        two = tb.mobility(f, [ld, TubeLoad(Gaussian(0.005), volume=0.0)], 0.05)   # forces the sampled kernel
        # 2e-5 with 100 evanescent branches: the far ones decay over ~0.1 mm, near the grid step
        assert np.max(np.abs(two.Y / one.Y - 1)) < 1e-4
    ring = tb.mobility(f, TubeLoad(Ring(), inner=1.0), 0.05)
    assert np.all(ring.Y.real > 0)


def test_surface_load_drives_the_higher_branches():
    """Above the T(0,2) cutoff an inner traction's Re Y is several times its T(0,1) part."""
    tb = Tube(0.5, 0.04, RHO, E=E, nu=NU)
    m = tb.mobility([20e3, 100e3], TubeLoad(Gaussian(0.005), inner=1.0), 0.1)
    ratio = m.Y.real / m.Y_branch[0].real
    assert ratio[0] < 1.5 and ratio[1] > 5


def test_evanescent_tail_of_a_surface_traction():
    """A surface traction projects with O(1) weight on every branch; the far evanescent branches add
    Re Y ~ eta R_n^2/kappa_n^2, a 1/N tail.  With the analytic tail, 20 evanescent branches agree with 1600."""
    tb = Tube(0.5, 0.01, RHO, E=E, nu=NU)
    f = np.linspace(600e3, 1e6, 41)
    ld = TubeLoad(Gaussian(0.003), inner=1.0)
    ref = tb.mobility(f, ld, 0.1, n_evanescent=1600).Y.real
    n20 = len(tb.kappa(kappa_max=2 * math.pi * 1e6 / CT)) + 20           # 20 evanescent branches
    with_tail = tb.mobility(f, ld, 0.1, n_branches=n20).Y.real
    without = tb.mobility(f, ld, 0.1, n_branches=n20, tail=False).Y.real
    assert np.max(np.abs(with_tail / ref - 1)) < 2e-3
    assert np.mean(without / ref) < 0.995          # 1.2 % low without the tail
    ring = tb.mobility(f, TubeLoad(Ring(), inner=1.0), 0.1)
    assert any("logarithmically" in n for n in ring.notes)


def test_finite_tube_tends_to_the_infinite_tube_at_high_overlap():
    tb = Tube(0.5, 0.01, RHO, E=E, nu=NU)
    f = np.linspace(200e3, 300e3, 401)
    ld = TubeLoad(Gaussian(0.003), inner=1.0)
    A = tb.mobility(f, ld, 0.1).Y.real
    F = tb.finite_tube_mobility(f, ld, 0.1, 0.5).real
    assert abs(np.trapezoid(F, f) / np.trapezoid(A, f) - 1) < 0.01


# ------------------------------------------------------------------------------------------------ B
def test_named_loads_are_applied_exactly(small_tube):
    res, a, t, L = small_tube
    s = 0.003
    gs = math.sqrt(2 * math.pi) * s
    cases = [([{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": _gauss(s)}], 2 * math.pi * a * gs, a),
             ([{"surface": {"on": "tube", "from": "skin"}, "traction_Pa": _gauss(s)}],
              2 * math.pi * (a + t) * gs, a + t),
             # standard precedence: the docstring's unparenthesised form is a Gaussian
             ([{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": f"exp(-z^2/(2*{s!r}^2))"}],
              2 * math.pi * a * gs, a),
             ([{"curve": "liner", "traction_Pa": 2.0}], 2.0 * 2 * math.pi * a * L, a),
             ([{"region": "tube", "g_N_m3": 3.0}], 3.0 * math.pi * ((a + t) ** 2 - a ** 2) * L, None)]
    for loads, force, rarm in cases:
        tr = res.torsion(loads=loads, nmodes=0, dofs=False)
        tot = tr["loads"]["total"]
        assert abs(tot["force_N"] / force - 1) < 1e-10, (loads, tot)
        if rarm is not None:
            assert abs(tot["torque_N_m"] / (force * rarm) - 1) < 1e-10
    # a constant traction on every vacuum-facing face: the inner and outer walls and the two end annuli
    tr = res.torsion(loads=[{"surface": {"on": "tube", "from": "vacuum"}, "traction_Pa": 1.0}])
    total_area = 2 * math.pi * (a + (a + t)) * L + 2 * math.pi * ((a + t) ** 2 - a ** 2)
    assert abs(tr["loads"]["total"]["force_N"] / total_area - 1) < 1e-10


@pytest.mark.parametrize("kind", ["inner", "outer", "volume"])
def test_fem_equals_the_exact_finite_tube(small_tube, kind):
    res, a, t, L = small_tube
    s = 0.003
    loads = {"inner": [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": _gauss(s)}],
             "outer": [{"surface": {"on": "tube", "from": "skin"}, "traction_Pa": _gauss(s)}],
             "volume": [{"region": "tube", "g_N_m3": _gauss(s)}]}[kind]
    ld = {"inner": TubeLoad(Gaussian(s), inner=1.0), "outer": TubeLoad(Gaussian(s), outer=1.0),
          "volume": TubeLoad(Gaussian(s), volume=1.0)}[kind]
    f = np.linspace(100e3, 300e3, 81)
    tr = res.torsion(loads=loads, eta=0.03, band={"f_Hz": [100e3, 300e3], "n": 81})
    Y = np.array([x["Y_re"] for x in tr["harmonic"]["response"]])
    F = Tube(a, t, RHO, E=E, nu=NU).finite_tube_mobility(f, ld, 0.03, L).real
    assert np.max(np.abs(Y / F - 1)) < 1e-2            # 3e-3 at 8 elements per wavelength (P2)
    band = tr["harmonic"]["band"]
    assert abs(band["mean_Re_Y"] / np.trapezoid(F, f) * (f[-1] - f[0]) - 1) < 2e-3
    assert band["n"] == 81 and len(band["subband_means"]) == 8 and band["ripple_sem"] > 0
    assert tr["harmonic"]["resolution"]["elements_per_wavelength_min"] > 6


def test_backends_and_parallel_frequencies_agree(small_tube):
    res, a, t, L = small_tube
    loads = [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": 1.0}]
    freqs = [1.1e5, 2.2e5, 3.3e5]
    ref = res.torsion(loads=loads, eta=0.01, freqs_Hz=freqs, solver={"backend": "eigen"})["harmonic"]
    assert ref["timing"]["backend"] == "eigen-sparselu" and ref["timing"]["symbolic_analyses"] == 1
    others = [{"backend": "eigen", "parallel_frequencies": 3}, {"parallel_frequencies": "auto"}]
    if af.build_info().get("strumpack"):
        others += [{"backend": "strumpack"}, {"backend": "strumpack", "parallel_frequencies": 2}]
    for sv in others:
        h = res.torsion(loads=loads, eta=0.01, freqs_Hz=freqs, solver=sv)["harmonic"]
        if sv.get("parallel_frequencies") == "auto":
            pc = h["timing"]["parallel_choice"]
            assert pc["mode"] == "auto" and pc["chosen_p"] >= 1
            assert pc["chosen_backend"] in ("eigen", "strumpack")
            assert "available_bytes" in pc["memory"]          # 0 without /proc (then auto runs serially)
        for x, y in zip(h["response"], ref["response"]):
            assert abs(x["Y_re"] / y["Y_re"] - 1) < 1e-9 and abs(x["Y_im"] / y["Y_im"] - 1) < 1e-9


def test_mesh_guard_refuses_and_warns(small_tube):
    res, a, t, L = small_tube
    loads = [{"region": "tube", "g_N_m3": 1.0}]
    # h = 1 mm (longest edges ~1.3 mm), P2: lambda_T = c_T/f
    with pytest.raises(Exception, match="elements per shear wavelength"):
        res.torsion(loads=loads, eta=0.01, freqs_Hz=[3e6])
    out = res.torsion(loads=loads, eta=0.01, freqs_Hz=[3e6], mesh_check="warn")
    assert any("elements per shear wavelength" in w for w in out["warnings"])
    out = res.torsion(loads=loads, eta=0.01, freqs_Hz=[3e6], mesh_check="off")
    assert not any("elements per shear wavelength" in w for w in out["warnings"])
    assert out["harmonic"]["resolution"]["elements_per_wavelength_min"] < 2


@pytest.mark.parametrize("loads,match", [
    ([{"region": "bore", "g_N_m3": 1.0}], "not part of the structure"),
    ([{"region": "tube", "traction_Pa": 1.0}], "takes g_N_m3"),
    ([{"surface": {"on": "tube", "from": "nowhere"}, "traction_Pa": 1.0}], "not a region"),
    ([{"curve": "nope", "traction_Pa": 1.0}], "not a curve"),
    ([{"region": "tube", "g_N_m3": "exp(-q)"}], "cannot evaluate"),
    ([{"region": "tube", "surface": {"on": "tube"}, "g_N_m3": 1.0}], "exactly one of"),
    ([{"region": "tube", "g_N_m3": 1.0, "scale": 2}], "unknown key 'scale'")])
def test_load_errors_name_the_entry(small_tube, loads, match):
    res = small_tube[0]
    with pytest.raises(Exception, match=match):
        res.torsion(loads=loads, freqs_Hz=[1e4])


def test_band_requires_odd_n(small_tube):
    res = small_tube[0]
    with pytest.raises(Exception, match="odd"):
        res.torsion(loads=[{"region": "tube", "g_N_m3": 1.0}], band={"f_Hz": [1e4, 2e4], "n": 10})


# -------------------------------------------------------------------------- B against the finite tube, band diagnostics
def _band_mean(f, y):
    return float(np.trapezoid(y, f) / (f[-1] - f[0]))


def test_off_centre_load_fem_equals_the_finite_tube(small_tube):
    """The sign of the axial offset in F's cosine transform (a centred load cannot see it)."""
    res, a, t, L = small_tube
    s, z0 = 0.003, 0.025
    f = np.linspace(100e3, 300e3, 81)
    tr = res.torsion(loads=[{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": _gauss(s, z0)}], eta=0.03,
                     band={"f_Hz": [100e3, 300e3], "n": 81})
    Y = np.array([x["Y_re"] for x in tr["harmonic"]["response"]])
    F = Tube(a, t, RHO, E=E, nu=NU).finite_tube_mobility(f, TubeLoad(Gaussian(s, z0), inner=1.0), 0.03, L).real
    assert np.max(np.abs(Y / F - 1)) < 1e-2
    assert abs(_band_mean(f, Y) / _band_mean(f, F) - 1) < 2e-3
    Fm = Tube(a, t, RHO, E=E, nu=NU).finite_tube_mobility(f, TubeLoad(Gaussian(s, -z0), inner=1.0), 0.03, L).real
    assert abs(_band_mean(f, Fm) / _band_mean(f, F) - 1) < 1e-9     # the mirror load: same mobility (symmetry)


def test_fem_equals_the_finite_tube_above_several_cutoffs_at_high_damping(small_tube_p3):
    """At eta = 0.1 above two cutoffs the far evanescent branches are 0.5 % of Re Y, so F without its
    branch tail fails a 1e-3 comparison with B (P3, about 5 elements per wavelength)."""
    res, a, t, L = small_tube_p3
    s = 0.003
    f = np.linspace(330e3, 460e3, 41)
    tr = res.torsion(loads=[{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": _gauss(s)}], eta=0.1,
                     band={"f_Hz": [330e3, 460e3], "n": 41})
    tb = Tube(a, t, RHO, E=E, nu=NU)
    ld = TubeLoad(Gaussian(s), inner=1.0)
    F = tb.finite_tube_mobility(f, ld, 0.1, L).real
    F_no_tail = tb.finite_tube_mobility(f, ld, 0.1, L, branch_tail=False).real
    mB = tr["harmonic"]["band"]["mean_Re_Y"]
    assert abs(mB / _band_mean(f, F) - 1) < 1e-3
    assert abs(mB / _band_mean(f, F_no_tail) - 1) > 2e-3


@pytest.mark.parametrize("sigma,t", [(0.003, 0.01), (1e-4, 0.04), (3e-5, 0.01)])
def test_waveguide_default_branch_count_equals_brute_force(sigma, t):
    """A at its default branch count (with the tail) against four times as many branches."""
    tb = Tube(0.5, t, RHO, E=E, nu=NU)
    f = np.linspace(60e3, 120e3, 5)
    ld = TubeLoad(Gaussian(sigma), inner=1.0)
    a = tb.mobility(f, ld, 0.1)
    b = tb.mobility(f, ld, 0.1, n_branches=4 * len(a.kappa))
    assert np.max(np.abs(a.Y.real / b.Y.real - 1)) < 3e-4
    assert a.kappa[-1] * sigma >= 5 or len(a.kappa) >= 100


def test_band_diagnostics_are_what_they_say(small_tube):
    """quadrature_uncertainty, ripple_sem and edge_estimate recomputed from the rows."""
    res, a, t, L = small_tube
    f0, f1, n = 100e3, 300e3, 25          # undersampled: df = 8.3 kHz > eta f0/2 = 1.5 kHz
    tr = res.torsion(loads=[{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": _gauss(0.003)}], eta=0.03,
                     band={"f_Hz": [f0, f1], "n": n, "length_m": L})
    band = tr["harmonic"]["band"]
    f = np.array([x["f_Hz"] for x in tr["harmonic"]["response"]])
    y = np.array([x["Y_re"] for x in tr["harmonic"]["response"]])
    full, coarse = _band_mean(f, y), _band_mean(f[::2], y[::2])
    assert abs(band["mean_Re_Y"] / full - 1) < 1e-12
    assert abs(band["quadrature_uncertainty"] - abs(full - coarse)) < 1e-12 * abs(full)
    assert band["quadrature_uncertainty"] > 1e-3 * full        # visibly undersampled
    assert any("undersampled" in w for w in tr["warnings"])
    x = (f - 0.5 * (f0 + f1)) / (0.5 * (f1 - f0))
    V = np.vstack([np.ones_like(x), x, x * x]).T
    resid = y - V @ np.linalg.lstsq(V, y, rcond=None)[0]
    means = np.array([resid[b * n // 8:(b + 1) * n // 8].mean() for b in range(8)])
    assert abs(band["ripple_sem"] / (means.std(ddof=1) / math.sqrt(8)) - 1) < 1e-6
    rms = math.sqrt(np.mean(resid ** 2))
    assert abs(band["ripple_rms"] / rms - 1) < 1e-6
    assert abs(band["edge_estimate"] / (rms * math.sqrt(2) * (CT / L) / (math.pi * (f1 - f0))) - 1) < 1e-6
    short = res.torsion(loads=[{"region": "tube", "g_N_m3": 1.0}], eta=0.03, band={"f_Hz": [f0, f1], "n": 9})
    assert short["harmonic"]["band"]["ripple_sem"] is None and "edge_estimate" not in short["harmonic"]["band"]


def test_single_frequencies_are_flagged_and_mesh_check_is_always_validated(small_tube):
    res = small_tube[0]
    out = res.torsion(loads=[{"region": "tube", "g_N_m3": np.float32(2.0)}], eta=0.01, freqs_Hz=[1e5])
    assert "not certified" in out["harmonic"]["note"]
    assert abs(out["loads"]["total"]["force_N"] / (2.0 * math.pi * (0.51 ** 2 - 0.25) * 0.1) - 1) < 1e-10
    with pytest.raises(Exception, match="mesh_check must be"):
        res.torsion(nmodes=1, mesh_check="maybe")


def test_finite_tube_end_tolerance():
    tb = Tube(0.5, 0.01, RHO, E=E, nu=NU)
    near = tb.finite_tube_mobility([1e5], TubeLoad(Gaussian(0.003, 0.23), inner=1.0), 0.03, 0.5)
    assert near.real[0] > 0                              # 6.7 sigma from the end: accepted (was refused)
    with pytest.raises(ValueError, match="of its peak at a tube end"):
        tb.finite_tube_mobility([1e5], TubeLoad(Gaussian(0.003, 0.245), inner=1.0), 0.03, 0.5)


# -------------------------------------------------------------------- transforms, branch counts, memory, backend choice
def test_gaussian_cosine_transform_against_quadrature():
    """The finite tube's axial coefficients of an off-centre Gaussian (an all-Gaussian load cannot see the
    sign of the offset: the mirror load has the same mobility)."""
    g = Gaussian(0.004, 0.031)
    z = np.linspace(-0.1, 0.1, 400001)
    beta = np.array([0.0, 37.0, 150.0, 410.0])
    quad = [np.trapezoid(g.samples(z) * np.cos(b * (z + 0.1)), z) for b in beta]
    assert np.allclose(g.cosine(beta, -0.1), quad, rtol=1e-9, atol=1e-12)


def test_waveguide_at_high_frequency_against_a_fixed_reference():
    """A with its default branches at 0.6-1 MHz on a 1 cm wall against a brute-force sum (1000 and
    2000 branches without the tail, extrapolated in 1/N)."""
    tb = Tube(0.5, 0.01, RHO, E=E, nu=NU)
    f = np.linspace(600e3, 1e6, 5)
    ld = TubeLoad(Gaussian(0.003), inner=1.0)
    r1 = tb.mobility(f, ld, 0.1, n_branches=1000, tail=False).Y.real
    r2 = tb.mobility(f, ld, 0.1, n_branches=2000, tail=False).Y.real
    ref = 2 * r2 - r1
    assert np.max(np.abs(tb.mobility(f, ld, 0.1).Y.real / ref - 1)) < 1e-4


def test_waveguide_thick_wall_at_high_frequency():
    """26 branches propagate on a 4 cm wall at 1 MHz; n_prop + 100 branches were 7.7e-4 short, the 20 k_T rule
    is not (against a brute-force sum extrapolated in 1/N)."""
    tb = Tube(0.5, 0.04, RHO, E=E, nu=NU)
    f = np.linspace(600e3, 1e6, 3)
    ld = TubeLoad(Gaussian(0.003), inner=1.0)
    r1 = tb.mobility(f, ld, 0.1, n_branches=2000, tail=False).Y.real
    r2 = tb.mobility(f, ld, 0.1, n_branches=4000, tail=False).Y.real
    assert np.max(np.abs(tb.mobility(f, ld, 0.1).Y.real / (2 * r2 - r1) - 1)) < 2e-4


@pytest.mark.parametrize("outer", [1.0, -0.5])
def test_two_wall_tail_against_brute_force(outer):
    """With tractions on both walls R_n alternates with the parity of n; the tail must use both parities."""
    tb = Tube(0.5, 0.01, RHO, E=E, nu=NU)
    f = np.linspace(600e3, 1e6, 5)
    ld = TubeLoad(Gaussian(0.003), inner=1.0, outer=outer)
    r1 = tb.mobility(f, ld, 0.1, n_branches=1000, tail=False).Y.real
    r2 = tb.mobility(f, ld, 0.1, n_branches=2000, tail=False).Y.real
    ref = 2 * r2 - r1
    a = tb.mobility(f, ld, 0.1)
    b = tb.mobility(f, ld, 0.1, n_branches=len(a.kappa) + 1)       # the other parity of N
    assert np.max(np.abs(a.Y.real / ref - 1)) < 2e-4
    assert np.max(np.abs(b.Y.real / ref - 1)) < 2e-4


def test_sampled_scale_and_max_modes():
    """A sampled profile's axial scale is spectral, not its grid step; F refuses too many modes."""
    z = np.linspace(-0.05, 0.05, 20001)
    sp = Sampled(z, np.exp(-z ** 2 / (2 * 0.003 ** 2)))
    sc = sp.extent()[1]
    assert 0.3 * 0.003 < sc < 2 * 0.003
    tb = Tube(0.5, 0.01, RHO, E=E, nu=NU)
    with pytest.raises(ValueError, match="max_modes"):
        tb.finite_tube_mobility([1e5], TubeLoad(Gaussian(0.003), inner=1.0), 0.03, 0.5, max_modes=1000)


def _fake_proc(root, cgroup_line, status_rss_kb=100_000, meminfo_kb=50_000_000):
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text(cgroup_line + "\n" if cgroup_line else "")
    (proc / "self" / "status").write_text(f"Name: x\nVmRSS:\t{status_rss_kb} kB\n")
    (proc / "meminfo").write_text(f"MemTotal: 99999999 kB\nMemAvailable: {meminfo_kb} kB\n")
    return proc


def _reader(**opts):
    import json
    return json.loads(af._core.torsion_available_memory(json.dumps(opts)))


def test_memory_reader(tmp_path):
    """The job-memory reader on fake /proc and cgroup trees and environments."""
    GiB = 1024 ** 3
    # cgroup v2: the limit on an ancestor (job_1), the task's own memory.max "max"; usage less inactive file cache
    proc = _fake_proc(tmp_path / "a", "0::/slice/job_1/step_0/task_0")
    cg = tmp_path / "a" / "cg"
    for d, mx in (("slice/job_1/step_0/task_0", "max"), ("slice/job_1/step_0", "max"), ("slice/job_1", str(8 * GiB))):
        (cg / d).mkdir(parents=True, exist_ok=True)
        (cg / d / "memory.max").write_text(mx + "\n")
    (cg / "slice/job_1/memory.current").write_text(str(3 * GiB))
    (cg / "slice/job_1/memory.stat").write_text(f"anon 1\ninactive_file {1 * GiB}\n")
    out = _reader(proc=str(proc), cgroup=str(cg), env={"SLURM_MEM_PER_NODE": "1000000"})
    assert out["found"] and out["source"] == "cgroup"
    assert abs(out["available_bytes"] - 6 * GiB) < 1                # 8 - (3 - 1)
    assert "SLURM_ignored" in out["sources"]
    # a container namespace: the cgroup is "/" and its limit sits at the root of the cgroup mount
    proc = _fake_proc(tmp_path / "b", "0::/")
    cg = tmp_path / "b" / "cg"
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text(str(4 * GiB))
    out = _reader(proc=str(proc), cgroup=str(cg), env={})
    assert out["found"] and abs(out["available_bytes"] - (4 * GiB - 100_000 * 1024)) < 1
    # no cgroup limit: SLURM; a limit below the RSS leaves 0 and is kept (not replaced by MemAvailable)
    proc = _fake_proc(tmp_path / "c", "")
    out = _reader(proc=str(proc), cgroup=str(tmp_path / "none"), env={"SLURM_MEM_PER_NODE": "64"})
    assert out["found"] and out["available_bytes"] == 0.0
    # SLURM_MEM_PER_CPU times SLURM_CPUS_ON_NODE (SLURM_CPUS_PER_TASK is unset without -c)
    out = _reader(proc=str(proc), cgroup=str(tmp_path / "none"),
                  env={"SLURM_MEM_PER_CPU": "1024", "SLURM_CPUS_ON_NODE": "16"})
    assert abs(out["available_bytes"] - (16 * GiB - 100_000 * 1024)) < 1
    assert out["sources"]["SLURM_cpus_from"] == "SLURM_CPUS_ON_NODE"
    # nothing but MemAvailable; and no /proc at all (macOS, Windows): not found, available 0, auto runs serially
    out = _reader(proc=str(proc), cgroup=str(tmp_path / "none"), env={})
    assert out["source"] == "MemAvailable" and out["available_bytes"] == 50_000_000 * 1024
    out = _reader(proc=str(tmp_path / "noproc"), cgroup=str(tmp_path / "none"), env={})
    assert out["found"] is False and out["available_bytes"] == 0.0


def _choice(**kw):
    import json
    return json.loads(af._core.torsion_auto_choice(json.dumps(kw)))


def test_auto_decisions():
    """When to probe Eigen, and which backend wins, from measured inputs."""
    base = dict(stage="probe", strumpack_available=True, free_dofs=3.5e5, matrix_nonzeros=3.5e5 * 25,
                remaining=200, available_bytes=30e9)
    assert _choice(**base, strumpack_fill_per_dof=61)["probe_eigen"] is True           # thin wall
    assert _choice(**base, strumpack_fill_per_dof=107)["probe_eigen"] is False         # compact section
    assert _choice(**{**base, "available_bytes": 1e9}, strumpack_fill_per_dof=61)["probe_eigen"] is False
    assert _choice(**{**base, "remaining": 5}, strumpack_fill_per_dof=61)["probe_eigen"] is False
    assert _choice(stage="probe", strumpack_available=False)["probe_eigen"] is True
    ch = dict(stage="choose", threads=16, remaining=200, available_bytes=30e9, strumpack_available=True,
              strumpack_factor_s=1.08, eigen_probed=True)
    thin = _choice(**ch, eigen_s=3.0, eigen_memory_bytes=1.3e9)
    assert thin["backend"] == "eigen" and thin["p"] == 8
    sec5 = _choice(**ch, eigen_s=14.2, eigen_memory_bytes=3e9)                          # 5 cm x 0.4 m: STRUMPACK
    assert sec5["backend"] == "strumpack" and sec5["p"] == 1
    tight = _choice(**{**ch, "available_bytes": 2e9}, eigen_s=3.0, eigen_memory_bytes=1.3e9)
    assert tight["backend"] == "strumpack"
    assert _choice(**{**ch, "eigen_probed": False}, eigen_s=0.0, eigen_memory_bytes=0.0)["backend"] == "strumpack"
    one = _choice(**{**ch, "strumpack_available": False, "available_bytes": 1.5e9}, eigen_s=3.0,
                  eigen_memory_bytes=1.3e9)
    assert one["backend"] == "eigen" and one["p"] == 1


def test_auto_end_to_end_and_the_band_default(small_tube):
    """The auto mode on a real (small) mesh: probes recorded, the memory estimate at least the factor's size,
    results equal to serial; the OpenMP thread count restored; torsion_band defaults to serial STRUMPACK."""
    res = small_tube[0]
    loads = [{"surface": {"on": "tube", "from": "bore"}, "traction_Pa": 1.0}]
    freqs = list(np.linspace(1.1e5, 3.3e5, 10))
    threads = af._core.get_threads()
    ref = res.torsion(loads=loads, eta=0.01, freqs_Hz=freqs, solver={"backend": "eigen"})["harmonic"]["response"]
    h = res.torsion(loads=loads, eta=0.01, freqs_Hz=freqs, solver={"parallel_frequencies": "auto"})["harmonic"]
    assert af._core.get_threads() == threads
    pc = h["timing"]["parallel_choice"]
    nfree = h["timing"]["free_dofs"]
    if pc["probe_eigen_s"] is not None:
        assert pc["eigen_memory_per_solve_bytes"] >= 1.5 * 20.0 * pc["probe_eigen_factor_nnz_per_dof"] * nfree
    if af.build_info().get("strumpack"):
        assert pc["probe_strumpack_factor_s"] > 0
    for x, y in zip(h["response"], ref):
        assert abs(x["Y_re"] / y["Y_re"] - 1) < 1e-9
    b = res.torsion_band(1.0e5, 1.2e5, loads=loads, eta=0.01, n=9)
    want = "strumpack" if af.build_info().get("strumpack") else "eigen-sparselu"
    assert b["harmonic"]["timing"]["backend"] == want and b["harmonic"]["timing"]["parallel_frequencies"] == 1
    assert "parallel_choice" not in b["harmonic"]["timing"]          # serial by default, not auto


def test_guard_refuses_an_under_resolved_loaded_region(tmp_path):
    """A finely meshed tube with a coarse steel pad: loaded, the pad's own 1st percentile refuses; not
    loaded, the coarse pad only warns (refusing on the minimum would refuse here)."""
    a, t, L = 0.5, 0.01, 0.1
    doc = dict(_tube_design(a, t, L, 0.001).doc)
    doc["regions"] = list(doc["regions"]) + [
        {"name": "pad", "shape": {"rect": {"r": [a + 1.2 * t, a + 1.2 * t + 0.02], "z": [-0.01, 0.01]}},
         "material": "steel", "mesh_size": 0.02}]
    res = af.Design(doc).solve(tmp_path, verbose=0)
    f = [4.0e5]       # the tube: about 6 elements per wavelength; the 2 cm pad meshed at 2 cm: below 1
    with pytest.raises(Exception, match="loaded region 'pad'"):
        res.torsion(loads=[{"region": "pad", "g_N_m3": 1.0}], eta=0.03, freqs_Hz=f)
    out = res.torsion(loads=[{"region": "tube", "g_N_m3": 1.0}], eta=0.03, freqs_Hz=f)
    assert any("proceeds" in w for w in out["warnings"])
    assert out["harmonic"]["resolution"]["per_region"]["pad"]["min"] < out["harmonic"]["resolution"]["refuse_below"]
