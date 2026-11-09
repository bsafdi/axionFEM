"""Axisymmetric m = 0 torsional dynamics (Results.torsion, cpp/src/axi_torsion.cpp).

Fast closed-form checks, a subset of benchmarks/torsion.py (docs/validation/torsion.md):
- the rigid rotation u_phi = c r is a zero mode, and the free-free torsional modes of a tube are f_n = n c_T/(2 L);
- the free-edge thin-disc modes k a J_0(k a) = 2 J_1(k a);
- harmonic(): the generalised receptance q equals the complete modal sum sum_a c_a^2/(lam_a (1 - i eta) - w^2),
  time dependence e^{-i omega t} (Y = -i omega q; a mass-controlled response has Im Y > 0);
- the band-mean mobility of a Gaussian ring load against the damped infinite T(0,1) waveguide (+-1 %) and the
  lossless asymptote (+-3 %), at a participating modal overlap of 0.77-1.5;
- the waveguide asymptote's validity range: a surface traction above the thickness-shear cutoff exceeds it;
- per-region overrides, strict request keys, the applied load's integrals, the inertia completeness check, the
  rigid-mode classification (one mode, two bodies) and the warnings for regions without mechanical data or density.
"""
import math

import numpy as np
import pytest
from scipy import integrate, optimize, special

import axionfem as af

E, NU, RHO = 200e9, 0.3, 7900.0
MU = E / (2 * (1 + NU))
CT = math.sqrt(MU / RHO)


def _design(name, rect, size, order=2, extra_materials=None, extra_regions=()):
    materials = {"steel": {"kind": "linear", "density": RHO,
                           "mechanical": {"E_Pa": E, "nu": NU, "alpha": 0.0}}}
    materials.update(extra_materials or {})
    regions = [{"name": "body", "shape": {"rect": rect}, "material": "steel", "mesh_size": size},
               {"name": "coil", "shape": {"rect": {"r": [40.0, 40.05], "z": [-0.05, 0.05]}},
                "material": "winding"}]
    regions += list(extra_regions)
    return af.Design({"schema": "axionfem.design/1", "name": name, "materials": materials, "regions": regions,
                      "windings": [{"region": "coil", "type": "azimuthal", "ampere_turns": 1.0}],
                      "mesh": {"order": order, "size": 2.0, "domain_radius": 120.0}})


@pytest.fixture(scope="module")
def tube(tmp_path_factory):
    a, t, L = 1.0, 0.02, 4.0
    res = _design("torsion_tube", {"r": [a, a + t], "z": [-L / 2, L / 2]}, 0.2).solve(
        tmp_path_factory.mktemp("tube"), verbose=0)
    return res, a, t, L


@pytest.fixture(scope="module")
def long_tube(tmp_path_factory):
    """The 20 m benchmark tube, P2, h = 0.05 m (the one-mode rigid classification and the Gaussian test)."""
    a, t, L = 1.0, 0.01, 20.0
    res = _design("gauss_tube", {"r": [a, a + t], "z": [-L / 2, L / 2]}, 0.05).solve(
        tmp_path_factory.mktemp("long"), verbose=0)
    return res, a, t, L


def test_tube_free_free_modes_and_rigid_mode(tube):
    res, a, t, L = tube
    tr = res.torsion(nmodes=6)
    f = tr["modes"]["f_Hz"]
    exact = [n * CT / (2 * L) for n in range(1, 6)]
    assert f[0] < 1e-5 * exact[0]
    rel = [abs(f[n] / exact[n - 1] - 1) for n in range(1, 6)]
    assert max(rel) < 1e-3, rel      # record: 2.67e-4 at order 2, h = 0.2
    assert max(tr["modes"]["residual"][1:]) < 1e-8
    assert np.allclose(tr["modes"]["modal_mass_kg"], 1.0, rtol=1e-10)
    mass = RHO * math.pi * ((a + t) ** 2 - a ** 2) * L
    assert abs(tr["diagnostics"]["solid_mass_kg"] / mass - 1) < 1e-10
    # the built-in "winding" material has a mechanical block but no density: the coil is skipped, by name
    assert len(tr["warnings"]) == 1 and "coil: no density" in tr["warnings"][0]
    assert tr["structure_regions"] == "body"
    inertia = tr["modes"]["diagnostics"]["inertia"]
    assert inertia["complete"] is True and inertia["eigenvalues_below"] == 6
    assert tr["modes"]["diagnostics"]["unconverged_modes"] == []
    assert tr["modes"]["diagnostics"]["rigid_modes"] == 1


def test_harmonic_equals_complete_modal_sum(tube):
    res, a, t, L = tube
    dofs = res.torsion(dofs=True)
    r = np.array(dofs["dof_r"])
    z = np.array(dofs["dof_z"])
    load = np.where((r >= a - 1e-9) & (r <= a + t + 1e-9) & (np.abs(z) <= 0.3), 1.0 / t, 0.0)
    nfree = dofs["diagnostics"]["free_dofs"]
    full = res.torsion(nmodes=nfree, load=load)
    md = full["modes"]
    # the applied load's integrals: the step at |z| = 0.3 m is smeared over one element by the interpolant
    I2 = ((a + t) ** 3 - a ** 3) / 3
    torque_nominal = 2 * math.pi * I2 / t * 0.6
    assert abs(full["load"]["torque_N_m"] / torque_nominal - 1) < 0.25
    assert full["load"]["force_N"] > 0
    assert len(md["f_Hz"]) == nfree
    lam = (2 * math.pi * np.array(md["f_Hz"])) ** 2
    c = np.array(md["participation"])
    eta = 1e-2
    freqs = [150.0, 777.0, 2500.0]
    hr = res.torsion(load=load, freqs_Hz=freqs, eta=eta)["harmonic"]["response"]
    for f, row in zip(freqs, hr):
        w = 2 * math.pi * f
        q_modal = np.sum(c ** 2 / (lam * (1 - 1j * eta) - w * w))
        q = row["q_re"] + 1j * row["q_im"]
        assert abs(q / q_modal - 1) < 1e-6, (f, q, q_modal)
        Y = row["Y_re"] + 1j * row["Y_im"]
        assert abs(Y + 1j * w * q) < 1e-12 * abs(Y)
        assert row["ok"] == 1
        assert row["Y_re"] > 0       # a passive structure dissipates


def test_time_convention_is_minus_i_omega_t(tube):
    """e^{-i omega t}, as the full-wave engine: far below the first elastic mode the free body
    responds as a rigid rotor, q = -c_0^2/omega^2, so Y = -i omega q = i c_0^2/omega: Im Y > 0 (a mass)."""
    res, a, t, L = tube
    dofs = res.torsion(dofs=True)
    r = np.array(dofs["dof_r"])
    load = np.where((r >= a - 1e-9) & (r <= a + t + 1e-9), 1.0, 0.0)
    out = res.torsion(load=load, freqs_Hz=[20.0], eta=1e-2)
    row = out["harmonic"]["response"][0]
    assert row["Y_im"] > 0 and row["q_re"] < 0
    assert abs(row["Y_re"]) < 1e-6 * row["Y_im"]      # the rigid rotation carries no loss
    assert "e^{-i omega t}" in out["harmonic"]["convention"]


def test_thin_disc_free_edge(tmp_path):
    a, h = 1.0, 0.02
    fun = lambda x: x * special.jv(0, x) - 2.0 * special.jv(1, x)  # noqa: E731
    roots = [optimize.brentq(fun, lo, lo + 1.0) for lo in (4.5, 8.0, 11.5)]
    exact = [x * CT / (2 * math.pi * a) for x in roots]
    res = _design("torsion_disc", {"r": [0.0, a], "z": [-h / 2, h / 2]}, 0.1).solve(tmp_path, verbose=0)
    tr = res.torsion(nmodes=4)
    f = tr["modes"]["f_Hz"]
    assert tr["diagnostics"]["axis_dofs_constrained"] > 0
    rel = [abs(f[i + 1] / exact[i] - 1) for i in range(3)]
    assert max(rel) < 5e-3, rel      # record: 2.32e-3 at order 2, h = 0.1 (four modes)


def test_region_overrides_and_warnings(tube):
    res, a, t, L = tube
    f0 = res.torsion(nmodes=3)["modes"]["f_Hz"]
    tr = res.torsion(nmodes=3, regions={"body": {"mu_Pa": 4 * MU}})
    assert abs(tr["modes"]["f_Hz"][1] / f0[1] - 2.0) < 1e-10
    tr = res.torsion(nmodes=3, regions={"body": {"density": 4 * RHO}})
    assert abs(tr["modes"]["f_Hz"][1] / f0[1] - 0.5) < 1e-10
    with pytest.raises(Exception, match="no free degrees of freedom"):
        res.torsion(nmodes=3, regions={"body": {"exclude": True}})
    # strict request checking
    with pytest.raises(Exception, match="unknown key 'densty'"):
        res.torsion(nmodes=3, regions={"body": {"densty": RHO}})
    with pytest.raises(Exception, match="not a region of the design"):
        res.torsion(nmodes=3, regions={"bdoy": {"density": RHO}})
    with pytest.raises(Exception, match="nu is used only with E_Pa"):
        res.torsion(nmodes=3, regions={"body": {"nu": 0.3}})
    with pytest.raises(Exception, match="freqs_Hz or band needs a load"):
        res.torsion(freqs_Hz=[100.0])
    assert "nothing was computed" in res.torsion()["note"]
    # parameter ranges
    with pytest.raises(Exception, match="eta must be >= 0"):
        res.torsion(load=np.ones(len(res.torsion(dofs=True)["dof_r"])), freqs_Hz=[100.0], eta=-0.01)
    with pytest.raises(Exception, match="nmodes must be >= 0"):
        res.torsion(nmodes=-1)
    n = len(res.torsion(dofs=True)["dof_r"])
    only_load = res.torsion(load=np.ones(n))
    assert "applied load" in only_load["note"] and only_load["load"]["torque_N_m"] > 0


def test_region_without_density_is_skipped_with_a_warning(tmp_path):
    extra = {"nodens": {"kind": "linear", "mechanical": {"E_Pa": E, "nu": NU, "alpha": 0.0}}}
    ring = {"name": "ring", "shape": {"rect": {"r": [2.0, 2.1], "z": [-0.1, 0.1]}}, "material": "nodens",
            "mesh_size": 0.05}
    res = _design("torsion_two", {"r": [1.0, 1.02], "z": [-1.0, 1.0]}, 0.2, extra_materials=extra,
                  extra_regions=[ring]).solve(tmp_path, verbose=0)
    tr = res.torsion(nmodes=2)
    assert tr["structure_regions"] == "body"
    assert any("ring" in w and "no density" in w for w in tr["warnings"])
    tr = res.torsion(nmodes=2, regions={"ring": {"density": RHO}, "body": {"exclude": True}})
    assert tr["structure_regions"] == "ring" and tr["excluded_regions"] == "body"
    assert not any("body" in w for w in tr["warnings"])


def _wall_I2_J(a, t):
    return ((a + t) ** 3 - a ** 3) / 3, 2 * math.pi * ((a + t) ** 4 - a ** 4) / 4


def _exact_gaussian_tube(a, t, L, sig, eta, freqs, nmodes=4000):
    """The exact finite free-free tube driven by the Gaussian ring load (closed-form modes and participations),
    e^{-i omega t}: q = sum_n c_n^2/(lam_n (1 - i eta) - w^2) - c_0^2/w^2, Re Y = Re(-i w q)."""
    I2, J = _wall_I2_J(a, t)
    T = 2 * math.pi * I2 / t
    n = np.arange(1, nmodes + 1)
    kn = n * math.pi / L
    cn2 = T ** 2 * 2 / (RHO * J * L) * (np.cos(kn * L / 2) * sig * math.sqrt(2 * math.pi)
                                         * np.exp(-(kn * sig) ** 2 / 2)) ** 2
    c02 = (T * sig * math.sqrt(2 * math.pi)) ** 2 / (RHO * J * L)
    w = 2 * math.pi * np.asarray(freqs)
    q = (cn2[None, :] / ((CT * kn[None, :]) ** 2 * (1 - 1j * eta) - w[:, None] ** 2)).sum(axis=1) - c02 / w ** 2
    return (-1j * w * q).real


def _damped_waveguide(P2fun, T, J, eta, freqs):
    """Re Y_d of the damped infinite T(0,1) waveguide:
    T^2/(pi J mu) int_0^inf |P(k)|^2 w eta k^2/((k^2 - k0^2)^2 + eta^2 k^4) dk."""
    out = []
    for f in freqs:
        w = 2 * math.pi * f
        k0 = w / CT
        fun = lambda k: P2fun(k) * w * eta * k * k / ((k * k - k0 * k0) ** 2 + eta * eta * k ** 4)  # noqa: E731
        val = integrate.quad(fun, 0.0, 4 * k0, points=[k0 * (1 - 5 * eta), k0, k0 * (1 + 5 * eta)], limit=400)[0]
        val += integrate.quad(fun, 4 * k0, np.inf, limit=200)[0]
        out.append(T * T / (math.pi * J * MU) * val)
    return np.array(out)


def test_gaussian_ring_load_band_mean_matches_the_waveguide(long_tube):
    """A smooth (Gaussian, sigma_z = 0.1 m) ring load uniform through the wall of a 20 m tube, eta = 0.03,
    band 4-8 kHz: modal overlap of the participating (even) modes eta f L/c_T = 0.77-1.5 (1.09 at the centre).
    - FEM band mean = the exact finite-tube band mean (closed-form modes) to 0.5 %;
    - FEM band mean = the damped infinite waveguide Re Y_d to +-1 % (measured: 1.0006). At finite eta the
      band mean tends to Re Y_d, not to the lossless value; Re Y_d differs from it at first order in eta with a sign
      set by the shape of |P(k)|^2: -1.7 % for this Gaussian at eta = 0.03;
    - and the lossless T(0,1) value of the closed-form transform to +-3 % (0.9835)."""
    sig, eta = 0.1, 0.03
    res, a, t, L = long_tube
    dofs = res.torsion(dofs=True)
    r = np.array(dofs["dof_r"])
    z = np.array(dofs["dof_z"])
    g = np.where((r >= a - 1e-9) & (r <= a + t + 1e-9), np.exp(-z ** 2 / (2 * sig * sig)) / t, 0.0)
    freqs = np.linspace(4e3, 8e3, 121)
    out = res.torsion(load=g, freqs_Hz=list(freqs), eta=eta)
    I2, J = _wall_I2_J(a, t)
    torque = 2 * math.pi * I2 / t * sig * math.sqrt(2 * math.pi)
    assert abs(out["load"]["torque_N_m"] / torque - 1) < 1e-8
    Y = np.array([x["Y_re"] for x in out["harmonic"]["response"]])
    k = 2 * math.pi * freqs / CT
    asym = (2 * math.pi * I2 / t) ** 2 * 2 * math.pi * sig * sig * np.exp(-(k * sig) ** 2) / (2 * RHO * CT * J)
    assert eta * math.sqrt(4e3 * 8e3) * L / CT > 1.0
    exact = _exact_gaussian_tube(a, t, L, sig, eta, freqs)
    assert abs(Y.mean() / exact.mean() - 1) < 5e-3, Y.mean() / exact.mean()
    damped = _damped_waveguide(lambda kk: 2 * math.pi * sig * sig * np.exp(-(kk * sig) ** 2),
                               2 * math.pi * I2 / t, J, eta, freqs)
    assert abs(Y.mean() / damped.mean() - 1) < 0.01, Y.mean() / damped.mean()
    ratio = Y.mean() / asym.mean()
    assert abs(ratio - 1) < 0.03, ratio


def test_rigid_mode_classification_one_mode(long_tube, tmp_path):
    """With nmodes = 1 the highest returned mode is the rigid one; its Ritz value rounds to a
    small non-zero number. It must count as rigid, not as unconverged, and raise no warning."""
    res, a, t, L = long_tube
    tr = res.torsion(nmodes=1)
    dg = tr["modes"]["diagnostics"]
    assert dg["rigid_modes"] == 1 and dg["unconverged_modes"] == []
    assert dg["inertia"]["complete"] is True
    assert not any("residual" in w or "inertia" in w for w in tr["warnings"])
    p1 = _design("p1_tube", {"r": [1.0, 1.02], "z": [-2.0, 2.0]}, 0.2, order=1).solve(tmp_path, verbose=0)
    dg = p1.torsion(nmodes=1)["modes"]["diagnostics"]
    assert dg["rigid_modes"] == 1 and dg["unconverged_modes"] == []


def test_rigid_mode_classification_two_bodies(tmp_path):
    """Two disconnected tubes have two rigid modes; with nmodes = 2 both are returned and
    both count as rigid, with no warning; with nmodes = 3 the first elastic mode follows."""
    second = {"name": "tube2", "shape": {"rect": {"r": [1.0, 1.02], "z": [1.0, 3.0]}}, "material": "steel",
              "mesh_size": 0.1}
    res = _design("two_tubes", {"r": [1.0, 1.02], "z": [-3.0, -1.0]}, 0.1,
                  extra_regions=[second]).solve(tmp_path, verbose=0)
    tr = res.torsion(nmodes=2)
    dg = tr["modes"]["diagnostics"]
    assert dg["rigid_modes"] == 2 and dg["unconverged_modes"] == []
    assert dg["inertia"]["complete"] is True
    assert not any("residual" in w or "inertia" in w for w in tr["warnings"])
    f = res.torsion(nmodes=4)["modes"]["f_Hz"]
    assert abs(f[2] / (CT / 4) - 1) < 1e-3 and abs(f[3] / (CT / 4) - 1) < 1e-3   # both 2 m tubes: c_T/(2 L)


def test_surface_traction_above_the_cutoff_exceeds_the_waveguide(tmp_path):
    """The T(0,1) asymptote's validity range. A short thick tube (cutoff c_T/2t = 39 kHz); a load uniform
    through the wall stays near the T(0,1) value above the cutoff, a traction at the inner surface exceeds it
    several times."""
    a, t, L, sz, delta, eta = 0.5, 0.04, 0.6, 0.005, 0.003, 0.1
    res = _design("short_tube", {"r": [a, a + t], "z": [-L / 2, L / 2]}, 0.002).solve(tmp_path, verbose=0)
    dofs = res.torsion(dofs=True)
    r = np.array(dofs["dof_r"])
    z = np.array(dofs["dof_z"])
    wall = (r >= a - 1e-9) & (r <= a + t + 1e-9)
    freqs = [60e3, 100e3, 120e3]
    k = 2 * math.pi * np.array(freqs) / CT
    P2 = 2 * math.pi * sz * sz * np.exp(-(k * sz) ** 2)
    _, J = _wall_I2_J(a, t)
    xs = np.linspace(a, a + t, 4001)
    ratios = {}
    for name, w in (("uniform", lambda x: np.ones_like(x) / t),
                    ("surface", lambda x: np.exp(-(x - a) / delta) / (delta * (1 - math.exp(-t / delta))))):
        g = np.where(wall, w(r) * np.exp(-z ** 2 / (2 * sz * sz)), 0.0)
        tq = np.trapezoid(w(xs) * xs * xs, xs)
        asym = (2 * math.pi * tq) ** 2 * P2 / (2 * RHO * CT * J)
        hr = res.torsion(load=g, freqs_Hz=freqs, eta=eta)["harmonic"]["response"]
        ratios[name] = np.array([x["Y_re"] for x in hr]) / asym
    assert np.all(np.abs(ratios["uniform"] - 1) < 0.15), ratios["uniform"]   # measured: 0.89, 0.94, 0.95
    assert np.all(ratios["surface"] > 3.0), ratios["surface"]                # measured: 4.2, 8.5, 17.2
