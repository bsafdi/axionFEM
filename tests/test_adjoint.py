"""Adjoint shape sensitivities and gradient-based optimization.

The exact checks are analytic:
  * sources: every quantity is a known power of the ampere-turns for linear media, so
    dPsi/dNI = Psi/NI, dW/dNI = 2W/NI, dL_p/dNI = 0, dF/dNI = 4F/NI and dcost/dNI = 0
    (the cost is evaluated after scaling the amplitude to the requirement);
  * dilation: all template lengths are proportional to R and H, so Psi, L_p, W and F are
    homogeneous in (R, H) of degree 2, 1, 1 and 6; Euler's identity
    R dJ/dR + H dJ/dH = n J then tests the mesh velocity and the adjoint together.
The adjoint is also compared with finite differences on the same morphed mesh.
"""
from __future__ import annotations

import pytest

import axionfem as af

COARSE = {"order": 2, "size": "max(R, H)/10", "surface_size": "max(R, H)/30"}
# exponents of a uniform dilation (lengths -> lambda * lengths at fixed ampere-turns); the cost at
# the cost-optimal size is scale invariant, hence degree 0
DILATION_DEGREE = {"Psi": 2, "L_p": 1, "W_mag": 1, "F": 6, "A_cold": 2, "device_radius": 1, "cost_optimal": 0}
# exponents of the ampere-turns
SOURCE_DEGREE = {"Psi": 1, "L_p": 0, "W_mag": 2, "F": 4, "cost": 0}


def coarse(name: str, **params) -> af.Design:
    d = af.template(name, **params)
    d.doc["mesh"] = dict(COARSE)
    return d


@pytest.fixture(scope="module")
def toroid(tmp_path_factory):
    d = coarse("toroid")
    return d.solve(workdir=tmp_path_factory.mktemp("toroid"), verbose=0)


@pytest.fixture(scope="module")
def solenoid(tmp_path_factory):
    d = coarse("solenoid")
    return d.solve(workdir=tmp_path_factory.mktemp("solenoid"), verbose=0)


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------- analytic checks
@pytest.mark.parametrize("fixture", ["toroid", "solenoid"])
def test_source_scaling(fixture, request):
    """Linear media: every figure of merit is an exact power of the ampere-turns."""
    res = request.getfixturevalue(fixture)
    objs = list(SOURCE_DEGREE)
    s = res.sensitivities(params=["NI"], objectives=objs)
    NI = s["parameter_values"]["NI"]
    for name, n in SOURCE_DEGREE.items():
        J, dJ = s["values"][name], s["gradient"][name]["NI"]
        if n == 0:  # L_p and the cost at the requirement do not depend on the amplitude
            assert abs(dJ) * NI < 1e-8 * abs(J), f"{name}: dJ/dNI = {dJ}, J = {J}"
        else:
            assert rel(dJ, n * J / NI) < 1e-8, f"{name}: {dJ} vs {n * J / NI}"


@pytest.mark.parametrize("fixture", ["toroid", "solenoid"])
@pytest.mark.parametrize("method", ["adjoint", "fd_morph"])
def test_dilation_identity(fixture, request, method):
    """Euler's identity for a uniform dilation: exact for the discrete problem too."""
    res = request.getfixturevalue(fixture)
    objs = list(DILATION_DEGREE)
    s = res.sensitivities(params=["R", "H"], objectives=objs, method=method)
    R, H = s["parameter_values"]["R"], s["parameter_values"]["H"]
    for name, n in DILATION_DEGREE.items():
        g = s["gradient"][name]
        lhs = R * g["R"] + H * g["H"]
        J = s["values"][name]
        if n == 0:
            assert abs(lhs) < 1e-5 * abs(J), f"{name}: {lhs} should vanish (J = {J})"
        else:
            assert rel(lhs, n * J) < 1e-5, f"{name}: {lhs} vs {n * J}"


# ---------------------------------------------------------------- adjoint vs finite differences
@pytest.mark.parametrize("fixture,extra", [("toroid", "y"), ("solenoid", "x")])
def test_adjoint_vs_fd_morph(fixture, request, extra):
    """The adjoint reproduces finite differences on the same morphed mesh."""
    res = request.getfixturevalue(fixture)
    params = ["R", "H", extra]
    objs = ["Psi", "L_p", "W_mag", "F"]
    a = res.sensitivities(params=params, objectives=objs, method="adjoint")
    f = res.sensitivities(params=params, objectives=objs, method="fd_morph")
    for name in objs:
        J = a["values"][name]
        for p in params:
            da, df = a["gradient"][name][p], f["gradient"][name][p]
            # a logarithmic derivative below 1e-5 is not resolvable: the discrete figures of merit
            # themselves move by ~1e-5 relative between meshes (the solenoid's W does not depend on the
            # copper coax at all, so d W/d x is pure round-off)
            floor = 1e-5 * abs(J) / abs(a["parameter_values"][p])
            if max(abs(da), abs(df)) < floor:
                continue
            assert rel(da, df) < 1e-4, f"{name} d/d{p}: adjoint {da} vs fd_morph {df}"


def test_gradient_sign_and_relative(toroid):
    """The toroid's coupling grows with the magnet size and falls with the hole fraction y."""
    s = toroid.sensitivities(params=["R", "H", "y"], objectives=["Psi", "L_p", "F"])
    assert s["gradient"]["Psi"]["R"] > 0 and s["gradient"]["Psi"]["H"] > 0
    assert s["gradient"]["Psi"]["y"] < 0
    # logarithmic derivatives are consistent with the raw ones
    for name in ("Psi", "L_p", "F"):
        for p in ("R", "H", "y"):
            want = s["gradient"][name][p] * s["parameter_values"][p] / s["values"][name]
            assert rel(s["relative"][name][p], want) < 1e-10


# ---------------------------------------------------------------- API and errors
def test_api(toroid):
    cat = af.objectives()
    assert {"Psi", "L_p", "F", "cost"} <= set(cat)
    s = toroid.sensitivities()  # defaults: all parameters, the standard figures of merit
    assert set(s["params"]) == set(toroid.design.params)
    assert {"Psi", "L_p", "W_mag", "F", "cost"} <= set(s["objectives"])
    assert s["diagnostics"]["checks"]["closed_form_vs_post_processing"] < 1e-9
    assert s["diagnostics"]["adjoint"]["adjoint_solves"] >= 1
    rep = s["diagnostics"]["morph"]["R"]
    assert rep["on_geometry"] > 0 and rep["interior_harmonic"] > 0
    assert rep["max_displacement_m"] > 0
    # the template's element size is an expression in R: say that it is held fixed
    assert any("mesh sizes" in w for w in s["warnings"]), s["warnings"]
    with pytest.raises(af.AxionFEMError):
        toroid.sensitivities(params=["R"], objectives=["not_a_quantity"])
    with pytest.raises(af.AxionFEMError):
        toroid.sensitivities(params=["not_a_parameter"], objectives=["Psi"])
    with pytest.raises(af.AxionFEMError):
        toroid.sensitivities(params=["R"], objectives=["Psi"], method="nonsense")


def test_benchmark_objectives(solenoid):
    s = solenoid.sensitivities(params=["R"], objectives=["cost:optimistic", "cost:conservative_solenoid",
                                                         "W_at_req", "F_over_F_req"])
    for b in ("optimistic", "conservative_solenoid"):
        bench = solenoid.benchmark(b)
        assert rel(s["values"][f"cost:{b}"], bench["cost"]["total_MCHF"]) < 1e-12
    assert s["gradient"]["F_over_F_req"]["R"] > 0


def test_design_gradient(tmp_path):
    d = coarse("solenoid")
    g = d.gradient(params=["x"], objectives=["F"], workdir=tmp_path)
    assert g["ok"] and "F" in g["gradient"]


IRON = {
    "name": "iron_sens",
    "parameters": {"R": 1.0, "NI": 2.0e5},
    "regions": [
        {"name": "coil", "shape": {"frame": {"r": ["0.55*R", "0.95*R"], "z": [-0.4, 0.4], "thickness": 0.05}},
         "material": "winding"},
        {"name": "iron", "shape": {"rect": {"r": ["0.62*R", "0.88*R"], "z": [-0.3, 0.3]}}, "material": "iron"},
    ],
    "windings": [{"name": "w", "region": "coil", "type": "poloidal", "ampere_turns": "NI"}],
    "mesh": {"order": 2, "size": 0.25, "surface_size": 0.06},
}


def test_nonlinear_iron_is_refused(tmp_path):
    """Nonlinear iron has no adjoint yet; the error says what to do instead."""
    res = af.Design(IRON).solve(workdir=tmp_path, verbose=0)
    assert res.data["linear"] is False
    with pytest.raises(af.AxionFEMError, match="fd_remesh"):
        res.sensitivities(params=["R"], objectives=["W_mag"])
    s = res.sensitivities(params=["NI"], objectives=["W_mag"], method="fd_remesh", step=1e-3)
    assert s["gradient"]["W_mag"]["NI"] > 0  # more current, more energy (less than quadratically)


# ---------------------------------------------------------------- masters, ties, circuits, DtN
def circuit_design() -> af.Design:
    """A toroid (slit superconducting sheath as the pickup) plus a passive cancellation loop whose
    current follows from flux conservation: the adjoint has to carry the fluxoid ties of the sheath,
    the circuit constraint in the pickup problem (a saddle-point tangent) and the exterior DtN map.
    The loop is azimuthal, so it reacts only to the pickup mode; the static field stays toroidal."""
    d = coarse("toroid")
    d.doc["regions"].append({"name": "cancel", "shape": {"rect": {"r": ["1.3*R", "1.35*R"],
                                                                  "z": ["-0.05*R", "0.05*R"]}},
                             "material": "winding"})
    d.doc["windings"].append({"name": "cancel", "region": "cancel", "type": "azimuthal", "turns": 50,
                              "circuit": "loop"})
    d.doc["circuits"] = [{"name": "loop", "type": "persistent"}]
    return d


@pytest.fixture(scope="module")
def circuit(tmp_path_factory):
    return circuit_design().solve(workdir=tmp_path_factory.mktemp("circuit"), verbose=0)


def test_circuit_and_tie_sensitivities(circuit):
    res = circuit
    circ = res.data["pickup"]["solver"]["circuits"][0]
    assert abs(circ["current_A"]) > 1e-9  # the passive loop carries an induced current
    assert res.Psi != 0 and res.L_p > 0
    objs = ["Psi", "L_p", "W_mag", "F"]
    a = res.sensitivities(params=["R", "H", "NI"], objectives=objs, method="adjoint")
    f = res.sensitivities(params=["R", "H", "NI"], objectives=objs, method="fd_morph")
    R, H, NI = (a["parameter_values"][k] for k in ("R", "H", "NI"))
    for name, n in (("Psi", 2), ("L_p", 1), ("W_mag", 1), ("F", 6)):
        lhs = R * a["gradient"][name]["R"] + H * a["gradient"][name]["H"]
        assert rel(lhs, n * a["values"][name]) < 1e-5, f"dilation {name}"
        for p in ("R", "H"):
            assert rel(a["gradient"][name][p], f["gradient"][name][p]) < 1e-4, f"{name} d/d{p}"
    for name, n in (("Psi", 1), ("L_p", 0), ("W_mag", 2), ("F", 4)):
        J, dJ = a["values"][name], a["gradient"][name]["NI"]
        if n == 0:
            assert abs(dJ) * NI < 1e-8 * abs(J), name
        else:
            assert rel(dJ, n * J / NI) < 1e-8, name


# ---------------------------------------------------------------- curved boundaries (arcs)
# A superconducting ring of circular cross-section: a `disk` shape is one full circle, so its
# boundary exercises the arc primitive of the mesh velocity.  `s` scales every length (L_p and
# W_mag are then homogeneous of degree 1), while `a` moves the circle alone.
RING = {
    "name": "disk_ring",
    "parameters": {"s": 1.0, "a": 0.12, "NI": 1.0e5},
    "regions": [
        {"name": "coil", "shape": {"rect": {"r": ["1.0*s", "1.05*s"], "z": ["-0.5*s", "0.5*s"]}},
         "material": "winding"},
        {"name": "ring", "shape": {"disk": {"center": ["0.5*s", 0], "radius": "a*s"}}, "material": "nb"},
    ],
    "windings": [{"name": "c", "region": "coil", "type": "azimuthal", "ampere_turns": "NI"}],
    "superconductors": [{"body": "ring", "azimuthal": {"net_current": 0}, "azimuthal_slit": True}],
    "pickup": {"type": "azimuthal", "conductor": "ring"},
    "mesh": {"order": 2, "size": "0.1*s", "surface_size": "0.03*s"},
}


def test_arc_primitive(tmp_path):
    res = af.Design(RING).solve(workdir=tmp_path, verbose=0)
    objs = ["L_p", "W_mag"]
    a = res.sensitivities(params=["s", "a"], objectives=objs)
    f = res.sensitivities(params=["a"], objectives=objs, method="fd_morph")
    r = res.sensitivities(params=["a"], objectives=objs, method="fd_remesh", step=1e-2)
    assert a["diagnostics"]["morph"]["a"]["on_geometry"] > 0
    for name in objs:
        # dilation: exact
        assert rel(a["parameter_values"]["s"] * a["gradient"][name]["s"], a["values"][name]) < 1e-5, name
        # the ring radius alone moves the curved boundary only
        assert rel(a["gradient"][name]["a"], f["gradient"][name]["a"]) < 1e-4, name
        assert rel(a["gradient"][name]["a"], r["gradient"][name]["a"]) < 5e-2, name


# ---------------------------------------------------------------- off-origin domain centre
def _shifted_toroid(z0: float) -> af.Design:
    """The toroid template moved along z, with the domain (and the DtN origin) moved with it."""
    d = coarse("toroid", R=1.0, H=1.0).doc
    d["parameters"]["z0"] = z0
    for reg in d["regions"]:
        f = reg["shape"]["frame"]
        f["z"] = [f"({f['z'][0]})+z0", f"({f['z'][1]})+z0"]
    d["mesh"]["domain_center_z"] = "z0"
    return af.Design(d)


def test_off_origin_domain_centre(tmp_path):
    """`mesh.domain_center_z` moves the outer semicircle, so the mesh velocity must scale it about
    (0, z0), not about the origin.  The gradients are then the same as for the centred device."""
    objs = ["Psi", "L_p", "W_mag"]
    here = _shifted_toroid(0.0).solve(workdir=tmp_path / "z0", verbose=0)
    far = _shifted_toroid(10.0).solve(workdir=tmp_path / "z10", verbose=0)
    a0 = here.sensitivities(params=["R"], objectives=objs)
    a1 = far.sensitivities(params=["R"], objectives=objs)
    f1 = far.sensitivities(params=["R"], objectives=objs, method="fd_morph")
    assert a1["diagnostics"]["morph"]["R"]["on_outer_boundary"] > 0
    for name in objs:
        assert rel(a1["gradient"][name]["R"], a0["gradient"][name]["R"]) < 1e-3, name
        assert rel(a1["gradient"][name]["R"], f1["gradient"][name]["R"]) < 1e-4, name


# ---------------------------------------------------------------- optimization
def test_optimize_one_parameter(tmp_path):
    """Maximise the geometric factor of the toroid over the hole fraction: the long-toroid limit's
    optimum of g_tor = y^2 ln(1/y) is y = exp(-1/2) = 0.6065, and a toroid with H = R sits near it."""
    d = coarse("toroid", y=0.45)
    out = af.optimize(d, objective="g_factor", sense="max", params={"y": (0.3, 0.85)}, max_iter=10,
                      workdir=tmp_path, verbose=0)
    assert out["ok"]
    assert out["value"] >= out["start_value"]
    assert 0.50 < out["parameters"]["y"] < 0.70, out["parameters"]


def test_optimize_constrained(tmp_path):
    """Minimum cost at a fixed stored energy: SLSQP with one inequality constraint."""
    d = coarse("solenoid")
    res = d.solve(workdir=tmp_path / "start", verbose=0)
    W0 = res.benchmark("optimistic")["W_mag_at_req_J"]
    out = af.optimize(d, objective="cost", params={"x": (0.3, 0.8), "H": (1.0, 4.0)},
                      constraints=[{"quantity": "W_at_req", "type": "<=", "value": W0}], max_iter=10,
                      workdir=tmp_path, verbose=0)
    assert out["ok"]
    assert out["values"]["W_at_req"] <= W0 * 1.02
    assert out["value"] <= out["start_value"]


@pytest.mark.parametrize('step', [0., -1., float('nan'), float('inf')])
def test_invalid_step_is_rejected(toroid, step):
    with pytest.raises(af.AxionFEMError, match='finite and positive'):
        toroid.sensitivities(params=['NI'], objectives=['Psi'], step=step)


def test_peak_response_and_required_peak_scaling(solenoid):
    names = ['B_peak', 'sc_surface_peak', 'B_peak_at_req', 'sc_surface_at_req']
    a = solenoid.sensitivities(params=['NI', 'R'], objectives=names, step=1e-3)
    f = solenoid.sensitivities(params=['NI', 'R'], objectives=names, method='fd_morph', step=1e-3)
    ni = a['parameter_values']['NI']
    for name in names[:2]:
        assert a['gradient'][name]['NI'] == pytest.approx(a['values'][name]/ni, rel=1e-6)
        assert a['gradient'][name]['R'] == pytest.approx(f['gradient'][name]['R'], rel=1e-6, abs=1e-12)
    for name in names[2:]:
        assert abs(a['gradient'][name]['NI']*ni) < 2e-5*max(a['values'][name], 1e-15)
        assert a['gradient'][name]['R'] == pytest.approx(f['gradient'][name]['R'], rel=2e-4, abs=1e-12)


@pytest.mark.parametrize('method', ['adjoint', 'fd_morph', 'fd_remesh'])
def test_explicit_benchmark_dependence(tmp_path, method):
    d = coarse('solenoid')
    d.doc['parameters']['Qtest'] = 1e6
    d.doc['benchmarks'] = [{'name':'probe', 'Q':'Qtest', 'T':.01, 'eta_A':.1}]
    r = d.solve(tmp_path/method, verbose=0)
    s = r.sensitivities(params=['Qtest'], objectives=['F_over_F_req', 'W_at_req', 'cost'], method=method, step=1e-3)
    assert s['relative']['F_over_F_req']['Qtest'] == pytest.approx(1., rel=1e-5)
    assert s['relative']['W_at_req']['Qtest'] == pytest.approx(-.5, rel=1e-5)
    assert s['gradient']['cost']['Qtest'] < 0


def test_translating_domain_has_zero_gradient(tmp_path):
    r = _shifted_toroid(2.).solve(tmp_path/'translated', verbose=0)
    for method in ['adjoint', 'fd_morph']:
        s = r.sensitivities(params=['z0'], objectives=['Psi','L_p'], method=method, step=1e-3)
        for name in ['Psi','L_p']:
            assert abs(s['gradient'][name]['z0']) < 1e-7*abs(s['values'][name])


def test_zero_primitive_retains_identity_derivative(tmp_path):
    d = coarse('toroid', NI=0.)
    d.doc['regions'].append({'name':'aux','shape':{'rect':{'r':[.1,.2],'z':[-.1,.1]}},'material':'winding'})
    d.doc['windings'].append({'name':'aux','region':'aux','type':'azimuthal','ampere_turns':100.})
    r = d.solve(tmp_path/'zero', verbose=0)
    d.set(NI=1.)
    unit = d.solve(tmp_path/'unit', verbose=0).Psi
    assert unit != 0
    for method in ['adjoint','fd_morph','fd_remesh']:
        s = r.sensitivities(params=['NI'], objectives=['Psi'], method=method, steps={'NI':1.})
        assert s['gradient']['Psi']['NI'] == pytest.approx(unit, rel=1e-6)


@pytest.mark.parametrize('method', ['adjoint','fd_morph','fd_remesh'])
def test_all_warm_cost_has_no_singular_inactive_area_chain(tmp_path, method):
    d = coarse('solenoid')
    d.doc.setdefault('materials', {})['winding'] = {'kind':'linear', 'warm':True}
    d.doc['materials']['copper'] = {'kind':'conductor', 'warm':True}
    r = d.solve(tmp_path/method, verbose=0)
    s = r.sensitivities(params=['NI'], objectives=['cost','A_cold'], method=method)
    assert s['values']['A_cold'] == 0
    assert s['gradient']['A_cold']['NI'] == 0
    assert abs(s['gradient']['cost']['NI'] * s['parameter_values']['NI']) < 1e-5*s['values']['cost']
