"""The hand-maintained JSON Schema (axf schema) must agree with the C++ parser (axf validate)."""
import json
import math
from pathlib import Path

import jsonschema
import pytest

import axionfem as af
from axionfem import _core
from axionfem.convert import to_3d

SCHEMA = json.loads(_core.schema())


def _circuit_design():
    return {"schema": "axionfem.design/1", "name": "circ",
            "regions": [{"name": "sol", "shape": {"rect": {"r": [1.0, 1.05], "z": [-1, 1]}}, "material": "winding"},
                        {"name": "pc", "shape": {"rect": {"r": [0.5, 0.52], "z": [-0.05, 0.05]}}, "material": "winding"}],
            "windings": [{"region": "sol", "type": "azimuthal", "ampere_turns": 1e5},
                         {"region": "pc", "type": "azimuthal", "turns": 100, "circuit": "loop"}],
            "circuits": [{"name": "loop", "type": "persistent", "flux_linkage": 0}],
            "background": {"type": "uniform", "Bz": 0.01}}


def _hts_design():
    d = af.template("core").doc
    d["name"] = "core_hts"
    d["conductors"] = {"tape": {"law": "rebco_kim", "T_op": 20.0, "fill_factor": 0.5, "n_value": 25,
                                "tape_normal": [1, 0]},
                       "tab": {"law": "table", "B_T": [0, 1, 2], "T_K": [4.2, 20.0],
                               "Jc": [4e9, 2e9, 2e9, 1e9, 0, 0]}}
    d["windings"][0]["conductor"] = "nbti"
    d["cost"] = {"load_line_cap": 0.9, "sc_wall_cap": 0.086}
    d["hts"] = {"regions": [{"region": "cavity", "conductor": "tape", "initial": "zero_field_cooled"}],
                "drive": {"t_ramp": 100.0, "t_hold": 10.0, "ramp_down": True, "cycles": 2},
                "steps_per_segment": 8, "field_of_view": [[0.0, 0.0], [0.1, 0.0]],
                "newton_tol": 1e-4, "max_newton": 60, "e_floor": 1e-6, "profile_points": 21}
    return d


def _valid_docs():
    docs = [af.template(t["name"]).doc for t in af.templates()]
    docs.append(af.template("core", iron_t=0.1, fl=0).doc)
    docs.append(to_3d(af.template("toroid")).doc)
    docs.append(_circuit_design())
    docs.append(_hts_design())
    for f in sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json")):
        docs.append(json.loads(f.read_text()))
    return docs


@pytest.mark.parametrize("doc", _valid_docs(), ids=lambda d: d["name"])
def test_valid_designs_pass_schema_and_parser(doc):
    jsonschema.validate(doc, SCHEMA)
    af.Design(doc).validate()


@pytest.mark.parametrize("mutate,what", [
    (lambda d: d["windings"].append({"region": "sol", "type": "azimuthal"}), "winding without current or circuit"),
    (lambda d: d.update(unknown_key=1), "unknown top-level key"),
    (lambda d: d["regions"][0].update(colour="red"), "unknown region key"),
    (lambda d: d["background"].update(interpolation="spline"), "unknown interpolation"),
    (lambda d: d.update(cost={"cold_envelope": "sphere"}), "unknown cold envelope"),
    (lambda d: d.update(conductors={"c": {"law": "unobtainium"}}), "unknown conductor law"),
    (lambda d: d.update(conductors={"c": {"law": "rebco_kim", "colour": "red"}}), "unknown conductor key"),
    (lambda d: d.update(hts={"regions": [{"region": "sol"}], "drive": {"t_ramp": 1, "oops": 1}}),
     "unknown hts drive key"),
    (lambda d: d.update(hts={"drive": {"t_ramp": 1}}), "hts without regions"),
    (lambda d: d.update(electrostatics={"boundary": "floating"}), "unknown electrostatics boundary"),
    (lambda d: d.update(electrostatics={"modes": 4}), "unknown electrostatics key"),
    (lambda d: d.setdefault("mesh", {}).update(adapt={"cycles": 2, "watch": ["Q"]}), "unknown adapt.watch quantity"),
    (lambda d: d.setdefault("mesh", {}).update(adapt={"cycles": 2, "watch": []}), "empty adapt.watch"),
])
def test_invalid_designs_fail_schema_and_parser(mutate, what):
    doc = _circuit_design()
    mutate(doc)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)
    with pytest.raises(Exception):
        af.Design(doc).validate()


def test_parameter_expressions_use_standard_precedence():
    """Design parameters: unary minus binds looser than '^', which is right-associative."""
    d = af.template("core")
    d.add_param("u", 2.0)
    d.add_param("neg_sq", "-u^2")
    d.add_param("sq_of_neg", "(-u)^2")
    d.add_param("inv_sq", "u^-2")
    d.add_param("tower", "u^3^2")
    d.add_param("gauss", "exp(-u^2/(2*0.5^2))")
    d.add_param("chained", "1 - neg_sq")
    p = d.resolved_params()
    assert p["neg_sq"] == -4.0
    assert p["sq_of_neg"] == 4.0
    assert p["inv_sq"] == 0.25
    assert p["tower"] == 512.0
    assert p["gauss"] == pytest.approx(math.exp(-8.0), rel=1e-14)
    assert p["chained"] == 5.0
