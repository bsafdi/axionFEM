"""Parser guards for the genuine FEM current-port/source extension (no solves)."""
import json
from pathlib import Path

import pytest
import axionfem as af


def _doc():
    return json.loads((Path(__file__).resolve().parents[1] / "benchmarks/designs/stage1_twoport.json").read_text())


def test_multiport_source_fixture_validates():
    af.Design(_doc()).validate()


@pytest.mark.parametrize("change", [
    {"region": "gap0"}, {"name": "pickup"}, {"toward": [0, 0, 0]}, {"region": "loop"},
])
def test_reject_duplicate_pec_and_unoriented_ports(change):
    d = _doc()
    d["fullwave"]["ports"][0].update(change)
    with pytest.raises(Exception):
        af.Design(d).validate()


def test_reject_uniform_source_with_radiation_boundary():
    d = _doc()
    d["fullwave"]["boundary"] = "radiation"
    for key in ("source_check", "validation_shunt_R_ohm"):
        d["fullwave"].pop(key)
    with pytest.raises(Exception, match="uniform"):
        af.Design(d).validate()


def test_closed_segment_source_with_radiation_boundary_validates():
    d = _doc()
    d["fullwave"]["boundary"] = "radiation"
    for key in ("source_check", "validation_shunt_R_ohm"):
        d["fullwave"].pop(key)
    d["fullwave"]["source"] = {"kind": "segments", "segments_m_A": [
        [0.1, 0, 0, 0.1, 0.1, 0, 1], [0.1, 0.1, 0, 0, 0.1, 0, 1], [0, 0.1, 0, 0.1, 0, 0, 1]]}
    af.Design(d).validate()


def test_reject_open_filament_with_radiation_boundary():
    d = _doc()
    d["fullwave"]["boundary"] = "radiation"
    for key in ("source_check", "validation_shunt_R_ohm"):
        d["fullwave"].pop(key)
    d["fullwave"]["source"] = {"kind": "segments", "segments_m_A": [
        [0.1, 0, 0, 0.1, 0.1, 0, 1], [0.1, 0.1, 0, 0, 0.1, 0, 1]]}
    with pytest.raises(Exception, match="closed"):
        af.Design(d).validate()


@pytest.mark.parametrize("key,value", [("source_check", True), ("validation_shunt_R_ohm", [10, 17])])
def test_reject_pec_only_source_validation_with_radiation(key, value):
    d = _doc()
    d["fullwave"]["boundary"] = "radiation"
    for k in ("source_check", "validation_shunt_R_ohm"):
        d["fullwave"].pop(k)
    d["fullwave"]["source"] = {"kind": "segments", "segments_m_A": [
        [0.1, 0, 0, 0.1, 0.1, 0, 1], [0.1, 0.1, 0, 0, 0.1, 0, 1], [0, 0.1, 0, 0.1, 0, 0, 1]]}
    d["fullwave"][key] = value
    with pytest.raises(Exception, match="PEC"):
        af.Design(d).validate()


@pytest.mark.parametrize("loads", [[1], [1, 0], [1, -1]])
def test_reject_invalid_direct_closure_loads(loads):
    d = _doc()
    d["fullwave"]["validation_shunt_R_ohm"] = loads
    with pytest.raises(Exception):
        af.Design(d).validate()


def test_legacy_oneport_stays_valid():
    d = _doc()
    d["fullwave"] = {"frequencies": [1e6], "boundary": "pec"}
    af.Design(d).validate()


def test_reject_direct_load_check_without_sweep():
    d = _doc()
    d["fullwave"].pop("frequencies")
    d["fullwave"]["modes"] = 1
    d["fullwave"]["source_check"] = False
    with pytest.raises(Exception, match="sweep"):
        af.Design(d).validate()


@pytest.mark.parametrize("order", [-1, 2.5, 31])
def test_reject_invalid_source_quadrature_order(order):
    d = _doc()
    d["fullwave"]["source_quadrature_order"] = order
    with pytest.raises(Exception, match="source_quadrature_order"):
        af.Design(d).validate()


def test_capacitor_esl_fields_and_probes_validate():
    d = _doc()
    fw = d["fullwave"]
    fw.pop("validation_shunt_R_ohm")
    fw.update(validation_loads=[{"C_F": 1e-8}, {"R_ohm": .1, "C_F": 2e-8, "L_H": 20e-9}],
              field_export_directory="/tmp/axf_energy_schema_only",
              field_currents={"re": [1, 0], "im": [0, .5]},
              field_probes=[[0, 0, 0], [.1, .1, .1]], rf_only=True)
    af.Design(d).validate()


@pytest.mark.parametrize("loads", [[{}], [{"C_F": -1}, {}], [{"L_H": -1}, {}],
                                   [{"R_ohm": -1}, {}], [{"unknown": 1}, {}]])
def test_invalid_reactive_validation_loads_are_rejected(loads):
    d = _doc()
    d["fullwave"].pop("validation_shunt_R_ohm")
    d["fullwave"]["validation_loads"] = loads
    with pytest.raises(Exception):
        af.Design(d).validate()


@pytest.mark.parametrize("currents", [{"re": [1], "im": [0]}, {"re": [1, 0]},
                                      {"re": [1, 0], "im": [0, 0, 0]},
                                      {"re": [1, "bad"], "im": [0, 0]}])
def test_invalid_field_current_vectors_are_rejected(currents):
    d = _doc()
    d["fullwave"]["field_currents"] = currents
    with pytest.raises(Exception):
        af.Design(d).validate()


def test_reactive_validation_without_frequency_sweep_is_rejected():
    d = _doc()
    fw = d["fullwave"]
    fw.pop("validation_shunt_R_ohm")
    fw.pop("frequencies")
    fw.update(modes=1, source_check=False, validation_loads=[{}, {"C_F": 1e-8}])
    with pytest.raises(Exception):
        af.Design(d).validate()


def test_source_overlap_controls_validate_without_extra_ports():
    d = _doc()
    d["fullwave"]["source_controls"] = [
        {"name": "fe_refinement", "source": {"kind": "uniform", "B_T": [1, .2, .3]},
         "quadrature_order": 6},
        {"name": "source_refinement", "source": {"kind": "uniform", "B_T": [1.1, .2, .3]}}]
    af.Design(d).validate()


@pytest.mark.parametrize("controls", [
    [{"name": "a", "source": {"kind": "uniform", "B_T": [1, 0, 0]}}]*2,
    [{"name": "", "source": {"kind": "uniform", "B_T": [1, 0, 0]}}],
    [{"name": "a", "source": {"kind": "uniform", "B_T": [1, 0, 0]}, "quadrature_order": 2.5}],
    [{"name": "a", "source": {"kind": "uniform", "B_T": [1, 0, 0]}, "quadrature_order": 31}],
    [{"name": "a", "source": {"kind": "bad"}}],
])
def test_invalid_source_overlap_controls_are_rejected(controls):
    d = _doc()
    d["fullwave"]["source_controls"] = controls
    with pytest.raises(Exception):
        af.Design(d).validate()
