"""Finite-pack geometry generation, separate from FEM and self-field accuracy."""
import math

import numpy as np
import pytest

from axionfem.prescribed_source import racetrack_source


def _spec():
    return {"coils": {"N": 1, "list": [dict(phi_rad=0., inner_leg_r=1., outer_leg_r=2.,
            height=1.2, corner_radius=.15, pack_normal=.04, pack_azimuthal=.03,
            ampere_turns=1234.)]}}


def test_racetrack_closed_pack_current_and_area():
    source = racetrack_source(_spec(), quadrature_order=3, arc_segments_per_quarter=128)
    segments = np.asarray(source["segments_m_A"])
    per_loop = 4*129
    assert segments.shape == (9*per_loop, 7)
    loops = segments.reshape(9, per_loop, 7)
    np.testing.assert_array_equal(loops[:, :, 3:6], np.roll(loops[:, :, :3], -1, axis=1))
    assert loops[:, 0, 6].sum() == pytest.approx(1234., rel=2e-15)
    # Magnetic moment m = I/2 int r x dl. Rounded-rectangle area and its
    # parallel-offset integral provide an independent orientation/pack check.
    moment = .5*np.sum(np.cross(segments[:, :3], segments[:, 3:6])*segments[:, 6, None], axis=0)
    width, height, corner, thickness = 1., 1.2, .15, .04
    exact_area_average = width*height-(4-math.pi)*corner**2+math.pi*thickness**2/12
    assert moment[1] == pytest.approx(1234.*exact_area_average, rel=2e-6)
    assert abs(moment[0])+abs(moment[2]) < 1e-11


def test_racetrack_rotation_and_rejections():
    first = racetrack_source(_spec(), quadrature_order=1, arc_segments_per_quarter=4)
    spec = _spec()
    spec["coils"]["list"][0]["phi_rad"] = math.pi/2
    second = racetrack_source(spec, quadrature_order=1, arc_segments_per_quarter=4)
    a, b = np.asarray(first["segments_m_A"]), np.asarray(second["segments_m_A"])
    np.testing.assert_allclose(b[:, 0], -a[:, 1], atol=1e-15)
    np.testing.assert_allclose(b[:, 1], a[:, 0], atol=1e-15)
    for bad in (0, 1.2, True):
        with pytest.raises(ValueError):
            racetrack_source(spec, quadrature_order=bad)
    spec["coils"]["N"] = 2
    with pytest.raises(ValueError, match="coils.N"):
        racetrack_source(spec)
