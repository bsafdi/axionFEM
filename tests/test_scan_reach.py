"""Independent analytic checks of the full-wave reach (axionfem.scan) and its guards.

Each case has an exact answer: the explicit band must stay inside the sweep and below the pole, bulk loss is kept in
the MQS comparison, the tuned-pole (Foster) slope and loaded Q match a tuned series circuit, losses add with their own
noise temperatures, and the noise-bandwidth and coupling-variation guards refuse inputs outside their validity.
"""
import math
import numpy as np
import pytest
from axionfem import scan, circuit


def table():
    return {'coupling': {'Psi_SI_H_T_m2':1e-5,'L_p_H':1e-6},
            'fullwave':{'sweep':[{'f_Hz':f,'Psi_re':1e-5,'Psi_im':0.,'L_H':L,
                                   'Z_re_ohm':.1,'R_rad_ohm':.001,'R_bulk_ohm':.099}
                                  for f,L in [(1e6,1e-6),(2e6,1e-6),(3e6,-1e-6)]]}}


@pytest.mark.parametrize('band', [{'f_max':1e7}, {'f_min':1e5}, {'f_min':1.5e6,'f_max':3e6}])
def test_explicit_band_cannot_cross_sweep_or_pole(band):
    with pytest.raises(ValueError, match='outside.*sweep'):
        scan.reach_fullwave(table(), **band)


def test_bulk_loss_is_retained_in_mqs_comparison():
    t = circuit.fullwave_tables(table())
    assert t['R_rad_ohm'] == pytest.approx(.001)
    assert t['R_loss_ohm'] == pytest.approx(.099)
    assert t['R_ohm'] == pytest.approx(.1)
    r = scan.reach_fullwave(table(), {'Q':1e6,'T':0.,'eta_A':1.}, n=5)
    assert r['mqs']['Q'][0] == pytest.approx(2*math.pi/(.099+2*math.pi/1e6))
    bad = table(); bad['fullwave']['sweep'][0]['R_bulk_ohm'] = .05
    with pytest.raises(ValueError, match='inconsistent'):
        circuit.fullwave_tables(bad)


def test_foster_tuned_pole_slope():
    f = np.geomspace(1e5, 8e5, 200)
    L = lambda f: 1e-6/(1-(f/1e6)**2)
    exact = 1e-6/(1-(f/1e6)**2)**2
    got = scan._pole_inductance(f, L, f)
    assert got == pytest.approx(exact, rel=2e-7)
    assert scan._pole_inductance(f, L(f), f) == pytest.approx(exact, rel=1e-3)
    r = scan.reach_dispersive(f, 1e-5, L, R_ohm=1e-6, n=200)
    assert r['L_H'] == pytest.approx(exact, rel=2e-7)
    assert r['Q'] == pytest.approx(2*math.pi*f*exact/1e-6, rel=2e-7)


def test_sparse_resonance_and_broad_pole_are_refused():
    with pytest.raises(ValueError, match='four increasing'):
        scan._pole_inductance(np.array([1.,2.]), [1.,2.], np.array([1.,2.]))
    with pytest.raises(ValueError, match='bandwidth'):
        scan.reach_dispersive([1e5,8e5], 1e-5, lambda f:1e-6/(1-(f/1e6)**2), Q=10)


def test_background_fullwave_3d_validates_and_a_second_static_field_is_rejected():
    # 3D sweeps couple an imposed background (docs/validation/fullwave3d_background.md); a prescribed
    # fullwave.source next to it, or rf_only (no static solve, so no reaction field), is refused before meshing
    import axionfem as af
    from axionfem.convert import to_3d
    d = to_3d(af.template('solenoid'), pickup='slit')
    d.doc['background'] = {'type':'uniform','Bz':1.}
    d.doc['fullwave'] = {'frequencies':[1e6]}
    d.validate()
    d.doc['fullwave'] = {'frequencies':[1e6], 'boundary':'pec', 'source':{'kind':'uniform','B_T':[0,0,1]}}
    with pytest.raises(af.AxionFEMError, match='background'):
        d.validate()
    d.doc['fullwave'] = {'frequencies':[1e6], 'boundary':'pec', 'rf_only':True}
    with pytest.raises(af.AxionFEMError, match='background'):
        d.validate()


def test_foster_slope_matches_tuned_circuit():
    L0, fs, resistance = 1e-6, 1e6, .001
    def impedance(w):
        return complex(resistance, -w*L0/(1-(w/(2*math.pi*fs))**2))
    for f in [2e5, 5e5, 8e5, 9e5]:
        ro = circuit.series_readout(Z_p=impedance, Psi_SI=1e-5, f_Hz=f, T_K=.02, eta_A=1)
        ro.tune(f)
        eq = ro.equivalent(2*math.pi*f)
        le = scan._pole_inductance(np.array([f]), lambda x:L0/(1-(x/fs)**2), np.array([f]))[0]
        assert eq['L_eff_H'] == pytest.approx(le, rel=1e-6)
        assert eq['Q_loaded'] == pytest.approx(2*math.pi*f*le/resistance, rel=1e-6)


def foster_table(f, resistance=1e-6):
    return {'coupling':{'Psi_SI_H_T_m2':1e-5,'L_p_H':1e-6},
            'fullwave':{'sweep':[{'f_Hz':float(x),'L_H':1e-6/(1-(x/1e6)**2),'Psi_re':1e-5,'Psi_im':0.,
                                  'Z_re_ohm':resistance,'R_bulk_ohm':resistance,'R_rad_ohm':0.}
                                 for x in f]}}


def test_noise_bandwidth_guard_rejects_broad_but_accepts_narrow_foster():
    f = np.geomspace(1e5,8e5,401)
    with pytest.raises(ValueError,match='noise bandwidth'):
        scan.reach_fullwave(foster_table(f,1e-3),{'T':.02,'eta_A':1.},compare_mqs=False)
    r=scan.reach_fullwave(foster_table(f,1e-6),{'T':.02,'eta_A':1.},n=401)
    assert r['band']['noise_bandwidth_over_f_max'] < .1
    assert r['L_eff_H'] == pytest.approx(1e-6/(1-(f/1e6)**2)**2, rel=1e-4)
    assert r['L_reactive_H'] == pytest.approx(1e-6/(1-(f/1e6)**2),rel=1e-12)
    assert r['Q'] == pytest.approx(2*math.pi*f*r['L_eff_H']/1e-6,rel=1e-12)


def test_independent_losses_add_and_radiation_has_its_own_temperature():
    f=np.array([4.9e5,5e5,5.1e5])
    data=foster_table(f)
    for row in data['fullwave']['sweep']:
        row.update(L_H=1e-6,R_bulk_ohm=0.,R_rad_ohm=2*math.pi*row['f_Hz']*1e-6/2000,
                   Z_re_ohm=2*math.pi*row['f_Hz']*1e-6/2000)
    r=scan.reach_fullwave(data,{'Q':2000,'T':0.,'eta_A':1.},n=3)
    assert r['Q'] == pytest.approx(1000.)  # min(Q_rad,Q_other) incorrectly gave 2000
    for row in data['fullwave']['sweep']:
        row['R_rad_ohm']=row['Z_re_ohm']=2*math.pi*row['f_Hz']*1e-6/1e7
    r=scan.reach_fullwave(data,{'Q':1e7,'T':.01,'eta_A':1.},T_radiation_K=.1,n=3)
    w=2*math.pi*f[1]
    expected=circuit.T_from_occupation(w,(circuit.n_bose(w,.01)+circuit.n_bose(w,.1))/2)
    assert r['T_K'][1] == pytest.approx(expected,rel=1e-10)
    assert not r['band']['radiation_temperature_assumed_equal']
    with pytest.raises(ValueError,match='zero material/intrinsic loss'):
        scan.reach_fullwave(data,{'T':.01,'eta_A':1.},n=3)


def test_tabulated_foster_against_exact_finite_band_circuit():
    f=np.geomspace(7.9e5,8.1e5,61)
    data=foster_table(f)
    r=scan.reach_fullwave(data,{'T':.02,'eta_A':1.},n=61)
    def Z(w):return complex(1e-6,-w*1e-6/(1-(w/(2*math.pi*1e6))**2))
    def factory(f0):return circuit.series_readout(Z_p=Z,Psi_SI=1e-5,f_Hz=f0,T_K=.02,eta_A=1.)
    for i in [0,30,60]:
        ro=factory(f[i]);ro.tune(f[i]);eq=ro.equivalent(2*math.pi*f[i])
        assert r['L_eff_H'][i] == pytest.approx(eq['L_eff_H'],rel=1e-5)
        assert r['Q'][i] == pytest.approx(eq['Q_loaded'],rel=1e-5)
    band=scan.finite_band(factory,f[0],f[-1],n_settings=5,n_targets=5,n_v=48,
                          points_per_decade=48,band_points=100)
    assert band['comparison']['kernel_on_resonance_max_rel_dev'] < 1e-5
    assert band['comparison']['g95_same_schedule_max_rel_dev'] < .01

    # Use the sweep adapter's detector parameters to score the SAME finite-band plan.
    # Comparing each method's independently normalized target reach would be tautological.
    from axionfem import _core
    adapter=scan.reach_fullwave(data,{'T':.02,'eta_A':1.},n=5)
    K=np.empty_like(band['K_per_s'])
    for k,(mass,g) in enumerate(zip(band['targets']['m_eV'],band['targets']['g_target_GeV'])):
        for j,f0 in enumerate(adapter['f_Hz']):
            K[k,j]=_core.kernel_row(float(adapter['Psi_SI'][j]),float(adapter['L_eff_H'][j]),
                float(adapter['Q'][j]),float(adapter['T_K'][j]),float(adapter['eta_A'][j]),
                float(mass),float(g),[2*math.pi*float(f0)],[float(band['settings']['xi'][j])],
                "weak",48,0.,float(band['band']['Q_axion']),2.71,.3)[0]
    exposure=band['schedule']['t_s']
    assert np.max(np.abs(((band['K_per_s']@exposure)/(K@exposure))**.25-1)) < .01


def test_local_variation_only_and_nonpositive_slope_are_refused():
    # Narrow width, but strong coupling variation: independently exercises the second guard.
    with pytest.raises(ValueError,match=r'variation guard failed at .*width/f=.*limit 0.05'):
        scan.reach_dispersive([1e6,2e6],lambda f:1e-5*(f/1e6)**20,1e-6,Q=300,T_K=0.,eta_A=1.)
    with pytest.raises(ValueError,match='nonpositive/nonfinite tuned-pole inductance'):
        scan.reach_dispersive([1e6,2e6],1e-5,1e-6,L_eff_H=-1e-6,Q=1e6)
