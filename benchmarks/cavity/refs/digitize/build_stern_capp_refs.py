"""Assemble refs/stern_multipost_1603.json, refs/capp_multicell_1710.json, refs/capp_multicell_2205.json.

Run after stern_multipost_1603_digitize.py and capp_multicell_2205_digitize.py (it reads their raw outputs
in this directory).  Pure bookkeeping + the closed-form sector-cell check (capp_sector_analytic.py) + the
Stern figure-of-merit consistency check.  Usage: python3 -I build_stern_capp_refs.py
"""
import os, sys, json
import numpy as np
from scipy import special as sp, optimize

HERE = os.path.dirname(os.path.abspath(__file__)); REFS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from capp_sector_analytic import sector  # noqa: E402

X01 = sp.jn_zeros(0, 1)[0]; C0 = 299792458.0; MU0 = 4e-7 * np.pi


def dump(name, d):
    json.dump(d, open(os.path.join(REFS, name + '.json'), 'w'), indent=1, ensure_ascii=False)
    print('wrote', name + '.json')


# ----------------------------------------------------------------------------------------------- Stern
def stern():
    raw = json.load(open(os.path.join(HERE, 'stern_multipost_1603_digitized.json')))
    figmap = {('fig6', 0): 'Fig. 6a', ('fig6', 1): 'Fig. 6b', ('fig7', 0): 'Fig. 7a', ('fig7', 1): 'Fig. 7b'}
    qlab = {0: ('Q0', 'unloaded Q (300 K copper, d = 25 cm, R = 5 cm)', ''), 1: ('C', 'form factor C010-like', ''),
            2: ('FoM', 'figure of merit F (Eq. 18)', '')}
    curves = []
    for fig in ('fig6', 'fig7'):
        for col in (0, 1):
            for row in range(3):
                p = raw[fig]['panels'][f'{col},{row}']
                cal = p['calibration']
                ux = 1 / cal['px_per_unit_x']; uy = 1 / cal['px_per_unit_y']
                series = []
                for d, pts in sorted(p['design_points'].items()):
                    sd = dict(label=f'design {d}', points=[[q['x'], q['y']] for q in pts],
                              occluded_point_indices=[i for i, q in enumerate(pts) if q['occluded']])
                    if (fig, col, row, d) == ('fig6', 1, 1, 'E'):
                        sd['missing_points'] = [dict(x=0.2515, y=None, note='Fig. 6b C panel shows 6 design-E markers vs 7 in the Q and F '
                                                     'panels; the marker at Lambda ~0.2515 (Q/F panels: native x px 985-987) is not visible in '
                                                     'the C panel (re-inspected at 10x on a 10 px grid: only the D circle, F triangle and G square '
                                                     'are there) - fully hidden under them or not plotted; not recoverable. Expected C ~0.54-0.55 '
                                                     'by interpolation of its neighbours (not a measurement)')]
                    series.append(sd)
                q = qlab[row]
                curves.append(dict(
                    source=f"{figmap[(fig, col)]} ({['top', 'middle', 'bottom'][row]} panel)",
                    description=f'{q[1]} vs Lambda = f*R, simulation points (markers) only; dashed lines in the figure are spline guides',
                    x=dict(label='Lambda = f * R', unit='GHz m'), y=dict(label=q[0], unit=q[2]),
                    series=series,
                    digitization=dict(
                        method='native embedded RGB bitmap (1270x1582 px, no resampling); axes calibrated from detected frame + labelled ticks '
                               '(x: 0.25, 0.30; y: frame top/bottom = labelled limits); marker centres read by eye on 3-9x zoomed crops with a '
                               '10 px grid (digitize/stern_multipost_1603_manual.json) and cross-checked against an automatic ring-coverage marker '
                               'detector: 194 of the 256 points were also found automatically, rms difference 0.9 px per axis, max 3.3 px',
                        page=7 if fig == 'fig6' else 8, clip_bbox_pt='whole embedded image',
                        axis_calibration=dict(x=cal['x'], y=cal['y'], log=False),
                        uncertainty_x=round(2 * ux, 5), uncertainty_y=round(2 * uy, 4 if row else 0),
                        uncertainty_basis='+-2 px (marker-centre reading 1 px rms + calibration 1 px); occluded points (listed) +-3 px; '
                                          'markers are ~12 px wide so overlapping series were separated by colour/shape by eye',
                        figure_png=f'stern_multipost_1603_{fig}.png', overlay_png=f'stern_multipost_1603_{fig}_overlay.png',
                        script='digitize/stern_multipost_1603_digitize.py')))
    # FoM consistency: F = (Lambda/Lambda0) sqrt(C/C0) (Q/Q_ref)^(1/4), Q_ref = (R0/delta)/(1 + R0/d)
    rows = []
    for fig in raw:
        for col in (0, 1):
            P = [raw[fig]['panels'][f'{col},{k}']['design_points'] for k in range(3)]
            for d in P[0]:
                for a in P[0][d]:
                    b = [t for t in P[1].get(d, []) if abs(t['px'][0] - a['px'][0]) < 5]
                    e = [t for t in P[2].get(d, []) if abs(t['px'][0] - a['px'][0]) < 5]
                    if len(b) == 1 and len(e) == 1: rows.append((a['x'], b[0]['y'], a['y'], e[0]['y']))

    L0_exact = X01 * C0 / (2 * np.pi) / 1e9

    def pred(L, C, Q, sig, L0=L0_exact):
        f = L / 0.05 * 1e9; R0 = X01 * C0 / (2 * np.pi * f); dl = np.sqrt(2 / (2 * np.pi * f * MU0 * sig))
        Qr = (R0 / dl) / (1 + R0 / 0.25)
        return (L / L0) * np.sqrt(C / 0.692) * (Q / Qr) ** 0.25
    sig = optimize.brentq(lambda s: np.mean([pred(*r[:3], s) / r[3] - 1 for r in rows]), 1e6, 1e10)
    sig115 = optimize.brentq(lambda s: np.mean([pred(*r[:3], s, 0.115) / r[3] - 1 for r in rows]), 1e6, 1e10)
    res = np.array([pred(*r[:3], sig) / r[3] - 1 for r in rows])
    res58 = np.array([pred(*r[:3], 5.8e7) / r[3] - 1 for r in rows])
    res58_115 = np.array([pred(*r[:3], 5.8e7, 0.115) / r[3] - 1 for r in rows])
    fom_check = dict(
        formula='F = (Lambda/Lambda0) * sqrt(C/C0) * (Q/Q_ref)^(1/4), Lambda0 = x01 c/(2 pi) = %.5f GHz m (printed as 0.115 on p.5), C0 = 0.692, ' % L0_exact +
                'Q_ref = (R0/delta)/(1 + R0/d) for an empty cylinder of radius R0 = x01 c/(2 pi f), height d = 0.25 m, same delta '
                '(reconstruction of Eqs. 17-26; the extracted PDF text of these equations is garbled)',
        n_points=len(rows), sigma_fit_S_per_m=float('%.3g' % sig), sigma_fit_with_printed_Lambda0_0p115_S_per_m=float('%.3g' % sig115),
        rms_rel_residual_at_fit=float('%.2g' % res.std()), max_rel_residual_at_fit=float('%.2g' % np.abs(res).max()),
        mean_rel_residual_sigma_5p8e7=float('%.2g' % res58.mean()),
        mean_rel_residual_sigma_5p8e7_Lambda0_0p115=float('%.2g' % res58_115.mean()),
        conclusion='the digitized F is reproduced from the digitized Q and C to 0.25%% rms, confirming the formula reconstruction and the '
                   'digitization. sigma is NOT constrained by this check: F ~ sigma^(-1/8), so the fitted sigma is degenerate '
                   'with the rounding of Lambda0 - exact Lambda0 gives sigma_fit = %.3g S/m, the printed Lambda0 = 0.115 gives %.3g S/m '
                   '(mean residual at 5.8e7: %+.2f%% vs %+.2f%%). Treat sigma as 5.8e7-6.0e7 S/m; recommended 5.8e7 with Lambda0 = 0.115'
                   % (sig, sig115, 100 * res58.mean(), 100 * res58_115.mean()))

    table = dict(source='Table I p.4', description='Scalable geometry of the best designs (all rotate about the cavity axis)', simulated=False,
                 columns=['design', 'n_stationary', 'n_moving', 'R_MOV/R', 'R_STAT/R', 'POST_DIA/R'], units=['', '', '', '', '', ''],
                 rows=[['A', 2, 2, 0.413, 0.413, 0.350], ['B', 2, 2, 0.425, 0.425, 0.300], ['C', 2, 2, 0.454, 0.454, 0.350],
                       ['D', 3, 3, 0.375, 0.625, 0.250], ['E', 3, 3, 0.375, 0.750, 0.375], ['F', 3, 3, 0.400, 0.675, 0.275],
                       ['G', 3, 3, 0.400, 0.800, 0.300], ['H', 4, 3, 0.450, 0.450, 0.200], ['J', 4, 3, 0.525, 0.525, 0.175],
                       ['K', 4, 4, 0.550, 0.550, 0.200]])
    table_v = dict(source='Table I p.4 (regulating vanes)', simulated=False, description='Vane designs; vane angle in units of pi rad',
                   columns=['design', 'n_stationary', 'n_moving', 'R_INNER/R', 'R_OUTER/R', 'vane_angle_over_pi'], units=['', '', '', '', '', ''],
                   rows=[['L', 2, 2, 0.250, 0.800, 0.150], ['M', 2, 2, 0.300, 0.700, 0.150], ['N', 3, 3, 0.300, 0.600, 0.050]])
    d = dict(
        name='stern_multipost_1603', title='Stern et al. multi-post and regulating-vane TM010-like cavities (COMSOL trade study)',
        benchmark_rank='6',
        citations=[dict(arxiv='1603.06990', authors_short='I. Stern et al.', journal='Rev. Sci. Instrum. 86, 123305 (2015)',
                        items_used=['Table I', 'Fig. 6a', 'Fig. 6b', 'Fig. 7a', 'Fig. 7b', 'Eqs. 17-26 (FoM normalization)'])],
        software=dict(code='COMSOL Multiphysics', version='4.2', solver_type='3D eigenmode', boundary='impedance boundary condition (copper walls, posts, vanes)',
                      mesh_notes='~20k tetrahedra per model (text p.4); authors note convergence difficulties near field transitions may exaggerate bumps'),
        geometry=dict(units='mm', description='Right circular cylinder R = 50 mm, d = 250 mm (aspect d/R = 5); metal posts or vanes run the full '
                      'length with no end gaps. N_stat stationary + N_mov moving posts; moving posts rotate rigidly about the cavity axis on a '
                      'circle of radius R_MOV, stationary posts on radius R_STAT, all posts of diameter POST_DIA (Table I). Vanes: "inner vane '
                      'rounds" at R_INNER, "outer vane rounds" at R_OUTER, "vane angle" between the vane sides (teardrop/wedge shape, inset Fig. 7b).',
                      parameters=dict(
                          R=dict(value=50.0, unit='mm', source='text p.4'), d=dict(value=250.0, unit='mm', source='text p.4'),
                          end_gaps=dict(value=0.0, unit='mm', source='text p.4 ("without any air gaps")'),
                          post_and_vane_dims=dict(value='see reference_values.tables (Table I), ratios to R', unit='', source='Table I p.4'),
                          moving_post_angle_range=dict(value=None, unit='deg', source='not given', note='only endpoint sketches in Fig. 4/Fig. 5; see gaps'),
                          angular_phase_stationary_vs_moving=dict(value=None, unit='deg', source='not given; Figs. 4-5 show equal spacing within each set'),
                          vane_round_radii=dict(value=None, unit='mm', source='not given', note='inferred r_round = R_x * sin(vane_angle/2) if sides are radial'))),
        materials=dict(walls_posts_vanes=dict(material='copper', sigma_S_per_m=None, temperature_K=300,
                                              source='Fig. 6/7 captions "temperature of 300 K"; sigma not stated; the FoM check does not constrain it (5.8e7-6.0e7 S/m '
                                                     'depending on the Lambda0 rounding, see fom_consistency_check); use 5.8e7')),
        conditions=dict(temperature_K=300, B0_T=None, B0_direction='z (cavity axis)', field_map='uniform (implicit, form factor Eq. 3)'),
        conventions=dict(
            form_factor_definition='C = |int B0.E dV|^2 / (B0^2 V int eps |E|^2 dV) (Eq. 3, uniform B0 || z); C0 = 0.692 for the empty TM010 '
                                   'reference. V convention (full cylinder vs vacuum excluding posts) NOT stated - see gaps; report C both ways',
            Q_definition='Q0 = omega U / P_loss (Eq. 9), impedance BC; plotted for R = 5 cm, d = 25 cm, 300 K. Scale Q as R^(1/2) (normal skin) or R^(2/3) (ASE)',
            notes='Lambda = f*R (GHz m), scalable; figure of merit F (Eq. 18) compares to an empty cylinder of equal TM010 frequency and equal height d = 25 cm'),
        reference_values=dict(tables=[table, table_v], curves=curves, fom_consistency_check=fom_check),
        measured_values=dict(note='none: the paper has no measurements'),
        gaps=[
            dict(parameter='moving-post / vane rotation range (start and end angle) and the angular phase between stationary and moving sets',
                 issue='only the end configurations are sketched (Fig. 4, Fig. 5 insets); no angles are given',
                 proposal='parametrize by the relative rotation angle phi of the moving set; sweep phi over its full geometric range (from contact/'
                          'minimum clearance to the symmetric position) at ~1 deg steps, track TM010-like by overlap, then compare along Lambda '
                          '(both Q, C and F are plotted vs Lambda, so the unknown phi mapping drops out); use the digitized Lambda_min/Lambda_max '
                          'of each design to locate the endpoints'),
            dict(parameter='vane shape (round radii, whether the sides are radial)', issue='only R_INNER, R_OUTER and the side angle are given',
                 proposal='assume a teardrop: two circles centred at R_INNER and R_OUTER tangent to two straight sides that meet at the axis with '
                          'opening angle (vane_angle*pi); check against the Fig. 5d-f / Fig. 7b inset images; vane designs are lower priority'),
            dict(parameter='copper conductivity / surface resistance', issue='not stated; the FoM check is degenerate with the Lambda0 rounding, '
                 'so it does not pin sigma', proposal='use sigma = 5.8e7 S/m (300 K); the plausible range 5.8e7-6.0e7 moves Q by '
                 '<= 1.7% (Q ~ sqrt(sigma)) - carry that as a Q systematic'),
            dict(parameter='form-factor volume convention (V in Eq. 3)', issue='Eq. 3 divides by "V" without saying whether V is the full '
                 'cylinder (pi R^2 d) or the vacuum volume excluding the posts/vanes. The posts occupy a large area fraction: '
                 + ', '.join('%s %.1f%%' % (r[0], 100 * (r[1] + r[2]) * (r[5] / 2) ** 2) for r in table['rows']) +
                 ' of pi R^2, so the two conventions differ by that factor (C_vac = C_full / (1 - a))',
                 proposal='axionFEM must report C both ways (C_full with V = pi R^2 d, C_vac with V = vacuum volume) for every Stern design; '
                          'neither is preferred a priori. C0 = 0.692 (empty cylinder) does not discriminate. If one convention matches across designs '
                          'with very different area fractions (D 9% vs E 21%), record it here'),
            dict(parameter='geometry of the "6 Post Design H" in Fig. 6b', issue='Table I has no 6-post design H (see inconsistencies)',
                 proposal='skip this series; it has no geometry')],
        inconsistencies=[
            dict(description='Design labels differ between Table I and Figs. 6-7',
                 evidence='Table I lists post designs A-H, J, K (no I): H = 4 stationary + 3 moving (7 posts). Fig. 6b plots "6 Post Design H" '
                          '(5 six-post series D-H while the table has 4: D-G); Fig. 7a plots "7 Post Design I" and "7 Post Design J" and "8 Post Design K".',
                 proposal='map Table H (7-post) -> Fig. 7a "Design I"; Table J -> Fig. 7a J; Table K -> Fig. 7a K; Fig. 6b "H" has no geometry. '
                          'Do not use I/H/K as a benchmark before checking this (see also the Fig. 7a inset vs Table I K item)'),
            dict(description='Fig. 7a 8-post inset geometry vs Table I design K',
                 evidence='the Fig. 7a inset (p.8, the only 8-post design, i.e. K) shows two rings of 4 posts at ~0.31 R and ~0.65 R with post '
                          'diameter ~0.2 R (re-measured on the embedded bitmap: inner posts ~26 px, outer ~55 px from the centre of an ~85 px '
                          'radius disk, posts ~18 px across), but Table I gives K = R_MOV/R = R_STAT/R = 0.55, POST_DIA/R = 0.20 (one ring). '
                          'The Fig. 6b 6-post inset measures 0.37/0.64/0.24 R = Table I design D (0.375/0.625/0.25), so the insets are drawn '
                          'to scale and the K mismatch is real',
                 proposal='treat the design K geometry as uncertain (Table I and the inset disagree); do not use K (nor I/H) as a benchmark until '
                          'unresolved; if used, run both the Table I and the inset geometry'),
            dict(description='Some Fig. 7a design J points fall below the Q axis limit', evidence='the last J point (Lambda ~0.316) is outside the '
                 'Q panel (Q < 10,000) but present in the C and F panels', proposal='use C/F at that point only; Q for it is not recoverable'),
            dict(description='the earlier literature survey check', evidence='its summary (R = 5 cm, H = 25 cm, COMSOL 4.2, ~20k tets, Table I A = '
                 '2/2/0.413/0.413/0.350, ten post + three vane designs) matches the PDF', proposal='none')],
        solver_class=dict(axisymmetric_2d_possible=False, needs_3d=False,
                          reason='posts/vanes are off-axis, so no axisymmetry; but they run the full height with no gaps, so the TM010-like mode '
                                 'is z-invariant: f and C follow from a 2D scalar Helmholtz problem for Ez on the cross-section (Dirichlet on '
                                 'all metal), and Q adds the analytic end-cap term 1/Q = 1/Q_wall + delta/d. A 3D extruded mesh also works '
                                 '(the paper used 3D). axionFEM currently has no 2D planar scalar solver -> 3D unless one is added'),
        priority='P2', priority_reason='tuning-angle mapping unknown, label inconsistencies (H/I, K inset vs table) and C volume convention '
                                       'unknown (posts 9-21% of the area); curves digitized to ~0.5% of full scale; good test of multi-rod mode '
                                       'tracking (A-G, J, L-N), not of absolute accuracy',
        readme_row=dict(quantities='Q0, C, FoM vs Lambda=f*R: 14 digitized series (13 Table I designs + the geometry-less Fig. 6b "H"), 256 points '
                                   '(241 excluding "H"); Table I geometry ratios',
                        geometry_complete='no', axisym_2d='no (2D cross-section scalar Helmholtz would suffice: z-invariant)', needs_3d='yes for axionFEM today'))
    dump('stern_multipost_1603', d)


# -------------------------------------------------------------------------------------------- CAPP 1710
def capp1710():
    base = sector(0, 0.025, 5.96e7)
    tab2 = {2: (17794, 17819, 17836), 4: (16151, 16151, 16335), 6: (14320, 14310, 14325), 8: (12907, 12851, 12930)}
    tab3 = {0: (1.00, 1.00, 0.00), 2: (1.59, 0.68, 0.84), 4: (2.14, 0.62, 1.36), 6: (2.65, 0.55, 1.96), 8: (3.16, 0.50, 2.59)}
    rows = []
    for N in (0, 2, 4, 6, 8):
        o = sector(N, 0.025, 5.96e7)
        r = dict(N_cells=N, nu=o['nu'], chi=round(o['chi'], 6), f_GHz=round(o['f_GHz'], 5), delta_um=round(o['delta_um'], 5),
                 Q_no_endcaps=round(o['Q_no_endcaps'], 1), Q_over_Q0=round(o['Q_no_endcaps'] / base['Q_no_endcaps'], 4),
                 w_over_w0=round(o['f_GHz'] / base['f_GHz'], 4), Prad_over_Pazi=round(o['Prad_over_Pazi'], 4), C=round(o['C'], 5),
                 paper_table3=dict(w_over_w0=tab3[N][0], Q_over_Q0=tab3[N][1], Prad_over_Pazi=tab3[N][2]))
        if N:
            r['paper_table2'] = dict(analytical=tab2[N][0], numerical=tab2[N][1], numerical_ax=tab2[N][2])
            r['our_Q_minus_paper_analytical_pct'] = round(100 * (o['Q_no_endcaps'] / tab2[N][0] - 1), 3)
            r['paper_COMSOL_minus_our_Q_pct'] = round(100 * (tab2[N][1] / o['Q_no_endcaps'] - 1), 3)
        rows.append(r)
    sig_implied = 5.96e7 * np.mean([(tab2[N][0] / sector(N, 0.025, 5.96e7)['Q_no_endcaps']) ** 2 for N in (2, 4, 6, 8)])
    d = dict(
        name='capp_multicell_1710', title='CAPP multiple-cell ("pizza") cavity concept: ideal sector cells and multi-cell vs multi-cavity',
        benchmark_rank='7',
        citations=[dict(arxiv='1710.06969', authors_short='J. Jeong, S. Youn, S. Ahn, J. E. Kim, Y. K. Semertzidis', journal='Phys. Lett. B 777, 412 (2018)',
                        items_used=['Table 1', 'Table 2', 'Table 3', 'Eqs. 10-15', 'Sec. 5 (double-cell prototype)'])],
        software=dict(code='COMSOL Multiphysics', version=None, solver_type='eigenmode (Numerical) and driven with axion source (Numerical^ax)',
                      boundary='Table 2: copper impedance walls, no end caps', mesh_notes='not given'),
        geometry=dict(units='mm', description='Table 2/3: ideal circular-sector cells, radius Rc = 25 mm, N = 2,4,6,8 cells of angle theta = 2 pi/N, '
                      'zero-thickness PEC-like partitions meeting on the axis, infinite length ("no end caps"). Table 1: 100 mm magnet bore, cavity '
                      'height 200 mm, wall thickness 5 mm, quad-cavity (4 separate cylinders) vs quad-cell vs sext-cell. Prototype (Sec. 5): '
                      'double-cell OFHC, 90 mm ID, 100 mm length, 5 mm wall, 10 mm hollow gap, alumina rods',
                      parameters=dict(
                          Rc_table2=dict(value=25.0, unit='mm', source='Table 2 caption'),
                          length_table2=dict(value='infinite (no end caps)', unit='', source='Table 2 caption'),
                          partition_thickness_table2=dict(value=0.0, unit='mm', source='inferred: analytic PEC sector solution (Eq. 11)'),
                          bore_table1=dict(value=100.0, unit='mm', source='text p.2'), height_table1=dict(value=200.0, unit='mm', source='text p.2'),
                          wall_thickness_table1=dict(value=5.0, unit='mm', source='text p.2'),
                          inner_radius_table1=dict(value=45.0, unit='mm', source='inferred: bore/2 - wall', note='quad-cavity: 4 cylinders of R ~15.7 mm (from f = 7.30 GHz and V = 0.62 L)'),
                          partition_thickness_table1=dict(value=None, unit='mm', source='not given', note='V = 1.08 L (quad) and 1.02 L (sext) vs pi*45^2*200 = 1.272 L imply ~5 mm partitions'),
                          center_gap_table1=dict(value=None, unit='mm', source='not given', note='Sec. 3.4 optimum for quad-cell is d/D = 0.1'),
                          prototype_ID=dict(value=90.0, unit='mm', source='Sec. 5'), prototype_length=dict(value=100.0, unit='mm', source='Sec. 5'),
                          prototype_gap=dict(value=10.0, unit='mm', source='Sec. 5'), prototype_wall=dict(value=5.0, unit='mm', source='Sec. 5'),
                          tuning_rod=dict(value='eps_r = 10 lossless, r/R = 0.1 (simulation); 99.5% Al2O3 in the prototype', unit='', source='Sec. 3.2, Sec. 5'))),
        materials=dict(walls=dict(material='copper', sigma_S_per_m=5.96e7, temperature_K=293, source='Table 2 caption ("nominal copper ... room temperature")')),
        conditions=dict(temperature_K=293, B0_T=None, B0_direction='z', field_map='uniform (Table 1: "homogeneous within the magnet bore")'),
        conventions=dict(form_factor_definition='C = |int E.B0 dV|^2 / (int eps|E|^2 dV int |B0|^2 dV) (Eq. 2)',
                         Q_definition='Q = omega U / P_loss with P_loss = (1/(2 sigma mu^2 delta)) int |n x B|^2 dA (Eqs. 12-14); Table 2 excludes end caps',
                         notes='Table 3 ratios are relative to the empty cylinder (N=0) of the same Rc at its own frequency'),
        reference_values=dict(tables=[
            dict(source='Table 1 p.2', simulated=True, description='multi-cavity vs multi-cell in a 100 mm bore (COMSOL, RT, unloaded Q); P and df/dt relative to quad-cavity',
                 columns=['quantity', 'quad-cavity', 'quad-cell', 'sext-cell'], units=['', '', '', ''],
                 rows=[['V [L]', 0.62, 1.08, 1.02], ['f_TM010 [GHz]', 7.30, 5.89, 7.60], ['Q', 19150, 19100, 16910], ['C', 0.69, 0.65, 0.63],
                       ['P_a->gg (rel.)', 1.00, 1.65, 1.32], ['df/dt (rel.)', 1.00, 2.72, 1.98]]),
            dict(source='Table 2 p.3', simulated=True, description='Q of N-cell cavity, no end caps, Rc = 25 mm, sigma = 5.96e7 S/m; parentheses = % deviation from analytic',
                 columns=['N_cells', 'theta', 'Q_analytical', 'Q_numerical', 'Q_numerical_ax'], units=['', 'rad', '', '', ''],
                 rows=[[2, 'pi', 17794, '17,819 (0.1)', '17,836 (0.2)'], [4, 'pi/2', 16151, '16,151 (0.0)', '16,335 (1.1)'],
                       [6, 'pi/3', 14320, '14,310 (0.1)', '14,325 (0.0)'], [8, 'pi/4', 12907, '12,851 (0.4)', '12,930 (0.2)']]),
            dict(source='Table 3 p.3', simulated=True, description='TM010-like frequency, Q and loss ratio relative to N = 0 (same geometry as Table 2)',
                 columns=['N_cells', 'theta', 'omega/omega0', 'Q/Q0', 'P_loss(rad B_phi)/P_loss(azi B_r)'], units=['', 'rad', '', '', ''],
                 rows=[[0, 'no wall', 1.00, 1.00, 0.00], [2, 'pi', 1.59, 0.68, 0.84], [4, 'pi/2', 2.14, 0.62, 1.36], [6, 'pi/3', 2.65, 0.55, 1.96],
                       [8, 'pi/4', 3.16, 0.50, 2.59]])],
            curves=[],
            analytic_check=dict(
                description='closed-form TM_{nu 1} sector mode, Ez = J_nu(chi r/Rc) sin(nu phi), nu = N/2, computed by digitize/capp_sector_analytic.py '
                            '(quadrature to 1e-13): Q = (2/delta) k^2 int|Ez|^2 dA / oint|dEz/dn|^2 dl, C = |int Ez dA|^2/(A int|Ez|^2 dA)',
                rows=rows,
                findings=['omega/omega0, Q/Q0, Prad/Pazi (0.849 vs printed 0.84 for N=2; others agree to rounding) and C reproduce Table 3 / 2205.01319 Table 2',
                          'our analytic Q with the stated sigma = 5.96e7 S/m is 0.32%% BELOW the paper analytic values for every N; the constant ratio '
                          'implies the paper effectively used sigma = %.4g S/m, which is exactly COMSOL\'s built-in copper (5.998e7 S/m) - the likely '
                          'explanation (analytic column evaluated with the material-library value, not the caption value). Use our closed form (stated sigma) as the axionFEM target; '
                          'the paper COMSOL values agree with it to +0.46/+0.32/+0.25/-0.12 %% (N = 2/4/6/8)' % sig_implied,
                          'C (uniform Bz) is the same for one cell and for all N cells in phase: 0.6440 / 0.6493 / 0.6341 / 0.6132 for N = 2/4/6/8'])),
        measured_values=dict(tables=[dict(source='Sec. 5 text', simulated=False, description='double-cell prototype at RT',
                                          columns=['quantity', 'value'], units=['', ''],
                                          rows=[['f TM010-like (no rods?) [GHz]', 4.2], ['Q0', 13500], ['TM010-like tuning range with rods [GHz]', '3.6-4.0 (Fig. 9)'],
                                                ['TE211-like crossing [GHz]', 3.625]])],
                             note='Fig. 9 measured mode map not digitized (rod positions unknown, measurement only)'),
        gaps=[dict(parameter='Table 1 partition thickness, centre gap, rod presence', issue='not stated',
                   proposal='infer from volumes (5 mm partitions; V(quad) = 1.08 L suggests partitions nearly to the axis, V(sext) = 1.02 L suggests a ~4.5 mm gap); '
                            'sweep t in {0, 5 mm} and Rg in {0, 0.05, 0.1}*Rc and match f and V; treat Table 1 as P2 sanity only'),
              dict(parameter='prototype rod size / pivot', issue='alumina rod diameter and trajectory not given', proposal='do not benchmark Fig. 9')],
        inconsistencies=[
            dict(description='Table 2 analytic Q vs stated sigma', evidence='closed form with sigma = 5.96e7 S/m gives 17737.5/16099.9/14274.8/12866.1, '
                 'uniformly 0.32% below the printed 17794/16151/14320/12907; the implied sigma = 5.998e7 S/m equals COMSOL\'s built-in copper '
                 'conductivity (likely explanation: the analytic column used the COMSOL material value)',
                 proposal='benchmark against the closed form with the stated sigma (or with 5.998e7 to compare digit-for-digit with the paper); '
                 'quote the paper COMSOL numbers as secondary (agreement within 0.5%)'),
            dict(description='mode naming of the double-cell prototype', evidence='Sec. 3.3/5 text: second mode is TM110-like; Fig. 9 caption: '
                 '"TM011-like"; Fig. 7 caption: "TM010-like and TM020-like"', proposal='irrelevant for the ideal-sector benchmark; ignore'),
            dict(description='prototype frequency', evidence='Sec. 5: TM010-like at 4.2 GHz; Fig. 9 tuning range 3.6-4.0 GHz', proposal='4.2 GHz presumably '
                 'without rods (empty double cell: chi11 c/(2 pi 45 mm) = 4.06 GHz for zero-thickness partition, higher with 5 mm partition) - unverified'),
            dict(description='Eq. 16 (loss ratio P_rad/P_azi) has a typo in the paper itself (rendered p.3, not an extraction artefact)',
                 evidence='with J_{n/2}(chi) = 0 the printed form reduces to 4 n^3/(pi (n^2 - 1)) = 3.40 at n = 2, vs Table 3 0.84. With the '
                          'denominator corrected to (J_{n/2-1}(chi) - J_{n/2+1}(chi))^2 it reduces to n^3/(pi (n^2 - 1)) = 0.849/1.358/1.964/2.587 '
                          'for n = 2/4/6/8 = Table 3 (0.84/1.36/1.96/2.59) and = our quadrature',
                 proposal='use the corrected closed form n^3/(pi (n^2 - 1)) (n = number of cells) or the direct integral'),
            dict(description='form-factor volume convention',
                 evidence='Eq. 2 integrates |B0|^2 over "the cavity volume" without saying whether partitions are excluded. For Tables 2-3 '
                          '(zero-thickness partitions) the question is moot. Table 1 gives V = 1.08 L (quad-cell) and 1.02 L (sext-cell) vs '
                          'pi 45^2 200 mm^3 = 1.272 L for the full 90 mm x 200 mm bore cylinder, i.e. V is the vacuum volume (partitions '
                          'excluded, 15% / 20% of the cylinder); its relative columns follow P ~ C V Q (1.636 vs printed 1.65 quad-cell, '
                          '1.326 vs 1.32 sext-cell) and df/dt ~ C^2 V^2 Q (2.68 vs 2.72, 1.99 vs 1.98), so C is quoted together with the '
                          'vacuum V. Whether C itself was normalized by V_vac is not stated',
                 proposal='for Table 1, report axionFEM C both ways (V = vacuum volume and V = full cylinder) and compare the products C V')],
        solver_class=dict(axisymmetric_2d_possible=False, needs_3d=False,
                          reason='ideal sector cells are z-invariant TM modes: a 2D scalar Helmholtz problem on the sector (or a 3D extrusion with PEC/'
                                 'impedance ends). Not axisymmetric. Closed forms exist (P0 analytic check, no FEM needed). Table 1 and the prototype need 3D'),
        priority='P0', priority_reason='closed-form analytic reference for f, Q (walls only), C; tabulated COMSOL agrees to <0.5%',
        readme_row=dict(quantities='Tab 2: Q (no end caps) for 2/4/6/8 sector cells; Tab 3: w/w0, Q/Q0, loss ratio; Tab 1: f, Q, C, V; closed-form check',
                        geometry_complete='yes (Tables 2-3); no (Table 1: partition thickness/gap)', axisym_2d='no (2D cross-section / closed form)', needs_3d='Table 1 only'))
    dump('capp_multicell_1710', d)


# -------------------------------------------------------------------------------------------- CAPP 2205
# Fig. 6 dashed-curve provenance (see track_dashed in capp_multicell_2205_digitize.py)
_T6 = 'continuity-constrained edge tracking, every 4 px'
IDEAL6 = {
    ('C', 'ideal_2cell'): _T6 + '; only a 2-3 px red sliver below the green 4-cell dash is visible, read from its bottom edge; TRUNCATED at '
                          'Rg/Rc = 0.0565: beyond ~0.06 it is hidden under the green dash, the red MC markers and line (it stays within ~0.003 of '
                          'C = 0.646-0.648 there, not separable). The previous trace (<= 0.10) jumped onto the MC line for x > 0.046',
    ('C', 'ideal_4cell'): _T6 + ' (top edge where the lower half is covered); previous 0.6487/0.6507 jitter at x = 0.06-0.10 removed',
    ('C', 'ideal_6cell'): _T6 + '; ends at 0.267 where it meets the 6-cell MC markers',
    ('C2V2Q_norm', 'ideal_2cell'): _T6 + '; TRUNCATED at Rg/Rc = 0.085 (hard limit 0.090): for x >= ~0.085 the dash lies on the solid MC line '
                                   '(within ~0.005) and cannot be separated. The previous trace stepped onto the MC line at x = 0.065 and 0.075',
    ('C2V2Q_norm', 'ideal_4cell'): _T6 + '; under the blue 6-cell dash for x < 0.1 (read from its bottom edge, starts at 0.017); ends at 0.237 '
                                   'where it merges with the 4-cell MC line',
    ('C2V2Q_norm', 'ideal_6cell'): _T6 + '; steep drop tracked with a local fit; TRUNCATED at 0.270 (coincides with the 6-cell MC line beyond)',
}


def capp2205():
    raw = json.load(open(os.path.join(HERE, 'capp_multicell_2205_digitized.json')))
    rows = []; base = sector(0, 0.025, 5.96e7)
    paper = {0: (1, 1, 1, 0.69, 0.17), 1: (1.59, 0.54, 0.86, 0.64, 0.17), 2: (2.14, 0.42, 0.91, 0.65, 0.17), 3: (2.65, 0.34, 0.89, 0.63, 0.16),
             4: (3.16, 0.28, 0.88, 0.61, 0.16)}
    for m in range(5):
        o = sector(2 * m, 0.025, 5.96e7)
        r_f = o['f_GHz'] / base['f_GHz']
        qbar = (o['Q_no_endcaps'] * o['delta_um'] * 1e-6 / 0.025)        # Q delta / Rc
        rows.append(dict(m=m, n_cells=2 * m, nu_over_nu0=round(r_f, 4), Qbar=round(qbar, 4), Q_cyl_ratio=round(qbar * r_f, 4), C=round(o['C'], 5),
                         paper=dict(nu=paper[m][0], Q=paper[m][1], Qcyl=paper[m][2], C=paper[m][3], dnu_over_nu=paper[m][4])))
    sx = {'fig5': 0.10, 'fig6': 0.30}
    curves = []
    for fig in ('fig5', 'fig6'):
        r = raw[fig]
        ux = sx[fig] / (2235.0 - 397.0)
        for panel, ylab, upx in (('C', 'C', 0.6 / 758.0), ('C2V2Q_norm', 'C^2 V^2 Q (normalized)', 1.0 / 867.0)):
            series = []
            for name, d in r['series'].items():
                pts = d[panel]
                series.append(dict(label=name + ' (Monte Carlo mean of 100 COMSOL runs; error bar = "statistical deviation" per the caption, '
                                   'asymmetric as plotted: [lo, hi] listed separately; not necessarily 1 std)',
                                   points=[[q['x'], q['y']] for q in pts],
                                   errors=[[q.get('err_lo'), q.get('err_hi')] for q in pts],
                                   point_uncertainty_y=[round(q['uncertainty_y_px'] * upx, 4) if q.get('uncertainty_y_px') else None for q in pts],
                                   notes=[q.get('note') for q in pts]))
            for name, d in r['ideal_curves'].items():
                c = [q[:2] for q in d[panel]]
                lab = name + ' (ideal model, dashed; tracked every 4 px, gaps where the dash is hidden by markers/solid lines)'
                if fig == 'fig6':
                    lab = name + ' (ideal model, dashed; ' + IDEAL6[(panel, name)] + ')'
                series.append(dict(label=lab, x_range_kept=[c[0][0], c[-1][0]] if c else None,
                                   points=c[::3] if len(c) > 90 else c))
            curves.append(dict(
                source=('Fig. 5' if fig == 'fig5' else 'Fig. 6') + (' top' if panel == 'C' else ' bottom'),
                description=('2-cell cavity, tolerance levels 0.01/0.03/0.05/0.10 %' if fig == 'fig5' else '2-, 4-, 6-cell cavities at 0.05 % tolerance')
                            + ': ' + ylab + ' vs partition gap Rg/Rc',
                x=dict(label='Rg/Rc', unit=''), y=dict(label=ylab, unit=''), series=series,
                digitization=dict(method='native embedded RGB bitmap (2400x2250, matplotlib) ; axes from detected frame + ticks; markers = bounding-box '
                                         'centre of the series-colour blob after binary opening (disk r = 7 px) at the nominal x grid (x_from_px agrees '
                                         'with the grid to <= 0.0005 in Fig. 5 and <= 0.0004 in Fig. 6); error-bar ends from the series-colour strip through the marker; '
                                         + ('Fig. 5: partial (clipped) and hidden markers re-measured from their unoccluded top-edge profile against a '
                                            'full-size template of the same series, flagged in notes, per-point uncertainty 5 px; '
                                            'dashed ideal curve by column-wise tracking of thin runs' if fig == 'fig5' else
                                            'dashed ideal curves by continuity-constrained edge tracking (truncations in the labels)'),
                                  page=13 if fig == 'fig5' else 14, clip_bbox_pt='whole embedded image',
                                  axis_calibration=r['calibration'] | dict(log=False),
                                  uncertainty_x=round(3 * ux, 5), uncertainty_y=round(3 * upx, 4),
                                  uncertainty_basis='+-3 px (marker centre 2 px, calibration 1 px) for full markers; partial/recovered markers: '
                                                    'point_uncertainty_y (5 px, or half the marker if the top-edge fit failed); error-bar ends +-4 px '
                                                    '(bar ends hidden under other series may be underestimated); error bars shorter than the marker '
                                                    'half-size (~0.015) are reported as null; markers still hidden are null with a note',
                                  figure_png=f'capp_multicell_2205_{fig}.png', overlay_png=f'capp_multicell_2205_{fig}_overlay.png',
                                  script='digitize/capp_multicell_2205_digitize.py')))
    d = dict(
        name='capp_multicell_2205', title='CAPP multiple-cell cavity in-depth study: sector-cell closed forms and optimal partition gap',
        benchmark_rank='7',
        citations=[dict(arxiv='2205.01319', authors_short='J. Jeong, S. Youn, J. E. Kim', journal='NIM A (2023) (published version not checked)',
                        items_used=['Eqs. 4-7', 'Table 2', 'Table 3', 'Fig. 5', 'Fig. 6'])],
        software=dict(code='COMSOL Multiphysics', version='5.2', solver_type='3D eigenmode, Monte Carlo over machining tolerance (100 samples per point)',
                      boundary='copper (impedance, presumably)', mesh_notes='not given'),
        geometry=dict(units='mm', description='N-cell cylinder (N = 2m) of radius Rc split by N/2 diametral partitions (N radial half-partitions) whose '
                      'tips stop at distance Rg from the axis (centre gap). Table 2: closed form for the ideal sector (Rg = 0, thin partitions, '
                      'long cavity). Figs. 5-6: COMSOL vs Rg/Rc with every dimension randomly perturbed by the stated tolerance',
                      parameters=dict(
                          Rc=dict(value=None, unit='mm', source='not given for Figs. 5-6', note='as-built cavities: 110 mm ID (Rc = 55 mm)'),
                          height=dict(value=None, unit='mm', source='not given for Figs. 5-6', note='as-built: 215 mm for the Sec. 5 2/4/6-cell cavities; the Sec. 4 tuning-demo 6-cell cavity is a different build (220 mm)'),
                          partition_thickness=dict(value=None, unit='mm', source='not given for Figs. 5-6', note='as-built: 5 mm'),
                          Rg=dict(value='swept 0.01-0.10 (2-cell), 0.15-0.25 (4-cell), 0.24-0.28 (6-cell) in units of Rc', unit='', source='Figs. 5-6'),
                          tolerance=dict(value='0.01, 0.03, 0.05, 0.10 % random perturbation of radius of each cell, thickness and position of each partition',
                                         unit='', source='Sec. 5'),
                          tuning_rods=dict(value='none in Figs. 5-6; Fig. 2 uses eps = 10 rods, Rr/Rc ~ 0.1 (cylinder-equivalent)', unit='', source='Sec. 3'))),
        materials=dict(walls=dict(material='copper (oxygen-free)', sigma_S_per_m=None, temperature_K=None, source='not stated for the simulations')),
        conditions=dict(temperature_K=None, B0_T=None, B0_direction='z', field_map='uniform (Eq. 7 "assuming a homogeneous magnetic field")'),
        conventions=dict(form_factor_definition='C = |int B0.E dV|^2 / (int |B0|^2 dV int eps |E|^2 dV) (Eq. 2); volume of the |B0|^2 integral '
                                                 '(full cylinder vs vacuum excluding partitions) not stated - see gaps',
                         Q_definition='closed form Eqs. 5-6: Qc = (Rc/delta)/(1/Qbar + Rc/Lc); Table 2 "Q" = Qbar = Q delta/Rc normalized to m = 0; '
                                      'Qcyl = Q relative to an empty cylinder of the same frequency',
                         notes='Fig. 5/6 C^2V^2Q is normalized to the ideal (0 % tolerance) cavity with Rg = 0 per Table 3 caption; Fig. 5 caption says '
                               '"normalized to the ideally modeled cavity" - consistent only if the reference is the Rg = 0 ideal cavity (dashed curve = 1 at Rg = 0)'),
        reference_values=dict(tables=[
            dict(source='Table 2 p.7', simulated=False, description='closed-form circular-sector parameters vs m (normalized to m = 0)',
                 columns=['m', 'n_partition(=cells)', 'nu_c', 'Q', 'Q_cyl', 'C', 'dnu_c/nu_c'], units=['', '', '', '', '', '', ''],
                 rows=[[0, 0, 1, 1, 1, 0.69, 0.17], [1, 2, 1.59, 0.54, 0.86, 0.64, 0.17], [2, 4, 2.14, 0.42, 0.91, 0.65, 0.17],
                       [3, 6, 2.65, 0.34, 0.89, 0.63, 0.16], [4, 8, 3.16, 0.28, 0.88, 0.61, 0.16]]),
            dict(source='Table 3 p.13', simulated=True, description='optimal centre gap and the corresponding C/C_ideal and normalized C^2V^2Q (Monte Carlo)',
                 columns=['m', 'error_pct', 'Rg_opt/Rc', 'C/C_ideal', 'C2V2Q_norm'], units=['', '%', '', '', ''],
                 rows=[[1, 0.01, 0.03, '0.99±0.01', '0.90±0.01'], [1, 0.03, 0.05, '0.98±0.01', '0.85±0.02'], [1, 0.05, 0.06, '0.98±0.01', '0.82±0.02'],
                       [1, 0.10, 0.07, '0.97±0.01', '0.78±0.01'], [2, 0.05, 0.20, '0.98±0.01', '0.84±0.02'], [3, 0.05, 0.26, '0.97±0.01', '0.90±0.02']])],
            curves=curves,
            analytic_check=dict(description='closed form (digitize/capp_sector_analytic.py) vs Table 2; Qbar = Q delta/Rc (fixed delta), Qcyl = Qbar * nu/nu0',
                                rows=rows,
                                findings=['nu, Q, Qcyl, C of Table 2 reproduced to the printed 2 digits for all m',
                                          'the printed Eq. 6 (1/Qbar) evaluates to our quadrature (m=1: Qbar = 0.5409)',
                                          'digitized ideal (dashed) C at Rg -> 0: 2-cell 0.644-0.646, 4-cell 0.649, 6-cell 0.628 vs closed form 0.6440, '
                                          '0.6493, 0.6341: 6-cell low by ~0.9% (7 px, beyond digitization error) - finite partition thickness/mesh in the '
                                          'COMSOL ideal model, unverified'])),
        measured_values=dict(note='Fig. 7 (measured per-cell field strengths) and Sec. 5 C/C_ideal = 1.00/0.99/0.98 for 2/4/6 cells; not digitized. '
                                  'Build note (not an inconsistency): the Sec. 4 tuning demonstration uses a 6-cell cavity 110 mm ID x '
                                  '220 mm, the Sec. 5 2/4/6-cell cavities are 110 mm ID x 215 mm - two different builds'),
        gaps=[dict(parameter='Rc, height, partition thickness and copper sigma of the Fig. 5-6 models', issue='not given',
                   proposal='results are normalized (C, C^2V^2Q relative to Rg = 0) so use the as-built 110 mm ID x 215 mm, 5 mm partitions, sigma = 5.8e7, '
                            'and sweep partition thickness {thin, 5 mm} to see sensitivity; benchmark only the deterministic ideal (dashed) curves'),
              dict(parameter='Monte Carlo perturbation model', issue='"dimensions allowed to randomly vary within 0.01-0.10 %" - distribution (uniform/'
                   'normal) and which dimensions exactly are not specified', proposal='do not benchmark the MC points quantitatively; at most reproduce '
                   'the trend with a uniform distribution on cell radii and partition positions'),
              dict(parameter='partition gap shape', issue='"distance between the centre and the tip of a partition"; tip shape (square/rounded) not given',
                   proposal='square tips'),
              dict(parameter='form-factor / volume convention for finite-thickness partitions',
                   issue='Eq. 2 normalizes by int |B0|^2 dV over "the cavity volume" without saying whether the partitions are excluded. For the '
                         'as-built 110 mm ID, 5 mm partitions the excluded area is ~5.8% (2-cell, one diametral plate, less the centre gap), '
                         '~11% (4-cell), ~16% (6-cell) of pi Rc^2 (about 7% for the 2-cell plate incl. typical Rg); C_vac = C_full/(1 - a). '
                         'C^2 V^2 Q also depends on which V is used, and V_vac grows with Rg (shorter partitions), so the normalized Figs. 5-6 '
                         'curves are convention-dependent at the few-% level',
                   proposal='axionFEM must report C (and C^2 V^2 Q) both ways: V = full cylinder pi Rc^2 h and V = vacuum volume; the Table 2 closed '
                            'form (zero-thickness partitions) is convention-free')],
        inconsistencies=[
            dict(description='Eq. 7 (closed-form C) as printed gives C = 0 identically',
                 evidence='Eq. 7 (p.7) contains 1F2~(m/2+1; m/2+1, m+1; -chi_m1^2/4): with a = b1 it collapses to 0F1~(; m+1; -chi^2/4) ~ '
                          'J_m(chi_m1) = 0 (mpmath: ~1e-32 for m = 0-4). With b1 = m/2 + 2 it gives 0.64402/0.64928/0.63411/0.61317 for m = 1-4 '
                          '= our quadrature and Table 2; it still fails at m = 0 (0.5606 vs 0.6917; the sin(m phi) angular factor does not apply)',
                 proposal='do not use Eq. 7; capp_sector_analytic.py computes C by quadrature of the field solution (independent of Eq. 7)'),
            dict(description='Table 3 4-cell optimum not on the simulated grid', evidence='Table 3: Rg_opt/Rc = 0.20 for m = 2; Fig. 6 4-cell points at '
                 '0.15, 0.17, 0.19, 0.21, 0.23, 0.25 (max at 0.19: 0.840)', proposal='treat 0.20 as an interpolated optimum'),
            dict(description='normalization wording', evidence='Fig. 5 caption: "normalized to those of the ideally modeled cavity"; Table 3 caption: '
                 '"scan rates normalized to that for Rg = 0"', proposal='normalize to the ideal Rg = 0 cavity (dashed curves start at 1.0)'),
            dict(description='figure references broken / mislabelled', evidence='Sec. 5 "as can be seen in Fig. ??"; Fig. 3 caption labels the Q plot '
                 '"(b)" twice', proposal='none'),
            dict(description='Table 2 column header', evidence='"# of partition" 0/2/4/6/8 equals the number of cells (half-partitions), not the number of '
                 'diametral plates', proposal='interpret as number of cells N = 2m')],
        solver_class=dict(axisymmetric_2d_possible=False, needs_3d=False,
                          reason='Table 2 = closed form (no FEM). Ideal gap curves: partitions full height, so the TM010-like mode is z-invariant: 2D '
                                 'scalar Helmholtz on the cross-section (not axisymmetric) or a 3D extrusion. Monte Carlo points: 3D, 100 solves per point'),
        priority='P0 (Table 2 closed form) / P2 (gap curves)',
        priority_reason='Table 2 is an exact analytic check; the ideal C(Rg) and C^2V^2Q(Rg) curves are deterministic but need unstated Rc/thickness '
                        'and an unstated volume convention; the Fig. 6 dashed traces are truncated where they merge with the MC lines',
        readme_row=dict(quantities='Tab 2 closed-form nu, Q, Qcyl, C vs m; Tab 3 optimal gap; Figs 5-6 C and C^2V^2Q vs Rg/Rc (MC + ideal curves)',
                        geometry_complete='yes (Tab 2); no (Figs 5-6: Rc, height, partition thickness, MC model)',
                        axisym_2d='no (closed form / 2D cross-section)', needs_3d='MC points only'))
    dump('capp_multicell_2205', d)


if __name__ == '__main__':
    stern(); capp1710(); capp2205()
