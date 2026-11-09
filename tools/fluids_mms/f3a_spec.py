"""Fixed constants of the transient Navier-Stokes cases (time grids, tolerances, error-control bands), used by
xcode/dolfinx_tr.py."""
from __future__ import annotations

FORMS = ("lap", "sym")
NEWTON_RTOL = 1e-12
SOLVER_MAX_RESIDUAL = 1e-11
FLOOR_REL = 1e-10

# ---------------------------------------------------------------- T1 temporal MMS (space-exact fields)
T1_T = 1.0
T1_MESH = {"builder": "structured", "N": [8]}            # h = 1/8, space-exact: any affine mesh
T1_MESH_CHECK = {"builder": "structured", "N": [16]}     # diagnostic only (space independence)
T1_NT = [40, 80, 160, 320, 640]                           # dt = T / N
T1_CYCLIC = {"kind": "cyclic", "pattern": [1.0, 1.25, 0.8, 1.1, 0.9],
             "construction": "step sizes proportional to the pattern repeated N/5 times, scaled to sum to T"}
T1_MAX_STEP_RATIO = 1.45                                   # cyclic grid's largest ratio 1.375 (< 1 + sqrt 2)
T1_VARIANTS = [(1, "newton", "uniform"), (1, "semi", "uniform"), (2, "newton", "uniform"), (2, "semi", "uniform"),
               (2, "newton", "cyclic"), (2, "semi", "cyclic")]
ORDER_TOL = 0.15

# ---------------------------------------------------------------- T2 error control
T2_FIELD = "f3a_tr_sq_adapt"
T2_T = 1.0
T2_MESH = {"builder": "structured", "N": [8]}
T2_EFF_T = 0.4                                             # t_n of the effectivity steps (inside the front's flank)
T2_EFF_H = {"1": [0.01, 0.005], "2": [0.02, 0.01]}          # h and h/2 (constant history)
T2_EFF_VARHIST = {"omegas": [2.0, 0.5], "h": 0.01}         # BDF2, history steps h/omega
T2_RTOLS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
T2_H0 = 1e-3
T2_REJ = {"rtol": 1e-4, "h0": 0.25}
# The error-control band is set from the spread of several legitimate controllers (NGSolve, PETSc TSBDF,
# safety/cap variants, an all-DOF estimate), which give err/rtol 5.3 ... 14.7 at the two tightest tolerances and
# slope 0.53 ... 0.59; the effectivity window [0.5, 2] allows a further factor 2^(2/3) = 1.6 either way.
R74_BAND = {"C_hi": 30.0, "C_lo": 3.0, "tightest_k": 2, "slope_band": [0.45, 0.70]}
T2_TOL_VARIANTS = [{"safety": 0.8, "ratio_max": 1.5}, {"safety": 0.95, "ratio_max": 2.4}]   # legitimate controllers
T2_NOADAPT_DT = 0.01                                       # fixed steps, rtol ignored
R75_STAB_BAND = [0.85, 1.15]                                # the BDF2 stability ratio of exact estimators 1.06
T2_AXI = {"field": "f3a_tr_axi_poly", "mesh": {"builder": "structured", "N": [8]}, "t_n": 0.4,
          "h": {"1": [0.01, 0.005], "2": [0.02, 0.01]}}    # axisymmetric effectivity and est_norm (r-weighted)
R75_NORM_BAND = [0.999, 1.001]                              # reported est_norm / the check module's r-weighted L2 norm of est
T2_RATIO_MAX = 2.0                                         # controller cap (inside BDF2's zero-stability bound 1 + sqrt 2)

# ---------------------------------------------------------------- T3 Taylor-Green
T3_FIELD = "f3a_tr_taylor_green"
T3_FAMILY = {"builder": "rect", "nx": 4, "ny": 4, "box": [0.0, 1.0, 0.0, 1.0], "cell": 0.25, "k": [0, 1, 2, 3, 4]}
T3_N = [4, 8, 16, 32, 64]                                  # cells per side (h = 1/N)
T3_T = 0.02                                                # short window: the BDF1 starter's O(dt^2) error must stay
T3_NT = 80                                                 #   << the finest spatial error (dt = 2.5e-4; dt/2 at the finest)
T3_VARIANTS = [("lap", "newton"), ("sym", "newton"), ("sym", "semi")]

# ---------------------------------------------------------------- T4 / T5 Womersley
W_T_SPACE = 0.0078125                                      # spatial study: 1/128 period from the exact state
W_NT_SPACE = 256
W_T_TIME = 1.0                                             # temporal study: one period
W_NT_TIME = [20, 40, 80, 160, 320]
W2_FAMILY = {"builder": "rect", "nx": 2, "ny": 4, "box": [0.0, 1.0, -1.0, 1.0], "cell": 0.5, "k": [1, 2, 3, 4, 5]}
W2_TIME_MESH = {"builder": "rect", "nx": 2, "ny": 4, "box": [0.0, 1.0, -1.0, 1.0], "cell": 0.5, "k": [4]}
WA_FAMILY = {"builder": "rect", "nx": 2, "ny": 2, "box": [0.0, 1.0, 0.0, 1.0], "cell": 0.5, "k": [1, 2, 3, 4, 5]}
WA_TIME_MESH = {"builder": "rect", "nx": 2, "ny": 2, "box": [0.0, 1.0, 0.0, 1.0], "cell": 0.5, "k": [4]}

# ---------------------------------------------------------------- T7 initial-data projection
T7_FIELD = "f3a_tr_sq_proj"
T7_N = [8, 16, 32, 64, 128]
T7_DT = 1e-3
T7_DIV_MAX = 1e-9           # a round-off quantity (a Stokes projection gives 1.2e-11)
T7_DIR_T0_MAX = 1e-12       # nodal Dirichlet data of the INITIAL state

# ---------------------------------------------------------------- T6 energy identities
T6_PLANAR = {"field": "f3a_tr_sq_energy", "mesh": {"builder": "structured", "N": [8]}}
T6_AXI = {"field": "f3a_tr_axi_energy", "mesh": {"builder": "structured", "N": [32]}}
T6_T = 1.0
T6_NT = 20
