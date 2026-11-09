# References

The literature the physics and validation pages rely on, grouped by area. Each page also lists its own sources next
to the equations that use them. The papers are not redistributed with axionFEM; the arXiv numbers below lead to them.

## Lumped-element detectors and the optimal scan

- S. Chaudhuri, K. D. Irwin, P. W. Graham, J. Mardon, arXiv:1803.01627 (optimal impedance matching and scan
  strategy); S. Chaudhuri et al., arXiv:1904.05806 (Bode–Fano limit). Used in
  [scan_and_signal.md](physics/scan_and_signal.md) and [readout_circuit.md](physics/readout_circuit.md).
- J. W. Foster, N. L. Rodd, B. R. Safdi, arXiv:1711.10489 (axion field statistics, likelihood and look-elsewhere
  effect). Used in [scan_and_signal.md](physics/scan_and_signal.md).
- J. N. Benabou, J. W. Foster, Y. Kahn, B. R. Safdi, C. P. Salemi, arXiv:2211.00008, Phys. Rev. D 108, 035009 (2023)
  (lumped-element detectors beyond the magnetoquasistatic limit). Compared in [fullwave.md](validation/fullwave.md)
  (the paper's dipole radiation Q) and [comsol_import.md](comsol_import.md) (the paper's COMSOL models, which are not
  distributed); methods in [multipole_scan.md](physics/multipole_scan.md).
- A. AlShirawi et al., arXiv:2302.14084, Phys. Rev. D 112, 052001 (2025) (DMRadio pickup and effective-inductance
  conventions). Used in [multipole_scan.md](physics/multipole_scan.md).
- Y. Kahn, B. R. Safdi, J. Thaler, arXiv:1602.01086 (ABRACADABRA); J. L. Ouellet et al., arXiv:1810.12257 and
  arXiv:1901.10652 (ABRACADABRA-10 cm results and instrument).
- DMRadio-m³, arXiv:2204.13781, Phys. Rev. D 106, 103008 (2022); DMRadio-GUT, arXiv:2203.11246, Phys. Rev. D 106,
  112003 (2022); DMRadio-Core, V. Ankel et al., arXiv:2604.16602, Phys. Rev. D 114, 092004 (2026); DMRadio-50L,
  arXiv:2210.07215 (SciPost Phys. Proc. 12, 036 (2023)) and arXiv:2607.05771. Compared, quantity by quantity, in
  [published_benchmarks.md](validation/published_benchmarks.md).
- Axion–photon coupling of the QCD axion: G. Grilli di Cortona et al., arXiv:1511.02867; lattice values of C_γ,
  arXiv:2603.29153 (the default, C_γ = 1.77) and arXiv:2609.22698. Used in
  [scan_and_signal.md](physics/scan_and_signal.md).

## Readout and dc SQUIDs

- V. Ankel et al., "Noise limits for dc SQUID readout of high-Q resonators below 300 MHz", J. Appl. Phys. 138,
  094505 (2025), arXiv:2504.20398. The basis of [squid_chain.md](physics/squid_chain.md).
- A. A. Clerk et al., Rev. Mod. Phys. 82, 1155 (2010) (quantum noise and measurement).

## Cavity haloscopes

- P. Sikivie, Phys. Rev. Lett. 51, 1415 (1983); Phys. Rev. D 32, 2988 (1985) (the haloscope).
- S. Al Kenany et al., arXiv:1611.07123 (HAYSTAC design); B. M. Brubaker et al., arXiv:1706.08388 (HAYSTAC analysis);
  HAYSTAC Collaboration, arXiv:2301.09721 (Phase II); M. Malnou et al., arXiv:1809.06470 (squeezed-state receiver);
  D. Kim et al., arXiv:2001.05605 (scan-rate conventions). Used in
  [haloscope_sensitivity.md](physics/haloscope_sensitivity.md).
- B. M. Brubaker, PhD thesis, arXiv:1801.00835 (definitions of Q_L, β); P. Balleyguier, Part. Accel. 57, 113 (1997)
  (external Q of a probed pillbox). Used in [coupling.md](cavity/coupling.md).
- Published haloscope runs whose parameters `axionfem.haloscope_presets` uses (the module gives the full
  references): arXiv:1804.05750, arXiv:2504.07279, arXiv:1610.02580, arXiv:2008.01853, arXiv:2210.10961,
  arXiv:2205.05574, arXiv:2407.18586, arXiv:2012.09498, arXiv:1903.06547.
- Published cavity simulations compared in [cavity_reproductions.md](validation/cavity_reproductions.md),
  [cavity_tracking.md](validation/cavity_tracking.md) and [cavity_xsec.md](validation/cavity_xsec.md): RADES,
  arXiv:1803.01243 and arXiv:2002.07639; QUAX, arXiv:2004.02754, arXiv:2201.04223 and arXiv:2410.07774; CAPP
  multiple-cell cavities, arXiv:1710.06969 and arXiv:2205.01319; HAYSTAC one-rod cavity, arXiv:2006.01248; TASEH,
  arXiv:2205.01477; BI-RME, arXiv:2107.03137; Stern et al. (multi-post cavities), arXiv:1603.06990; KLASH,
  arXiv:1707.06010; ADMX empty-cylinder modes (PhD thesis), arXiv:2408.03444.

## Surface resistance

- E. Métral, arXiv:1108.1643; D. Valentinis, arXiv:2003.06619 (anomalous skin effect); G. Ciovati, arXiv:1501.07398
  (RF superconductivity); D. Gonnella et al., arXiv:1509.04127 (trapped-flux losses). Used in
  [surface_resistance.md](physics/surface_resistance.md) and the SRF external benchmarks.

## SRF cavities and heterodyne detection

- B. Aune et al., arXiv:physics/0003011 (TESLA cavities); N. Juntong, R. M. Jones, arXiv:0910.1305 (low-loss cells);
  R. Lasenby, arXiv:1912.11056, and A. Berlin et al., arXiv:1912.11048 (heterodyne axion detection); S. Holmes et al.,
  arXiv:1306.5022 (Project X reference design, the cell geometry). Used in [srf/README.md](srf/README.md) and
  [sensitivity.md](srf/sensitivity.md).

## Superconductor critical surfaces

- L. Bottura, IEEE Trans. Appl. Supercond. 10, 1054 (2000); A. Godeke et al., Supercond. Sci. Technol. 19, R100
  (2006); L. Bottura, B. Bordini, IEEE Trans. Appl. Supercond. 19, 1521 (2009); F. Grilli et al., "Self-consistent
  modeling of the Ic of HTS devices: how accurate do models really need to be?", IEEE Trans. Appl. Supercond. 24,
  8000508 (2014). Used in [hts.md](validation/hts.md).

## Numerical methods

- MFEM: R. Anderson et al., Comput. Math. Appl. 81, 42 (2021); hypre (BoomerAMG, AMS); METIS; Eigen; STRUMPACK.
- Y. Nakatsukasa, O. Sète, L. N. Trefethen, SIAM J. Sci. Comput. 40, A1494 (2018) (the AAA rational approximation
  behind [adaptive_sweep.md](physics/adaptive_sweep.md)).
