# 02. Space Radiation Physics & Simulation Tools

> Scope: physics of the space radiation environment and the simulation tools we can integrate to build a 3-tier radiation simulator for SpaceLLM.
> Source: physics primer for the SpaceLLM simulator
> Last verified: 2026-04-24

---

## TL;DR

- Three main radiation effects on electronics in space: **SEU** (single bit-flip), **TID** (cumulative dose) and **SEL/SEFI** (latch-up / functional interrupt). The orbit and the shielding mass determine the relative weight of each.
- The standard physical-simulation toolchain is **SPENVIS (ESA, web)**, **OLTARIS (NASA, web)**, **CREME-MC (Vanderbilt, web)**, **Geant4 (C++ + Python via `geant4-pybind`)**, and the licence-restricted **FLUKA**. None of these is Python-native at the right level, SpaceLLM has to build its own stack.
- Recommended split: **L1 statistical (Poisson + per-orbit rate tables)**, **L2 physics-lite Python (AP-9 / AE-9 grids + Weibull cross-section + IRPP)**, **L3 external-tool integration (SPENVIS / OLTARIS HTTP, optional Geant4)**.

---

## 1. Radiation Primer, Precise Definitions

### 1.1 Single Event Upset (SEU)
A bit-flip in a memory cell or register caused by a single ionizing particle (proton, heavy ion, or secondary from nuclear reaction) depositing enough charge in a sensitive volume to exceed the cell's critical charge **Q_c**. SEUs are non-destructive and recoverable by rewrite/scrub.
- Characterized by a **cross-section curve σ(LET)** that is conventionally fit by a 4-parameter Weibull:
  σ(LET) = σ_sat · (1 − exp{ −[(LET − L_0)/W]^s })
  where σ_sat is saturation cross-section, L_0 is onset (threshold) LET, W is width, s is shape. (Source [11], [12])

### 1.2 Single Event Latch-up (SEL)
A high-current, potentially destructive parasitic-thyristor trigger in a CMOS structure caused by a single particle. Mitigation: power cycling, current-limiters, SOI processes. CREME-MC characterizes SEL immunity by an LET threshold, e.g. Jetson AGX Xavier shows SEL immunity up to LET 125 MeV·cm²/mg [9].

### 1.3 Single Event Functional Interrupt (SEFI)
A particle hit that disrupts a control structure (FSM, configuration register, clock tree) such that the device stops functioning correctly until reset. Common in FPGAs and SoCs with deep cache hierarchies, Jetson Xavier hits in L2/L3 cache control logic produced this class of error [9].

### 1.4 Total Ionizing Dose (TID)
Cumulative absorbed dose in silicon, measured in **rad(Si)** or **Gy(Si)** (1 Gy = 100 rad). Causes threshold voltage shift, leakage-current increase, and sub-threshold slope degradation in MOS devices [13]. Annealing ("self-healing") happens at elevated temperature; high-dose-rate exposures show partial annealing not seen at low-dose-rate (LDR) exposures, relevant because spaceflight is intrinsically LDR [13]. Standard tests: MIL-STD-883 TM 1019.9, ESCC 22900, ECSS-Q-ST-60-15 [13].

### 1.5 Cross-Section, LET, FIT
- **LET (Linear Energy Transfer)**: energy deposited per unit path length, MeV·cm²/mg in Si.
- **Cross-section σ**: effective area per bit (or per device) for an upset to occur, cm²/bit or cm²/device.
- **FIT (Failures in Time)**: failures per 10⁹ device-hours. 1 SEU/device/day ≈ 4.17×10⁷ FIT.

---

## 2. Orbit-Specific Environment Table

> All numbers cited; mark **[est.]** where derived rather than measured. Dose values are typical Si-equivalent under ~1 g/cm² Al shielding unless noted.

| Orbit | Altitude / Locus | Trapped p⁺ flux (E>10 MeV) | Trapped e⁻ flux (E>1 MeV) | GCR dose (interior) | Dominant SEU driver | Cited dose rate |
|---|---|---|---|---|---|---|
| **LEO 400 km, 51.6°** (ISS) | 400 km, 51.6° incl. | High only in SAA crossings (≈10×/day, minutes each) | Low | ≈40 µGy/day GCR component | Trapped protons in SAA + GCR | **GCR ≈0.354–0.770 nGy/s (≈30–66 µGy/day) at high lat.; whole-orbit ≈39.7 µGy/day total in Si, ≈226 µSv/day H*(10)** [4] |
| **LEO 600 km SSO** (~98° polar, EO sats) | 600 km, sun-sync | Higher than ISS due to SAA depth + horn entries | Higher (auroral horns) | Similar to ISS GCR | Trapped p⁺ + horn e⁻ | Modeled with AP-8 MAX, AE-8 MAX in SPENVIS [1][7]; specific µGy/day not measured publicly, must compute |
| **MEO / GPS** (~20,200 km, 55°) | Inside outer e⁻ belt | Moderate (sub-belt) | **Very high** (intense MeV electrons, outer belt) | Variable | Trapped e⁻ → TID + IDD; protons → SEU/SEL | "Particularly intense" outer-belt electrons; flight TID data is limited, BDD-I dosimeter on GPS is reference [16] |
| **GEO** (~35,786 km, 0°) | Outside inner belt; outer e⁻ belt during disturbances | Negligible trapped | Highly variable (CIR/storm-driven MeV electrons) | Highest of Earth orbits | SPE protons (storms), GCR heavy ions, e⁻ surface charging | Mission TID range ≈10–100 krad(Si) over multi-year missions [14] |
| **Cislunar / L1-L2** | 0.01–1.5×10⁶ km from Earth | None | None (outside belts) | Pure GCR + SPE | GCR Fe-56, O-16, Si-28 heavy ions | Artemis I crewed-equivalent estimate ≈22.3 mSv over mission [5] |
| **Mars cruise (cruise interior)** | Interplanetary | None | None | Pure GCR + SPE | GCR + occasional SPE | **MSL/RAD: 332±23 µGy/day (Si), tissue 458±32 µGy/day, dose-equiv. 1.75±0.30 mSv/day; SPE peaks >10,000 µGy/day** [3] |
| **Mars surface** | Gale crater |, |, | Atmospheric attenuation ~0.5× cruise | GCR + secondaries | **0.210±0.040 mGy/day at Gale (solar max)** [3] |

GCR composition by abundance (BON2020 model): ~87% protons, ~12% alphas, ~1% Z>2, but the **dose contribution** of heavy ions (Fe-56, Si-28, O-16) is disproportionately large because dose ∝ Z² for ionization. The Badhwar–O'Neill 2020 model is the NASA-standard GCR spectrum, calibrated to AMS-02 and PAMELA data and modulated by ACE/CRIS measurements [6][15].

### 2.1 SEU rate ballparks (modern devices)

- **NVIDIA Jetson AGX Xavier (12 nm FinFET, GPU SoC)**: SEE rate **<7.0×10⁻⁶ events/device/day** "for typical orbits", SEL-immune to LET 125 MeV·cm²/mg under proton irradiation [9]. Errors dominated by L2/L3 cache.
- **NVIDIA Jetson Nano**: TID tolerance "beyond 20 krad(Si)", adequate for short LEO smallsats [8].
- **NVIDIA Jetson Orin AGX**: characterized for SEE+TID; numbers in the IEEE NSREC 2023 paper [9b].
- **Generic 7 nm FinFET SRAM**: laser-and-heavy-ion testing reported in IEEE TNS 2024 [11]; Weibull L_0, σ_sat values are device-specific and **must be cited per part**, do NOT assume.

> **Rule for SpaceLLM**: never bake hardcoded SEU rates per chip without a citable Weibull fit. The architecture should accept Weibull (σ_sat, L_0, W, s) + RPP (x, y, z, Q_c) as device parameters and compute the rate from the orbit's LET spectrum.

---

## 3. Van Allen Belt & Environment Models

| Model | Particle | Source / Auth | Status | Access |
|---|---|---|---|---|
| **AP-8 / AE-8** | Trapped p⁺ / e⁻ | NASA NSSDC (1976/1991) | Legacy, still industry-default | SPENVIS, CCMC, IRBEM-LIB [1][7] |
| **AP-9 / AE-9** (v1.50.001) | Trapped p⁺ / e⁻ | AFRL (Ginet et al. 2013) | Modern consensus model with uncertainty quantification | SPENVIS, vdl.afrl.af.mil [7] |
| **CRRESPRO / CRRESELE** | Trapped p⁺ / e⁻ from CRRES mission | NASA | Specialty / storm-time | SPENVIS |
| **SPM (Solar Proton Model)** | SPE protons | AP-9 suite | Probabilistic | SPENVIS [7] |
| **ESP / PSYCHIC / King** | SPE long-term fluence | NASA | Mission-fluence | SPENVIS |
| **Badhwar–O'Neill 2020 (BON2020)** | GCR all-ion | NASA Slaba & Whitman 2020 | NASA standard | Software catalog MSC-26835-1 [6] |
| **CREME96 GCR + SEP** | LET spectrum behind shield | NRL/Vanderbilt | Industry standard for SEU rate | creme.isde.vanderbilt.edu [10] |
| **MSIS / NRLMSISE-00** | Atmosphere (drag, secondaries) | NRL | Standard | pyproj/pymsis Python wrappers (separate) |

---

## 4. Simulation Tool Catalog

| Tool | Maintainer | License | Interface | Python? | What we get |
|---|---|---|---|---|---|
| **SPENVIS** | ESA | Free, registration required | Web UI; produces CSV/text outputs; **no public REST API** [1] | Indirect (parse downloads) | AP-8/AE-8/AP-9/AE-9, CREME96 (legacy), GCR (Nymmik, ISO-15390), SHIELDOSE-2, trajectory generator |
| **NASA OLTARIS** | NASA LaRC | Free, registration required | Web UI, no documented API [2] | No | HZETRN transport through user-defined slab/sphere shielding; GCR, SPE, LEO trajectories, Mars/lunar surface; TID and dose-equivalent |
| **CREME-MC** (incl. CREME96) | Vanderbilt ISDE | Free, registration | Web UI + downloadable code | No (web-only) | LET spectrum behind shield; SEU rate via **RPP**, **IRPP (Weibull)**, **PROFIT**, **HUP**; SEL [10] |
| **Geant4** | CERN/SLAC/KEK collab. | Open-source (Geant4 license) | C++ native; Python via `geant4-pybind` (pybind11), `g4ppyy` (cppyy), legacy `g4py` | **Yes** [17][18] | Full Monte Carlo physics: hadronic, EM, RadBelt extension (proton/electron transport in belts), Geant4-DNA (sub-cellular). Heavyweight. |
| **FLUKA** | CERN (post-2019) + INFN fork | Free for non-commercial; **single-user license, 2-yr expiry, citation required** [19] | Fortran; Flair GUI | No native Python | Multi-particle transport, very strong hadronic. Two divergent codebases since 2019: `fluka.cern` vs INFN's `flukafiles.it`. |
| **PHITS** | JAEA | Free (registration) for non-commercial | Fortran | No | Used in MSL/RAD comparison studies [3]. |
| **MULASSIS** | ESA via SPENVIS | Web tool wrapping Geant4 | SPENVIS UI | No | 1-D shielding analysis (slab geometry) using Geant4 physics |
| **Geant4 RadBelt / Planetocosmics** | Geant4 collab. | Open-source | C++ | Via geant4-pybind | Trapped particle transport with Earth's magnetic field |
| **HZETRN-2020** | NASA LaRC | NASA software request | Fortran | No | Deterministic transport; backbone of OLTARIS |
| **IRBEM-LIB** | LANL/CNRS | Open-source | Fortran + Python wrapper | **Yes** | Magnetic field, L-shell, AE-8/AP-8/AE-9/AP-9 evaluation. **High-value: Python-native AE/AP access for SpaceLLM L2.** |
| **SpacePy** | LANL | Open-source (BSD) | Python | **Yes** | Wraps IRBEM, OMNI data, time-series space-weather utilities. **Recommended core dep.** |
| **pymsis** | community | Open-source | Python | **Yes** | NRLMSISE-00 atmosphere |

### 4.1 What is missing
There is **no first-class Python end-to-end space-radiation library** that goes orbit → LET spectrum → SEU rate. SpacePy + IRBEM gets you trapped fluxes; CREME-MC gets you LET spectra (web-only); CREME96's IRPP integral has to be reimplemented or the user has to upload the LET spectrum manually. This is the niche SpaceLLM's L2 should fill in Python.

---

## 5. Bit-Flip Rate Estimation Math

### 5.1 RPP (Rectangular Parallelepiped) method
The sensitive volume is modeled as a box (x, y, z) µm. The deposited charge from a particle traversing path length ℓ at LET L is
  Q_dep = L · ℓ · ρ_Si / (3.6 eV/e-h) ≈ 10.8 × L × ℓ  (fC, with L in MeV·cm²/mg and ℓ in µm)
An SEU occurs if Q_dep ≥ Q_c. CREME-MC default z values: 0.5 µm (SOI/SOS), 2 µm (CMOS) [10].

### 5.2 IRPP (Integral RPP, Weibull)
  Rate = ∫ φ(L) · σ(L) dL
where φ(L) is the differential LET-spectrum flux behind shield (from CREME96/SPENVIS) and σ(L) is the Weibull cross-section curve fit to ground test data [12]. This is the standard heavy-ion SEU-rate calculation method.

### 5.3 Petersen Figure of Merit (FoM), quick estimate
Petersen, Langworthy & Diehl (1983) [12]:
  Rate (errors/bit-day) ≈ C · σ_sat / L_0²
where C is an orbit-dependent constant (≈ 5×10⁵ for GEO solar-min behind 100 mil Al, with σ_sat in cm²/bit and L_0 in MeV·cm²/mg), useful for sanity checks but **explicitly unreliable** for sub-100 nm FinFETs where multi-bit upsets and proton direct ionization break the assumptions [11][12].

### 5.4 Proton-induced SEU
Two regimes:
- **Indirect (nuclear reaction)**, proton creates a recoil ion that deposits charge. Modeled by Bendel two-parameter fit or by Monte Carlo (Geant4/FLUKA/CREME-MC).
- **Direct ionization**, relevant for ≤65 nm nodes, low-energy protons (<2 MeV). Requires special RPP parameter extraction [12]. Critical for 7 nm/5 nm devices in SpaceLLM scope.

---

## 6. Recommended Stack for SpaceLLM 3-Tier Simulator

### Level 1, Statistical (real-time, in-loop with the LLM)
- **Inputs**: pre-computed orbit-environment table (this doc §2) keyed by orbit name; chip class (e.g., "12 nm GPU", "5 nm NPU"); shielding (g/cm² Al).
- **Math**: Poisson SEU process with rate from lookup; TID accumulator; configurable SEL trigger.
- **Why**: ≤1 ms per token; runs anywhere; no scientific dependencies.
- **Limit**: doesn't change with shield thickness or solar phase beyond table values.

### Level 2, Physics-lite Python (offline + scenario sweeps)
Python stack:
- `spacepy` + `IRBEM-LIB` Python wrapper → AE-9/AP-9 trapped fluxes along trajectory.
- Local **BON2020** GCR spectrum (port the public NASA code, or wrap MSC-26835-1 binary).
- Local **CREME96-style LET-spectrum integrator**: SHIELDOSE-2-style power-law-shielding attenuation + ion-by-ion LET binning.
- Local **IRPP integrator** for Weibull cross-sections (eq. §5.2).
- `numpy` + `scipy` only, no Geant4 dependency.
- **Output**: SEU/bit/day, SEL/device/day, TID rate (rad/day) per orbit phase.
- **Validation**: cross-check against published SPENVIS run results for ISS/GEO/Molniya (canonical test orbits).

### Level 3, External tool integration (offline, high-fidelity)
- **SPENVIS**: scripted browser session (Playwright) → submit project → parse CSV results. No API but the workflow can be automated for scheduled runs.
- **OLTARIS**: same pattern (web-form scraping). Use for shielded TID and HZETRN dose.
- **Geant4** (optional, "Pro" tier): containerized Geant4 + `geant4-pybind` for full Monte Carlo of a custom geometry. Gated behind an environment flag because builds are heavy.
- **CREME-MC**: web-only; we cite/reproduce its IRPP locally in L2.

### Hand-off contract
The L1/L2/L3 boundary is one schema:
```
RadiationStep {
  t: float                    # mission elapsed time (s)
  orbit_state: {alt, lat, lon, L_shell, B}
  flux_p:   array[E]          # protons / (cm²·s·MeV)
  flux_e:   array[E]          # electrons / (cm²·s·MeV)
  flux_ion: array[Z, E]       # heavy-ion differential flux
  let_spectrum: array[LET]    # for IRPP
  shield_g_cm2: float
}
DeviceModel {
  weibull: (sigma_sat, L0, W, s)
  rpp: (x_um, y_um, z_um, Qc_fC)
  bits: int
  sel_let_threshold: float
  tid_tolerance_krad: float
}
=> SEU_rate, SEL_rate, dTID/dt
```
L1 returns the same object with table-driven values; L2 computes from physics; L3 fills it from external runs. The rest of SpaceLLM (the LLM-injection and recovery layers) only sees this contract.

---

## 6.1 Concrete first-week implementation list

1. Vendor `numpy`/`scipy` + `spacepy` into the project; verify IRBEM Python wheels build on the target platform.
2. Write `radenv/orbits.py` with the table from §2 as authoritative defaults (cite each row).
3. Write `radenv/weibull.py` (4-param Weibull) and `radenv/irpp.py` (numerical integral with trapezoidal LET binning).
4. Capture three reference LET spectra (ISS, GEO, Mars cruise) from SPENVIS as static CSVs under `packages/docs/docs/let_spectra/` for L1/L2 validation.
5. Stub the L3 SPENVIS scraper but do not depend on it for CI.

---

## 7. Sources

1. **SPENVIS, ESA Space Environment Information System**. https://www.spenvis.oma.be/ (v4.6.14, 2026-03-05).
2. **OLTARIS, On-Line Tool for the Assessment of Radiation in Space (NASA)**. https://oltaris.nasa.gov/
3. **Zeitlin et al., "Variations of dose rate observed by MSL/RAD in transit to Mars"**, A&A 2015. https://www.aanda.org/articles/aa/full_html/2015/05/aa25680-15/aa25680-15.html, silicon dose 332±23 µGy/day, tissue 458±32 µGy/day, dose-eq. 1.75±0.30 mSv/day cruise; >10,000 µGy/day SPE peaks; Mars surface 0.210±0.040 mGy/day.
4. **Narici et al., "Radiation survey in the International Space Station"**, J. Space Weather Space Clim. 2015. https://www.swsc-journal.org/articles/swsc/full_html/2015/01/swsc150037/swsc150037.html, GCR 0.354–0.770 nGy/s, dose-eq. 1.21–6.05 nSv/s, whole-orbit ~39.7 µGy/day Si.
5. **Berger et al., "Space radiation measurements during the Artemis I lunar mission"**, Nature 2024. https://www.nature.com/articles/s41586-024-07927-7, Artemis I dose-rate reductions for shielding/orientation; cited 22.3 mSv mission estimate.
6. **Slaba & Whitman, "The Badhwar-O'Neill 2020 GCR Model"**, Space Weather 2020. https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020SW002456. NASA software catalog: https://software.nasa.gov/software/MSC-26835-1
7. **Ginet et al. 2013, "AE9/AP9/SPM" radiation belt models**; AE9/AP9 distribution: https://www.vdl.afrl.af.mil/programs/ae9ap9/ . CCMC AE-8/AP-8 access: https://ccmc.gsfc.nasa.gov/models/AE-8_AP-8_RADBELT~1.0/ . NTRS comparison: https://ntrs.nasa.gov/citations/20160013712
8. **Slater et al., "Total Ionizing Dose Radiation Testing of NVIDIA Jetson Nano GPUs"**, IEEE 2020. https://www.researchgate.net/publication/347869525, TID >20 krad(Si).
9. **Hiemstra & Jin, "Single Event Effect Evaluation of the Jetson AGX Xavier Module Using Proton Irradiation"**, IEEE 2021. https://ieeexplore.ieee.org/document/9325840/, SEE rate <7.0×10⁻⁶ events/device/day "typical orbits", SEL-immune ≤125 MeV·cm²/mg.
   - 9b. **Lovelly et al., "Single Event Effects and Total Ionizing Dose Radiation Testing of NVIDIA Jetson Orin AGX SoM"**, IEEE NSREC 2023. https://ieeexplore.ieee.org/document/10265818/
   - 9c. **Rodriguez-Ferrandez et al., "Sources of Single Event Effects in the NVIDIA Xavier SoC Family under Proton Irradiation"**, IEEE 2022. https://www.researchgate.net/publication/363907081
10. **CREME-MC / CREME96 (Vanderbilt ISDE)**. https://creme.isde.vanderbilt.edu/ . RPP method docs: https://creme.isde.vanderbilt.edu/CREME-MC/help/rpp-method . CRÈME96 NTRS: https://ntrs.nasa.gov/api/citations/20120016823/downloads/20120016823.pdf
11. **IEEE Trans. Nuclear Science, vol. 71, no. 8, Aug 2024**, 7 nm FinFET SRAM SEU laser/heavy-ion correlation: https://cds.cern.ch/record/2915017/files/document.pdf . Threshold/characteristic LETs in SRAM SEU curves: https://ieeexplore.ieee.org/document/10042426/
12. **Petersen, "Single Event Effects in Aerospace"**, Wiley 2011, ch. 17 (IRPP limitations). https://onlinelibrary.wiley.com/doi/10.1002/9781118084328.ch17 . Petersen FoM background and proton direct-ionization in sub-µm: https://ui.adsabs.harvard.edu/abs/2022ITNS...69..254L/abstract
13. **Total Ionizing Dose mechanisms**, USPAS lecture: https://uspas.fnal.gov/materials/19NewMexico/Radiation/lecture_6.pdf . MIL-STD-883 TM 1019 / ESCC 22900 references: https://www.seibersdorf-laboratories.at/en/products/ionizing-radiation/radiation-hardness-assurance/total-ionizing-dose-testing
14. **Total radiation dose at GEO (overview)**: https://ieeexplore.ieee.org/document/1420731/ . 10–100 krad(Si) typical-mission range (multiple sources, e.g., LASER2COTS https://www.laser2cots.com/en/article/25.orbit.html).
15. **HZETRN background and OLTARIS implementation**, see [2]. NASA TM-20220011775 "Mission Radiation Environment Modeling and Analysis": https://ntrs.nasa.gov/api/citations/20220011775/downloads/Mission_Radiation_Modeling_STI.pdf
16. **GPS / MEO dosimetry, BDD-I instrument**: https://www.osti.gov/biblio/5194048 . SaRIF satellite-risk system: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021SW002823
17. **geant4-pybind (HaarigerHarald)**: https://github.com/HaarigerHarald/geant4_pybind . PyPI: https://pypi.org/project/geant4-pybind/
18. **g4ppyy automated bindings**: https://arxiv.org/abs/2412.05593
19. **FLUKA (CERN)** licensing & overview: https://fluka.cern/about, https://fluka.cern/download/licences/fluka-single-user-licence . 2024 overview: https://www.epj-n.org/articles/epjn/full_html/2024/01/epjn20240025/epjn20240025.html

---

## Appendix A, Numbers we did NOT cite and why

The following commonly-quoted figures were **not** included as authoritative because we could not confirm a primary source within the time budget; they are flagged here so a future reviewer can fill them in:

- Per-bit-day SEU rate for **Google Coral / Snapdragon** in space, only commercial-Jetson on-orbit data was found.
- **Specific µGy/day for 600 km SSO**, search returned modeling references but no measured value with primary citation; must be computed via SPENVIS run.
- **GEO krad/year during solar maximum behind 1 g/cm² Al**, only the 10–100 krad mission-range was cited; the per-year figure depends strongly on shielding and storm-electron history and should be computed not asserted.
- **AP-9 v1.50.001 release date / current version**, CCMC and AFRL pages were referenced but the version string was not pinned.

These are the gaps to close in research round 2 before L2 implementation.
