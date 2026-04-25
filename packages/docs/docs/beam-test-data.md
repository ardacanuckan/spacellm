# 08. Real Beam-Test Data (publicly extracted)

_Generation date: 2026-04-25. All numbers in this file were fetched from
the cited URL and are direct table reads unless explicitly marked
"plot, approximate". No paywalled content used. No estimates._

---

## 1. Microchip RT PolarFire FPGA (28 nm SONOS-flash)

- **Source URL:** https://ww1.microchip.com/downloads/aemdocuments/documents/fpga/ProductDocuments/SupportingCollateral/rt_polarfire_radiation_test_report_2020-02-04.pdf
- **Document:** Wang et al., "Radiation Characteristics of FPGA Using Complementary-SONOS Configuration Cell," Microchip, 2020-02-04.
- **DUT:** MPF300 (28 nm Microsemi/Microchip PolarFire); device features 300,000 logic elements, 952 LSRAM blocks (20 Kb each), 2,772 µSRAM blocks (768 bits each).
- **Heavy-ion facility:** Texas A&M University (TAMU) cyclotron, 25 MeV/n tune.
- **TID source:** gamma ray.
- **Date fetched:** 2026-04-25.
- **Extraction method:** Direct table read of "Appendix: SEU Raw Data" and Table 1 (Weibull parameters) and Chip SEL section.

### 1.1 Weibull parameters (Table 1, page 5)

| Circuit | σ_sat (cm²) | LET_TH (MeV·cm²/mg) | W (MeV·cm²/mg) | s |
| --- | --- | --- | --- | --- |
| µSRAM | 3 × 10⁻⁹ | 0.1 | 12 | 1.3 |
| LSRAM | 5 × 10⁻⁹ | 0.1 | 11 | 1.3 |
| FF (All One) | 5 × 10⁻⁹ | 0.5 | 11 | 1.3 |
| FF (All Zero) | 4 × 10⁻⁹ | 0.5 | 10 | 1.3 |
| FF Checkerboard (CB) | 7 × 10⁻⁸ | 0.1 | 32 | 1.6 |
| FF CB-Slow | 3 × 10⁻⁸ | 0.1 | 28 | 1.5 |

Interpretation note from the document: σ for µSRAM and LSRAM is reported per bit; for the DFF "FF" rows the σ is per-bit as well (the appendix raw data is consistent with per-bit cm² values).

### 1.2 SEU cross-section vs. LET, DFF (Appendix raw data, page 6)

Direct numeric table (LET in MeV·cm²/mg, σ in cm²):

| LET | FF0 (all-0) | FF1 (all-1) | FF_CB | FF_CB_slow |
| --- | --- | --- | --- | --- |
| 0.9 | 4.28571E-10 | 2.78571E-10 | 6.14286E-10 | 5.42857E-10 |
| 1.9 | 4.25000E-10 | 5.25000E-10 | 7.75000E-10 | 6.25000E-10 |
| 2.2 | 3.42951E-10 | 4.91563E-10 | 1.10888E-09 | 5.02995E-10 |
| 2.8 | 6.72495E-10 | 5.52407E-10 | 1.39303E-09 | 7.20530E-10 |
| 6.3 | 1.14226E-09 | 8.30737E-10 | 2.54413E-09 | 2.49221E-09 |
| 6.4 | 1.90310E-09 | 1.71132E-09 | 4.48483E-09 | 3.79146E-09 |
| 7.6 | 1.73551E-09 | 1.89552E-09 | 4.33263E-09 | 3.17562E-09 |
| 7.8 | 2.15517E-09 | 1.90965E-09 | 5.23789E-09 | 4.11938E-09 |
| 9.7 | 2.21607E-09 | 2.14681E-09 | 6.50970E-09 | 4.50139E-09 |
| 9.8 | 2.72374E-09 | 2.04280E-09 | 8.31712E-09 | 3.98833E-09 |
| 24.1 | 2.86338E-09 | 4.36324E-09 | 3.02018E-08 | 1.55440E-08 |
| 24.3 | 3.56738E-09 | 3.99207E-09 | 3.08041E-08 | 1.38732E-08 |
| 29.4 | 4.05186E-09 | 5.15166E-09 | 3.99398E-08 | 1.77124E-08 |
| 29.7 | 3.46962E-09 | 6.47037E-09 | 4.38860E-08 | 1.80045E-08 |
| 29.8 | 1.05556E-08 | 1.16667E-08 | 1.00000E-07 | 3.97222E-08 |
| 30.37 | 7.88177E-09 | 1.03448E-08 | 8.02956E-08 | 4.66749E-08 |

### 1.3 SEU cross-section vs. LET, µSRAM and LSRAM (Appendix raw data, page 6)

| LET (MeV·cm²/mg) | µSRAM σ (cm²) | LSRAM σ (cm²) |
| --- | --- | --- |
| 0.9 | 3.65E-10 | 5.44E-10 |
| 1.9 | 4.42E-10 | 7.86E-10 |
| 2.2 | 5.00E-10 | 9.03E-10 |
| 2.8 | 5.22E-10 | 1.13E-09 |
| 6.3 | 6.39E-10 | 1.09E-09 |
| 6.4 | 9.12E-10 | 1.78E-09 |
| 7.6 | 8.58E-10 | 1.83E-09 |
| 7.8 | 9.08E-10 | 2.19E-09 |
| 9.7 | 1.20E-09 | 2.53E-09 |
| 9.8 | 1.30E-09 | 2.65E-09 |
| 24.1 | 2.14E-09 | 3.73E-09 |
| 24.3 | 2.06E-09 | 3.75E-09 |
| 29.4 | 2.67E-09 | 4.76E-09 |
| 29.7 | 2.60E-09 | 4.86E-09 |
| 29.8 | 5.95E-09 | 1.09E-08 |
| 30.37 | 5.36E-09 | 7.94E-09 |

### 1.4 Configuration SEU

> "A total fluence in excess of 5.0 × 10⁻⁷ ions/cm² is accumulated at LET levels up to 82.1 MeV·cm²/mg. No configuration upsets are detected." (page 5)

(Note: the printed "5.0 × 10⁻⁷" is almost certainly a typo for 5.0 × 10⁷ ions/cm² in the source PDF, quoted verbatim.)

### 1.5 SEL data (page 6)

Tested at 100 °C with three I/O bias conditions:

| I/O Bias | LET threshold (MeV·cm²/mg) | Notes |
| --- | --- | --- |
| 3.3 V + 5 % (3.465 V) | SEL detected at 48 | First SEL onset |
| 2.5 V + 5 % (2.625 V) | between 63 and 68.5 | No SEL at 63; SEL at 68.5 |
| 1.8 V + 5 % (1.89 V) | > 82.1 | No SEL up to 82.1 (max tested) |

> "Depending on the decoupling capacitance present on the I/O supply, the SEL may or may not be destructive. In testing, with the amount of capacitance in compliance with the guidance in the PolarFire User Guide, the SEL is non-destructive."

### 1.6 TID data

- **Propagation-delay test (4000-inverter chain):** "negligible up to 300 krad(SiO₂)" (page 2). No tabulated points, Figure 2 is a plot only.
- **Combined retention + TID:** retention 1000 h at 160 °C, TID up to 300 krad(SiO₂). VT distributions for PSONOS/NSONOS shown in Figs. 3–6, these are histograms, not tables of single-cell values.

### 1.7 Orbital error rates from CREME96 (GEO, solar min, 100 mils Al)

| Element | Pattern | Rate (upset/bit/day) |
| --- | --- | --- |
| µSRAM |, | 4.44 × 10⁻⁸ |
| LSRAM |, | 9.21 × 10⁻⁸ |
| DFF | All-zero | 3.36 × 10⁻⁸ |
| DFF | All-one | 4.07 × 10⁻⁸ |
| DFF | Checkerboard | 2.28 × 10⁻⁷ |
| DFF | CB-slow | 1.34 × 10⁻⁷ |

---

## 2. Microsemi PolarFire MPF300, NEPP/NASA-GSFC Independent Test (LBNL 2019)

- **Source URL:** https://nepp.nasa.gov/docs/tasks/041-FPGA/NEPP-TR-2019-Berg-TR-19-045-Microsemi-PolarFire-MPF300T-FCG1152-LBNL-2019Nov18-20205007083.pdf
- **Document:** Berg, Kim, Campola, Pellish, "Microsemi PolarFire FPGA Single Event Effects Test Report," NEPP-TR-2019-Berg, LBNL 88-inch Cyclotron, test date 2019-11-18, report date 2019-12-15.
- **Date fetched:** 2026-04-25.
- **Extraction method:** Direct table read for the LBNL LET table; figures (cross-section plots) NOT extracted as numeric points because the published PDF embeds them as charts only.

### 2.1 LBNL 16 MeV/Nucleon LET table (Table 19, page 37)

| Ion | Energy (MeV/Nucleon) | Effective LET at 0° (MeV·cm²/mg) |
| --- | --- | --- |
| N | 16 | 1.16 |
| O | 16 | 1.54 |
| Ne | 16 | 2.39 |
| Si | 16 | 4.35 |
| Ar | 16 | 7.27 |
| V | 16 | 10.9 |
| Cu | 16 | 16.5 |
| Kr | 16 | 25.0 |
| Xe | 16 | 49.3 |

### 2.2 Test conditions (Table 18, page 37)

- Flux: 1.0 × 10³ to 1.0 × 10⁵ particles/cm²/s
- Fluence: all tests run to 1 × 10⁷ particles/cm² or until destructive/functional event
- Temperature: room
- Vcc = 1.2 V; VIO = 2.5 V

### 2.3 Reported qualitative findings (verbatim, no plot digitization)

- Beam time was limited; only **N, O, Ne** ions used in the first-look campaign.
- Significant SEFI anomaly observed: core current drops from ~2.75 A to ~800 mA (sometimes as low as <100 mA). Recovery requires reset.
  - Onset LET < 1 MeV·cm²/mg.
  - Drops typically ~1.7 ms; one event lasted ~177 s.
- LSRAM SEUs were single-bit only (correctable with SECDED) except when masked by SEFIs.
- Counter-array per-bit DFF cross-section is statistically equivalent to WSR per-bit DFF cross-section at same frequency.

> Numeric SEU/SEFI cross-section data are presented only as figures (Figs. 19, 20, 22, 23, 24, 25). Plot reads would be approximate; we have not digitized them here.

---

## 3. JPL/NASA SRAM Heavy-Ion Test (Scheick, Swift, Guertin)

- **Source URL:** https://nepp.nasa.gov/DocUploads/5ADEBEDA-D7A9-4239-B80209E81315A22C/Sram-00.pdf
- **Document:** Scheick, Swift, Guertin (JPL), "SEU Evaluation of SRAM Memories for Space Applications."
- **Facilities:** Brookhaven National Laboratory (BNL) and Texas A&M (TAM) cyclotrons.
- **Date fetched:** 2026-04-25.
- **Extraction method:** Direct table read.

### 3.1 Devices tested (Table 1)

| Device | Manufacturer | Size | Part Code | Tech |
| --- | --- | --- | --- | --- |
| WMS128K8 | White Electronics | 128k × 8 | WMS128K8-55CI | CMOS |
| MT5C2564 | Austin | 64k × 4 | MT5C2564C-35/CT | CMOS |
| MT5C2568 | Austin | 32k × 8 | MT5C2568C-25/CT | CMOS |

### 3.2 Ions used (Table 2)

| Species | Site | LET (MeV·cm²/mg) | Angles used (°) |
| --- | --- | --- | --- |
| Carbon | BNL | 1.4 | 0, 45, 55, 60 |
| Argon | TAM | 5.4 | 0, 45, 50, 55, 60 |
| Argon | TAM | 10.0 | 0, 30, 45 |
| Argon | TAM | 15.0 | 0, 30, 45 |
| Chlorine | BNL | 11.4 | 0, 45, 55, 60 |
| Nickel | BNL | 26.6 | 0, 45, 55, 60 |
| Iodine | BNL | 59.9 | 0, 45, 55, 60 |

### 3.3 SEU/SEL thresholds (Table 2 of paper, page 3)

| Device | SEU Threshold (10 % of sat.) (MeV·cm²/mg) | SEU Threshold (10⁻⁷ cm⁻² floor) (MeV·cm²/mg) | SEL Threshold (MeV·cm²/mg) |
| --- | --- | --- | --- |
| WMS128K8 | 1.01 | 17 | 37.61806 |
| MT5C2568 | 1.88 | 15 | 37.61806 |
| MT5C2564 | 1.40 | 10 | 58 |

(Vdd = 5 V, room temperature ~25 °C; checkerboard pattern for most runs.)

> Cross-section vs. LET points are presented only as Weibull-fit log–log plots (Figs. 2, 3, 4). No raw σ table is published. Plot digitization would be approximate and we have not done it here.

---

## 4. Hynix 3D NAND vs. Micron 16 nm planar NAND (Bagatin/Gerardin et al., NTRS 2018)

- **Source URL:** https://ntrs.nasa.gov/api/citations/20180000792/downloads/20180000792.pdf
- **Document:** "Heavy Ion and Proton-induced Single Event Upset Characteristics of a 3D NAND Flash Memory" (IEEE TNS 2017/2018, NTRS-hosted preprint).
- **Facilities:** Massachusetts General Hospital (MGH) Burr Center for Space Effects (proton: 100/60/22 MeV), and a heavy-ion facility (BASE, 10 MeV beam).
- **Date fetched:** 2026-04-25.
- **Extraction method:** Direct table read of Table I.

### 4.1 Heavy-ion species used (Table I)

| Ion | LET (MeV·cm²/mg) | Range in Si (µm) | Energy (MeV) |
| --- | --- | --- | --- |
| B  | 0.9 | 306 | 108 |
| Ne | 3.5 | 175 | 216 |
| Si | 6.1 | 142 | 292 |
| Ar | 9.7 | 130 | 400 |
| Cu | 21.2 | 108 | 659 |
| Kr | 30.9 | 886 | 886 |
| Kr (TAMU) | 28.8 | 122 | 953 |
| Au | 85.8 | 90  | 1956 |

### 4.2 Reported qualitative findings (verbatim)

- Hynix 3D NAND SEFI LET threshold: > 0.9 but < 3.9 MeV·cm²/mg.
- Micron planar NAND SEFI LET threshold: < 9.7 MeV·cm²/mg.
- For MLC mode near LET threshold, 3D NAND SEU σ ≈ ~10× that of SLC-fw mode.
- Pattern dependence checked at LET = 9.7 MeV·cm²/mg; 00 pattern showed highest σ (>2× checkerboard).
- MBU comparison done with Ar at 60° base angle, effective LET = 19.5 MeV·cm²/mg.
- The 3D NAND showed lower MBU sensitivity than the planar Micron NAND under those conditions.

> Cross-section curves presented in Figs. 3–9 are plots only; we have not digitized them.

---

## 5. GLOBALFOUNDRIES 22 nm FDSOI SRAM (Casey et al., NEPP/MAPLD 2018)

- **Source URL:** https://nepp.nasa.gov/files/29901/NEPP-CP-2018-Casey-Presentation-SEE-MAPLD-TN66187-NEPPweb-reuse-TN56702.pdf
- **Document:** "A First Look at 22 nm FDSOI SRAM Single-Event Test Results," NEPP-CP-2018-Casey, MAPLD/SEE Symposium, La Jolla, May 2018.
- **Facility:** beam not named in slides (heavy-ion test).
- **Date fetched:** 2026-04-25.
- **Extraction method:** Direct table read of Test Conditions slide (page 6).

### 5.1 Heavy-ion test conditions (page 6)

| Ion species | Energy (MeV) | Nominal LET (MeV·cm²/mg) | Nominal Range (µm) | Tilt angles (°) | Roll angles (°) |
| --- | --- | --- | --- | --- | --- |
| ¹⁴N  | 195  | 1.3  | 379.6 | 0, 30, 45, 60, 62, 66, 71 | 0, 90 |
| ²⁰Ne | 270  | 2.8  | 267.5 | 0, 30, 45, 60 | 0, 90 |
| ⁴⁰Ar | 508  | 8.6  | 180.1 | 0, 30, 45, 60 | 0 |
| ⁶³Cu | 729  | 20.3 | 123.5 | 0, 30, 45 | 0, 90 |
| ¹⁰⁹Ag | 1170 | 43.6 | 107.2 | 0, 30 | 0 |
| ¹²⁹Xe | 1366 | 53.1 | 107.7 | 0, 30 | 0, 90 |

### 5.2 Reported qualitative findings (verbatim)

- 22 nm FDSOI SRAM upset cross-section per bit is "approximately an order of magnitude lower" than 32 nm and 45 nm SRAMs.
- One SEFI observed: at LET = 53.1 MeV·cm²/mg (Xe), all 1s pattern, ~500 cm⁻² s⁻¹ flux. Cleared only by power cycle.
- MBUs accounted for ≤ 0.01 % of total errors on any run; only single- and double-bit errors observed.
- No statistically significant dependence on roll angle or input pattern.

> Per-bit cross-section curves are plots only; we have not digitized them.

---

## 6. Xilinx Kintex-7 (XC7K325T-1FBG900) and Altera Stratix-V (5SGTMC7K3F40C2), NEPP 2015–2016

- **Source URL:** https://ntrs.nasa.gov/api/citations/20160009479/downloads/20160009479.pdf
- **Document:** "Single Event Effects in FPGA Devices 2015–2016," NEPP/NASA-GSFC (Berg, Pellish, Campola).
- **Date fetched:** 2026-04-25.
- **Extraction method:** Direct quote from text. No σ-vs-LET table in this deliverable.

### 6.1 Verbatim numeric findings

- Xilinx Kintex-7 SEL onset (NEPP, with real-time scrubbing): **11.6 MeV·cm²/mg** (vs. 19 MeV·cm²/mg reported by Lee et al. REDW 2014).
- Additional Kintex-7 testing at elevated temperature: susceptibility uncovered as low as **LET = 8.6 MeV·cm²/mg with σ ≈ 1 × 10⁻⁷ cm²/device** (NEPP cannot definitively classify as classical SEL vs. micro-SEL).
- Altera Stratix-V (28 nm bulk CMOS): non-recovering high-current event observed at **80 MeV·cm²/mg, ~105 °C, 60° angle (effective LET = 102 MeV·cm²/mg)** at flux 1 × 10⁴ particles/cm²/s. Re-runs at lower flux (1 × 10³) up to 1 × 10⁶ /cm² fluence completed without anomaly.

> No σ-vs-LET table is published in this slide deck. Use other Xilinx Kintex-7 references (Lee et al. REDW 2014) for numeric Weibull data, that is paywalled IEEE.

---

## 7. Devices investigated but NOT extractable

| Device | Why not | URL attempted |
| --- | --- | --- |
| Xilinx Kintex-7 (full Weibull σ vs. LET) | The NEPP 2015–2016 deliverable explicitly states the full dataset was being analyzed; primary numeric publication is Lee et al. REDW 2014 (paywalled IEEE). | https://ieeexplore.ieee.org/document/7004551 (paywalled) |
| Xilinx Virtex-5QV | Only mentioned as part of catalog; no public σ-vs-LET PDF found in search. |, |
| GPU/SoC space-grade radiation tables | NASA NEPP/RADhome searches returned no public σ-vs-LET PDFs for any space-grade GPU or full SoC under this scope. | https://radhome.gsfc.nasa.gov/ (no public dataset matched) |
| arXiv preprints with full radiation σ-vs-LET tables | Search returned simulation/ML preprints (arXiv:2404.01757, arXiv:2402.17489) but no primary beam-test data tables in PDF text. | https://arxiv.org/abs/2404.01757 (BNN simulation, not beam data) |
| Microchip RTG4 (referenced by RT PolarFire as predecessor) | Microchip publishes only summary in commercial marketing; the IEEE TNS RTG4 paper cited in the RT PolarFire report is paywalled. | https://ieeexplore.ieee.org/document/7370820 (paywalled) |

---

## 8. Sources actually fetched (URLs verified, PDF downloaded, text extracted)

1. https://ww1.microchip.com/downloads/aemdocuments/documents/fpga/ProductDocuments/SupportingCollateral/rt_polarfire_radiation_test_report_2020-02-04.pdf, Microchip RT PolarFire radiation test report (2020-02-04). MD5/size verified, 290,945 bytes, 6 pages, version 1.6.
2. https://nepp.nasa.gov/docs/tasks/041-FPGA/NEPP-TR-2019-Berg-TR-19-045-Microsemi-PolarFire-MPF300T-FCG1152-LBNL-2019Nov18-20205007083.pdf, NEPP/NASA-GSFC PolarFire LBNL 2019 SEE report (1,907,364 bytes, 48 pages).
3. https://nepp.nasa.gov/DocUploads/5ADEBEDA-D7A9-4239-B80209E81315A22C/Sram-00.pdf, JPL SRAM SEU evaluation (61,115 bytes, 3 pages).
4. https://nepp.nasa.gov/files/29901/NEPP-CP-2018-Casey-Presentation-SEE-MAPLD-TN66187-NEPPweb-reuse-TN56702.pdf, 22 nm FDSOI SRAM SEE first look, MAPLD 2018 (832,548 bytes, 17 pages).
5. https://ntrs.nasa.gov/api/citations/20180000792/downloads/20180000792.pdf, Hynix 3D NAND SEU paper (1,184,015 bytes, 8 pages).
6. https://ntrs.nasa.gov/api/citations/20160009479/downloads/20160009479.pdf, NEPP "Single Event Effects in FPGA Devices 2015–2016" (789,877 bytes, 31 pages).

---

## Summary

We verified **6 device families with publicly fetched primary-source beam-test data**:

1. **Microchip RT PolarFire (28 nm SONOS-flash FPGA)**, the gold-standard dataset here. The source PDF includes a full appendix with 16-row σ-vs-LET tables for µSRAM, LSRAM, and four DFF data patterns (LET 0.9 → 30.4 MeV·cm²/mg), plus full Weibull parameters and explicit SEL bracketing.
2. **Microsemi MPF300 / NEPP independent (LBNL 2019)**, facility/ion table is solid; numeric σ values are plot-only and were not digitized.
3. **JPL White Electronics WMS128K8 + Austin MT5C2564 + MT5C2568 SRAMs**, direct read of LET-threshold and SEL-threshold tables.
4. **Hynix 3D NAND + Micron planar NAND**, heavy-ion species/LET/range/energy table.
5. **GlobalFoundries 22 nm FDSOI SRAM**, full ion conditions table (6 species, energies, LETs, tilt/roll).
6. **Xilinx Kintex-7 + Altera Stratix-V**, verbatim quoted SEL onset LETs and one effective-LET non-recovery point.

**For maintainers**: section 1 (RT PolarFire) is directly machine-ingestible, 32 σ-vs-LET data points across DFF and SRAM, plus closed-form Weibull. Sections 3 and 5 give clean threshold/ion-condition tables. Sections 2, 4, 6 give validated facility/ion conditions but require a separate plot-digitization pass before numeric σ values can be ingested. Anything we did not pull is either paywalled (IEEE TNS, REDW) or simply absent from public NEPP/NTRS/arXiv search hits.
