# Sun–Jupiter L₂ Halo Orbit: Stabilization & Correction Strategy

Numerical pipeline for constructing periodic halo orbits near the Sun–Jupiter L₂ libration point, analyzing their stability via Floquet theory, globalizing invariant manifolds, and benchmarking minimum-ΔV station-keeping strategies in the Circular Restricted Three-Body Problem (CR3BP).

## 🚀 Interactive demo

**→ [Live 3D demo on GitHub Pages](https://cornada.github.io/sun-jupiter-l2-halo/)**

Explore the halo family, invariant manifolds, and correction strategy in your browser.
Full user guide in the accompanying [report.pdf](report.pdf), Section 9.

## 📘 Report

See [`report.pdf`](report.pdf) — 41-page technical write-up:
- Literature review (Szebehely, Farquhar, Richardson, Howell, Koon–Lo–Marsden–Ross)
- Mathematical framework (CR3BP, STM, Floquet, manifolds)
- Numerical architecture
- Results: 36 figures covering planar Lyapunov, halo family, manifolds, station-keeping
- Interactive-demo user guide
- Acronyms table
- Future-work plan

## 🗂 Repository layout

| Path | Purpose |
|------|---------|
| `docs/` | GitHub Pages site (demo + data) |
| `figs/` | All 36 figures (PNG) |
| `report.tex` / `report.pdf` | Technical report |
| `halo_correction_strategy.py` | Planar Lyapunov + Floquet + single-shot correction |
| `halo_visualization.py` | Figures 1–20 |
| `halo_3d_family.py` | 3D halo family + manifolds + multi-shooting, Figures 21–35 |
| `export_webdata.py` | Generates `docs/web_data.json` for the demo |
| `Sun_Jupiter_Orbit_Stabilization` | Original exploratory Python script |

## ⚙️ Reproducing the results

```bash
pip install numpy scipy matplotlib
python3 halo_correction_strategy.py     # quick demo (planar Lyapunov)
python3 halo_visualization.py           # Figures 1–20
python3 halo_3d_family.py               # Figures 21–35
python3 export_webdata.py               # regenerate web data
pdflatex report.tex && pdflatex report.tex   # compile report
```

## 🧮 Key numerical results

- **Reference Lyapunov orbit:** x₀ = 1.07081, vy₀ = −0.01244, T = 3.17932, residual 3·10⁻⁸
- **Jacobi drift per period:** 1.3·10⁻¹⁵ (machine precision)
- **Monodromy:** λ_u ≈ 1752 (planar Lyapunov), ≈ 1316 (representative halo)
- **Work–energy identity residual:** < 2·10⁻¹⁶ across 500 random impulsive burns
- **Cheapest correction phase advantage:** 1652× over worst phase

## 📜 License

CC BY 4.0. Cite as: Volkov A., "Stabilization of periodic orbits near the Sun–Jupiter L₂ libration point," 2026.
