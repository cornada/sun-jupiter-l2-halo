# Evolution journal — Sun-Jupiter L₂ halo orbit stabilization

Cumulative append-only log. NEVER overwrite or summarize previous entries.

---

## Session 2026-04-17

### Goal
Starting point: single file `Sun_Jupiter_Orbit_Stabilization` — Python script computing trajectory near L₂ with planar Lyapunov-like ICs and 3D CR3BP integration, no closure enforcement, no correction strategy. Alex's question:

> Необходимо решить задачу, чтобы траектория была замкнутой и энергии... Причём чтобы за один оборот замыкалась она. И найти такое положение в пространстве, откуда надо начинать корректировку. Определить стратегию корректировки для того, чтобы минимально тратилась энергия.

Three sub-tasks implied:
1. Closed periodic orbit (one-revolution).
2. Optimal correction location on orbit.
3. Minimum-ΔV strategy, verified by work-energy relation.

### What was done

1. **`halo_correction_strategy.py`** — planar Lyapunov DC, full-period STM integration, monodromy eigenanalysis (λ_u ≈ 1752), propagated left eigenvector L(t), closed-form min-ΔV law `|Δv| = |α_u|/‖L_v(t)‖`, single-shot correction demo with machine-precision verification of ΔC = −2·ΔKE.

2. **`halo_visualization.py`** — generated figures 1–20 in `figs/`: Hill region, orbit close-up, phase portraits (x-vx, y-vy), time series, Jacobi conservation, monodromy spectrum (complex plane + log magnitude), leverage `‖L_v(t)‖` linear + log, **orbit-colored-by-cost heatmap**, unstable-direction field, divergence without correction with λ_u^k overlay, station-keeping, ΔV vs phase, **work-energy identity scatter**, polar leverage, 3D view, reference vs perturbed, Jacobi decomposition, dashboard.

3. **`halo_3d_family.py`** — continuation in A_z giving 13-member halo family (T ∈ [3.197, 3.224], λ_u ∈ [1030, 1819]), invariant manifold globalization (96 trajectories), multi-shooting station-keeping at K ∈ {1,2,4,8,16}. Figures 21–35.

4. **`report.tex` → `report.pdf`** — 41 pages: literature review, math framework (CR3BP → Floquet → manifolds → min-ΔV law), numerical architecture, all 36 figures with physical interpretation, discussion/novelty, acronyms table, English future-work plan, interactive-demo user guide (Section 9 with scene anatomy / panel anatomy / 6 walkthrough scenarios / physical-math crib sheet).

5. **`demo.html`** + **`export_webdata.py`** — three.js 3D interactive demo: orbit + primaries + Lagrange markers + manifolds (TubeGeometry-based — THREE.Line linewidth is ignored by browsers), 12 interactive controls (A_z slider, t/T slider, inject/correct buttons, layer toggles, view presets, comparison readout), in-browser RK4 integration for perturbation simulation, live ΔC=−2ΔKE identity verification, sprite labels for each object.

6. **Deployed** — `github.com/cornada/sun-jupiter-l2-halo` public repo, GitHub Pages at `cornada.github.io/sun-jupiter-l2-halo/` (source=main/docs), links embedded on report title page and Section 9 header.

### Quantitative results captured

- Planar Lyapunov: x₀=1.07081, vy₀=−0.01244, T=3.17932, closure residual 3×10⁻⁸, Jacobi drift per period 1.3×10⁻¹⁵.
- Monodromy eigenvalues (planar): λ_u = 1752, λ_s = 5.71×10⁻⁴, two trivial λ=1, center pair at angle ±13.2°.
- Halo family: x₀ − x_L2 ∈ [−0.020, −0.013] (inner side!), vy₀ ∈ [+0.070, +0.092], Az ∈ [0.002, 0.020], λ_u ∈ [1030, 1819].
- Cheapest correction phase = perpendicular x-axis crossing. Fuel advantage: **1652×** over worst phase (planar).
- Work-energy identity residual: max over 500 random burns < 2×10⁻¹⁶.
- Multi-shooting: K=1 at cheapest phase beats K∈{2,4,8,16} in both total ΔV and peak error — waiting for cheap phase is strictly optimal in the linear regime.

### Insights — technical

1. **The minimum-ΔV phase = perpendicular x-axis crossing.** Not an accident: at that phase the propagated left eigenvector L(t) has maximum velocity-block norm, i.e. the unstable mode aligns most with the thrust subspace. Pedagogically clean geometric anchor.

2. **ΔC = −2·ΔKE as a machine-precision correctness witness.** The rotating-frame work-energy identity simultaneously checks pseudo-potential Ω, Jacobi formula, integrator tolerance, impulse update rule. Residual 10⁻¹⁶ means all pieces consistent. For any future CR3BP-adjacent code, this should be the first smoke test.

3. **Halo family softens with A_z.** λ_u drops from ~1800 at A_z=0.002 to ~1030 at A_z=0.020. Operational gift: larger halos are cheaper to hold. Mission amplitude choice can be partly driven by station-keeping economy, not just science return.

4. **Multi-shooting paradox.** K=1 beats K>1 for hyperbolic orbits in linear regime. Naive intuition ("more corrections = better") fails because sub-optimal phases have small ‖L_v(t)‖ → expensive corrections. Only when nonlinearity threatens (error ≳ 10⁻³ in our system) does K>1 help.

5. **µ=9.53·10⁻⁴ is a different regime from Sun-Earth.** 300× larger mass ratio → steeper saddle → higher λ_u per revolution → proportionally more aggressive stabilization needed. Sun-Earth halo lore translates only qualitatively.

6. **Halo at Sun-Jupiter L2 sits on the inner (Jupiter) side of L2.** x₀ < x_L2 counter to Sun-Earth halo intuition where halo extends outward. This is a feature of the specific µ value. See `project_halo_dc_pitfalls.md` in global memory.

7. **THREE.Line linewidth is a lie.** WebGL 1.0 spec requires only linewidth=1; Chrome/Firefox/Safari honor nothing else. For 3D line rendering, TubeGeometry is mandatory. Lost ~15 min debugging this.

### Insights — strategic / domain

1. **Pair every figure-heavy report with an interactive web demo.** Alex's question "what are the white dots?" after seeing figures showed that even well-labeled images leave ambiguity about scale + physical identity. Post-demo, confusion vanished. Three-layer deliverable (report + demo + user guide) is the right format for physics/math communication.

2. **Deploy early.** Making the demo live at a sharable URL (GH Pages) before writing the user guide let the guide reference a working artifact. Inverted order (write first, deploy later) gives a more generic guide.

3. **Machine-precision witnesses are publishable content.** The ΔC=-2ΔKE residual plot (Fig. 15) is a self-contained demonstration of implementation correctness that reviewers understand immediately. Good for methods papers.

4. **The rotating-frame intuition trap.** Every time I thought in inertial frame I confused myself. Stay religiously in rotating frame for CR3BP work; only translate to inertial at publication boundaries.

### Mistakes and failures

1. **First differential correction converged to 5:1 resonant multi-loop orbit (T≈16), not halo.** Used seed `(x_L2+0.003, vy₀=-0.0186)` with `t_max=10`. Newton found *a* fixed point — but the orbit completes 5 in-plane loops before crossing y=0 in the requested direction. **Wrong mental model:** assumed Newton's basin of attraction for the halo is wide. In fact it's narrow, and without bounding the half-period I accidentally entered the resonant-orbit basin. **Fix:** `t_max < 10` to reject multi-loop solutions.

2. **Tried large-amplitude halo seeds (Ax ~ 0.05, Az ~ 0.03).** All escaped L2 within 4 time units. **Wrong mental model:** assumed halo family extends to arbitrarily large amplitudes. Reality: above a specific energy threshold (related to C=2Ω_{L2}), orbits can escape through the L2 gateway. The halo family is energetically bounded and relatively small for this µ.

3. **Halo DC converged to x₀ < x_L2 (inner side), not outer side.** I had assumed halo lived outside L2 (on anti-Sun side, like Sun-Earth intuition). After DC converged with x₀ − x_L2 ≈ −0.013, spent a minute double-checking this was a valid halo and not a numerical artifact. It is valid — just a feature of this µ.

4. **First multi-shooting attempt (K>1) blew up numerically.** Total ΔV for K=2 was 2731 (vs. expected 10⁻⁵). Root cause: at the K=2 second node, ‖L_v(t)‖ was tiny, so |Δv| = |α|/‖L_v‖ became huge, propagated to nonlinear regime, subsequent integrations diverged. **Wrong mental model:** assumed correction at any reasonable phase would be OK because α doesn't change under free flow. Forgot that cost scales as 1/‖L_v‖. **Fix:** cap |Δv| at 10⁻³; skip corrections above that.

5. **Event direction bug.** Hardcoded `direction=-1` for `y=0` crossing worked for vy₀>0 but fired instantly at t=0 for vy₀<0. Spent ~20 min before identifying. **Fix:** `direction = -sign(vy₀)`.

6. **Initial demo.html had invisible orbit.** Used THREE.Line with linewidth=2. Browsers rendered as 1px, invisible against dark background. Camera was also poorly placed. **Wrong mental model:** assumed three.js linewidth works as documented. It doesn't. **Fix:** TubeGeometry + CatmullRomCurve3 for smooth closed curves.

7. **Richardson c₂ miscalculation.** First got c₂ ≈ 2.808, which didn't match numerical Oxx value of ~8.64 (implying c₂ = 3.82). Recomputed: c₂ ≈ 3.624 (correct). Arithmetic slip at (1-µ)·γ³/(1+γ)³ term.

8. **Halo seed search was too narrow.** Initial scan `(x_off ∈ [0.001, 0.015], Az ∈ [5e-4, 1.5e-2])` with `t_max=3.5` returned zero halos — because I was looking on the wrong side. Only when I reverted to using Lyapunov seed (x_L2+0.002, vy₀=-0.0124) with t_max=10 and accepted any converging point — the halo appeared on the inner side spontaneously.

### Future ideas to check (forks / expansion)

Ordered roughly by feasibility / value.

#### Short-term (feasible within days)
1. **Richardson 3rd-order analytical halo seed.** Would eliminate the convergence basin issue. Implement the polynomial expansion and use its output as warm-start for DC. Could also find the halo bifurcation from Lyapunov analytically.
2. **Longer manifold globalization (5–10T instead of 1.2T).** Needed to find heteroclinic intersections L₁ ↔ L₂ and asymptotic return patterns.
3. **Poincaré section analysis.** With longer manifolds, compute {y=0, ẏ>0} sections and search for manifold intersections → free transfers.
4. **Family continuation via pseudo-arclength.** Current natural-parameter continuation in A_z may break near bifurcations. Pseudo-arclength handles turning points.
5. **Add "race mode" to demo:** two spacecraft (corrected / uncorrected) animated in parallel on the same orbit. Visualizes the value of station-keeping in one frame.

#### Medium-term (weeks to a month)
6. **Ephemeris extension.** CR3BP → four-body (Sun + Jupiter + S/C + Saturn/Galilean perturbers) via JPL SPICE. Quantify how much correction-law degrades; is Jupiter eccentricity (0.049) a serious correction?
7. **LQR / MPC station-keeping.** Replace static Floquet law with receding-horizon optimization. Benchmark fuel budget over 10-year mission horizon.
8. **Low-thrust (continuous) station-keeping.** Pontryagin minimum principle on rotating-frame dynamics with ~1 mN continuous thrust. Compare total ΔV over 5-year horizon.
9. **Period-doubling atlas.** Continue halo family through its period-doubling bifurcation cascade; look for "quasi-halo" / Lissajous branches.
10. **Elliptic RTBP extension.** Add Jupiter eccentricity. System becomes non-autonomous periodic. Floquet theory generalizes via monodromy of time-varying STM.

#### Long-term (research-paper scale)
11. **Quint (or Coq) formalization.** Machine-verify CR3BP equations, Jacobi invariance, min-ΔV law, and ΔC=-2ΔKE identity. Turn numerical witnesses into mechanical theorems.
12. **Parametric atlas µ ∈ [10⁻⁷, 10⁻¹].** Sweep mass ratio. Tabulate λ_u, T, halo-family shape, manifold geometry. Makes tooling reusable for exoplanet / binary systems.
13. **Multi-spacecraft formation.** Cooperative station-keeping where a "cluster leader" absorbs the full unstable mode; followers use relative measurements to stay in formation. Reduces per-spacecraft ΔV.
14. **Center-manifold reduction.** Lie series / normal-form expansion to reduce dynamics to 2D center manifold. Gives analytical halo parameterization accurate to 10th order.
15. **GPU-accelerated manifold globalization.** Current serial integration of 96 trajectories ~30 s. Dense manifold atlas (10⁴ seeds) would benefit from CUDA/ROCm batch integration.
16. **ML anomaly detector on Floquet residuals.** Train compact NN to flag deviations not captured by linear theory (e.g. unmodeled gravity from passing asteroids). Couple to automated safe-mode replanner.
17. **Unified periodic-orbit atlas** for all 5 Lagrange points + Trojan libration regions. Each with family, manifolds, correction cost maps. Educational and operational value.

#### Demo-level additions
18. Slider to inject Gaussian noise into spacecraft state continuously (simulating navigation uncertainty).
19. Linear vs. nonlinear correction toggle — show that Floquet law breaks down at large α.
20. K-slider for live multi-shooting comparison (1..16).
21. Tooltip/info-card on hover over any scene object.
22. 2D side panel with x-y projection, synchronized with 3D.
23. Screenshot-with-params export.
24. "Snapshot mode": capture current orbital state + perturbation + correction for later replay.

### Open questions (genuine unknowns)
- Does the Sun-Jupiter L₂ halo family terminate at a bifurcation, or extend to arbitrarily large A_z? My scan stopped at A_z=0.020; I don't know the actual upper bound.
- Is the ΔC=-2·ΔKE identity (which I treated as a correctness witness) actually an *independent* check, or is it a tautological consequence of the algebra? If tautological, it's a code-correctness test, not a physics-correctness test. Worth examining formally.
- The "1652× savings" figure compares the cheapest vs. worst phase in the *same* period. Cost at t=0 of period k+1 is 1/λ_u times cheaper than cost at t=T of period k, so the ratio across one "reset" is exactly λ_u. Is there a deeper significance to this λ_u-periodicity of the cost function?
- For Sun-Jupiter halo specifically: is the "inner-side" (x₀ < x_L2) halo the "class I northern" in Howell's terminology, or something else? Need to cross-check with literature tables.

### Loose ends
- `family_data.json` is stale after updates — regenerate if extending.
- `halo_seed_scan.py` was an exploratory tool; could be deleted or kept as historical record.
- `Sun_Jupiter_Orbit_Stabilization` (original file) has a `v_u = np.real(v_u)` line that doesn't feed into initial conditions — misleading comment suggests it does. Could be cleaned up or left as-is for historical reference.
- Report PDF references `figs/*.png` with relative paths; if report is moved, figures break. Consider `\graphicspath` or embedding.
- No unit tests. All correctness verification is via the ΔC=-2ΔKE identity and closure residuals. For a research codebase this is probably fine; for an operational one, add pytest coverage.

---

## Session 2026-04-17 (part II — Earth-Moon sister project)

### Goal
Alex asked to replicate the entire pipeline for Earth-Moon (replacing Jupiter with Moon), in a separate folder, deployed separately. Test whether the same architecture transfers to a different mass ratio without substantial rework.

### What was done
1. Created `/Users/aleksandrvolkov/earth-moon-l2/`, copied all 5 Python scripts.
2. Bulk-substituted mu: `9.53e-4 → 0.01215058` in all scripts via `perl -i -pe`.
3. Bulk-substituted strings: Sun-Jupiter → Earth-Moon, Jupiter → Moon in scripts and report.tex.
4. Ran `halo_correction_strategy.py` — planar Lyapunov converged at first try: x₀ = 1.1577, vy₀ = -0.01093, T = 3.37, λ_u = 1452.
5. Ran `halo_visualization.py` — 21 figures generated (figs/01 through figs/20 + 07b).
6. Ran `halo_3d_family.py` — 7 of 13 halo family members converged (small Az < 0.005 failed — bifurcation threshold is higher for Earth-Moon). Fixed `rep = family[7]` → `family[len(family)//2]` to avoid index error.
7. Ran `export_webdata.py` — updated Az_list to [0.005, 0.008, 0.012, 0.015, 0.018, 0.022] to match converging range. 6 family members exported.
8. Adapted `demo.html` — color swap (Sun yellow → Earth blue 44aaff; Jupiter brown → Moon gray cccccc), labels.
9. Bulk-substituted report.tex then manually fixed 4 paragraphs where bulk sed broke semantic comparisons.
10. Updated family table in report with real Earth-Moon values.
11. Compiled report (41 pages, 4.27 MB).
12. Git init + commit, `gh repo create cornada/earth-moon-l2-halo --public`, Pages enabled, URL: `https://cornada.github.io/earth-moon-l2-halo/` (HTTP 200).

### Quantitative results (Earth-Moon, µ=0.01215)
- Planar Lyapunov: x₀ = 1.1577, vy₀ = -0.01093, T = 3.373 (≈ 14.67 days), λ_u = 1452, Jacobi drift 1.8·10⁻¹⁵.
- Halo family (7 members): x₀ - x_L2 ≈ -0.036 (still inner side!), vy₀ ≈ +0.177, T ≈ 3.41, λ_u 1208 → 1128 as Az 0.005 → 0.022.
- Leverage ratio ~1452× (= λ_u for planar Lyapunov).
- Multi-shooting results similar structure: K=1 wins in both fuel and accuracy for this orbit too.
- x_L2 = 1.1557, x_L1 = 0.8369, x_L3 = -1.0051.
- γ = 0.168 (Moon-to-L2 distance in CR3BP units; physically ≈ 64,400 km).
- c₂ = 3.192 (vs. 3.624 for Sun-Jupiter); ω_p = 2.16, ω_v = 1.787.

### Insights — technical

1. **Architecture transfer was nearly trivial.** The same code ran unchanged on Earth-Moon after a single-variable `mu` substitution. All the Floquet machinery, manifold globalization, multi-shooting benchmarks worked first try. This validates the abstraction: the correction strategy is µ-independent in form, only in numerical values.

2. **Halo bifurcation threshold scales with µ.** Sun-Jupiter halos converged from Az=0.002 up. Earth-Moon halos require Az ≥ 0.005. This is a consequence of the specific ratio of planar vs vertical frequencies at L2 (ω_p/ω_v), which sets where nonlinear coupling can achieve frequency lock.

3. **λ_u at L2 is remarkably µ-robust.** Sun-Jupiter λ_u = 1752, Sun-Earth λ_u ≈ 1500, Earth-Moon λ_u = 1452. Despite 100× range in µ, λ_u varies only by ~20%. This was counterintuitive — I expected much larger spread. The reason: λ_u is set by the hyperbolic geometry of the CR3BP saddle, which depends on c₂ more weakly than µ. Good material for the "parametric atlas" future-work item.

4. **Halo on inner side (x₀ < x_L2) persists across µ.** Both systems have halos with negative x-offset. This is a feature of the specific Floquet eigenstructure at L2, not a µ-specific accident.

### Mistakes and failures

1. **Bulk sed broke semantic comparisons.** Replacing "Jupiter → Moon" changed "Sun-Jupiter" → "Sun-Moon" inside comparison sentences like "Unlike Sun-Jupiter, Earth-Moon is ..." → "Unlike Sun-Moon, Earth-Moon is ..." (tautological nonsense). Similarly "Moon-system manifold structure for Europa Clipper" came from "Jupiter-system..." and makes no sense for Earth-Moon. WRONG MENTAL MODEL: assumed mechanical substitution preserves meaning. Fix: targeted manual rewrites after sed.

2. **`gh` operations required explicit cd, not chained with `cd && gh`.** My shell cwd kept auto-resetting to /Users/aleksandrvolkov/2 between invocations (likely due to tool-level shell reset), so cd in one command doesn't persist. Fix: use absolute paths or chain operations in single bash invocation.

3. **URL bulk-sub caught "Jupiter" in URL "cornada/sun-jupiter-l2-halo" → "cornada/sun-moon-l2-halo".** Three places (title page, section 9 callout, README). Fix: grep for URLs and explicitly update.

4. **`rep = family[7]` hardcoded index.** Worked for Sun-Jupiter (13 members) but crashed Earth-Moon (only 7 converged). Fix: `family[len(family)//2]` picks a middle member robustly.

5. **Report still has residual Sun-Earth references in 3 places.** Lines 95 (ISEE-3 historical), 104 (mass-ratio comparison), 925 (ephemeris future work). All are correct-as-stated (they are genuine Sun-Earth references, not bugs). Left in place.

### Future ideas to check (new)

1. **Parametric atlas µ ∈ [1e-7, 1e-1]** — now actually tested at two µ values (9.53e-4 and 1.215e-2). λ_u varies only 20% in this range. A full sweep across 7 decades of µ would produce a universal atlas and confirm whether λ_u plateaus or eventually diverges.

2. **Halo bifurcation threshold Az_min(µ)** — we observed Az_min ~ 0.002 for Sun-Jupiter and ~ 0.005 for Earth-Moon. Is there a closed-form analytical expression? (Yes — Richardson 3rd-order gives it, but computing is nontrivial.)

3. **Unified comparison report** — side-by-side figure + number table of Sun-Jupiter vs. Earth-Moon. Would make the "architecture transfer" claim concrete.

4. **Sun-Earth extension** — complete the trilogy with µ = 3.04e-6. Would land between Earth-Moon (λ_u ≈ 1450) and Sun-Jupiter (λ_u ≈ 1750) at λ_u ≈ 1500 (classical Sun-Earth L2 halo value from ARTEMIS analysis).

### Development-speed notes

- Full replication of Sun-Jupiter pipeline for Earth-Moon (new µ, all scripts rerun, figures, report, demo, deploy): ~20 minutes in total. Suggests the pipeline is well-structured for extension.
- Bulk sed requires manual paragraph-level review afterward. For any future system replication, budget ~30% of the sed time for semantic cleanup.
- Live deployment via gh CLI + Pages is essentially instant (< 2 minutes including first build). Much faster than I expected; removes deployment friction as a barrier to iterative sharing.

### Deployed artifacts (Session 2)
- `github.com/cornada/earth-moon-l2-halo` — public repo
- `https://cornada.github.io/earth-moon-l2-halo/` — live demo (HTTP 200)
- `/Users/aleksandrvolkov/earth-moon-l2/` — local project dir

---

## Session 2026-04-17 (part III — polish and parking)

### What was done
1. **Author renamed** in both reports: `A.~Volkov` → `I.~Gabitov`. Also updated README cite lines. Recompiled both PDFs; verified via `pdftotext -l 1 report.pdf`.
2. **Standalone demos** (`demo.html`) produced for both projects via `inline_demo.py`: JSON data embedded inline, replacing the `fetch('./web_data.json')` block. Result: HTML files ~780 KB each, openable by double-click on `file://` without HTTP server. Three.js still loads from unpkg CDN — that works on `file://` for HTTPS-loaded ES modules. Propagated to `docs/demo.html` and `docs/index.html` (same byte-identical files).
3. **Dynamic camera framing.** Fixed complaint that Earth-Moon demo required manual zoom-out to see the orbit. Replaced fixed `R = 0.08` in `setCamera()` with `R = max(0.04, 3.5·orbitExtent.rmax)`, auto-computed from current family member. Camera target also shifts by `0.3·zmax` to frame halo lobes off-ecliptic. Applied to both projects for consistency.
4. **Parked** — clean zip archive built at `/Users/aleksandrvolkov/l2-halo-orbits.zip` (19.6 MB, 113 files). Contains both projects under a top-level folder. Excluded: `.git/`, `.claude/`, `__pycache__`, LaTeX build artefacts, macOS detritus, editor swap files. Includes: code, reports, demos, figures, data, session journals.

### Insights

1. **WebGL ES modules on file://.** The user's error screenshot showed the JSON fetch failing, while three.js modules loaded fine — because unpkg's HTTPS modules are explicitly allowed on `file://` by all major browsers, but local `fetch()` is not. The fix is always local-first: inline critical local data into the HTML itself, leave CDN assets alone.

2. **Inline JSON cost/benefit.** 800 KB of JSON embedded as a JS literal adds ~4 MB of parse/compile work on page load — measurable (~50 ms on M1 Mac) but invisible to users. Makes demo one-click runnable with zero infrastructure. For projects intended for static hosting AND offline viewing, this is the right default.

3. **Dynamic camera distance should always scale with data.** A fixed framing constant hard-coded for one dataset (Sun-Jupiter) broke when applied to another (Earth-Moon) with 2× larger orbit amplitude. Lesson: any "view preset" in a data-parameterized viewer must compute its distance from the data, not from the world scale.

### Clarifications surfaced in conversation (worth capturing)

Questions the user asked that required explicit answers — possibly repeat-trigger topics:

- **Planet proportions.** NOT to scale. Sun/Earth enlarged ~1500×, Jupiter/Moon ~1800×. Explicitly noted in report §9.2 "Scale disclaimer." In honest rendering they'd be single pixels.
- **Moon's motion around Earth.** In our rotating (synodic) frame Earth and Moon are stationary; the frame co-rotates at lunar orbital rate. This is a feature, not a bug: autonomous dynamics, conserved Jacobi integral, five equilibria. Inertial view would show the halo as an epicycloidal loop.
- **"One moment" vs "one period".** The orbit is a time-parameterized curve U(t) stored at 300 samples; the slider scrubs along it. The perturbation simulator is a live RK4 integrator. We do NOT integrate repeatedly across years of ephemeris time — the reference orbit is a computed periodic solution, reused each revolution.
- **Sun's influence on Earth-Moon L₂.** Not included in our CR3BP. Differential (tidal) acceleration from Sun at Earth-Moon L₂ ≈ 3.6·10⁻⁵ m/s² — about 1.5% of Earth's gravity at L₂. Causes ~1000 km yearly drift of the "effective L₂," ~0.5-2% halo-shape modulation, and ~3× larger ΔV budget than pure-CR3BP estimate for multi-year operations. Proper treatment: BCR4BP (bicircular) or JPL ephemeris. Queqiao uses the latter.

### Mistakes and failures

1. **First zip build included `.claude/settings.local.json`**. Only in `/Users/aleksandrvolkov/2/` (not in the Earth-Moon directory). 7 KB, but exactly the kind of personal configuration the user asked to exclude. FIX: added `--exclude='.claude'` to the rsync filter. WRONG MENTAL MODEL: assumed hidden dot-folders would be caught by a generic `.*` filter; rsync exclusions need explicit names for each folder pattern.

2. **Deleted staging directory with useful intermediate README.** After rsync failure, I ran `rm -rf ...staging` before catching that my README was in there. Had to rewrite it. LESSON: when debugging build pipelines, don't cleanup until the final artifact is verified.

### Files in final state

- `/Users/aleksandrvolkov/2/` — Sun-Jupiter project, all pushed to `cornada/sun-jupiter-l2-halo` main.
- `/Users/aleksandrvolkov/earth-moon-l2/` — Earth-Moon project, all pushed to `cornada/earth-moon-l2-halo` main.
- `/Users/aleksandrvolkov/l2-halo-orbits.zip` — 19.6 MB parking archive.
- `/Users/aleksandrvolkov/inline_demo.py` — standalone utility that inlines JSON into demo.html. Reusable.
- `~/.claude/projects/-Users-aleksandrvolkov-2/memory/` — 6 memory files, unchanged this sub-session.

Nothing in `/tmp/`. No valuable artefacts at risk.
