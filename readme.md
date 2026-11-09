# AGOP Decision Engine

> A computable decision procedure for multi-criteria selection under uncertainty.
>
> NOTICE & FAIR WARNING: THIS SOFTWARE ASSUMES THAT THE GARBAGE IN & GARBAGE OUT -PROBLEM HAS BEEN SOLVED BY THE USER! IT WORKS PROPERLY, IF YOUR DATA IS PERFECT, OTHERWISE IT WILL BE WRONG! THEREFORE IT WILL ALWAYS BE WRONG, BUT ON THE OTHER HAND, THE RESULTS MAY BE USEFUL! 

AGOP (Attributed-Graph Operator Pipeline) ranks a finite set of alternatives
under multiple decision axes, a strategic counter-party, structural
compatibility relations, and uncertain outcomes. It composes graph-theoretic,
game-theoretic, systems-theoretic, topological, deductive, stochastic, and
probabilistic operators in a typed DAG with two nested fixed-point subroutines.
The output is a deterministic, admissibility-checked ranking.

This package provides:

- a Python reference implementation of the algorithm (`agop_core.py`);
- a REST API (`agop_api.py`, FastAPI, OpenAPI auto-generated);
- an interactive browser UI (`agop_ui.html`);
- a publication-quality PNG chart generator (`agop_charts.py`);
- a convenience launcher (`run.py`).

# The updated files. Design By Contract (Alternative strategy):

The updated agop_core.py and agop_api.py refactor AGOP from a pipeline that accepted capacity as a caller-supplied hyperparameter into one that derives it from the mathematical structure of the problem itself. agop_core.py introduces typed primitives (Alternative, GamePayoff, AdmissibilityState), removes K from the Theta vector, computes the cardinality of unique Weisfeiler–Lehman structural classes and the Shannon entropy of both Nash equilibrium strategies (σ_DM, σ_CP) at their respective phases, and derives K = max(1, ⌊n_wl_classes / (1 + H(σ_DM) · H(σ_CP))⌋) inside the pipeline before injecting it into the capacity-constrained population equilibrium; the verifies_theorem_5 decorator then enforces Theorem 2 (convergence depth exactly one) and Theorem 5 (every returned alternative has strictly positive LCB, adoption status one, positive composite Φ, and membership in D*) on every call, raising AssertionError before any partial result can escape. agop_api.py reflects the same reduction at the interface boundary: ThetaSpec carries model_config = {"extra": "forbid"} so any request containing K is rejected with a 422 contract violation, the response payload now surfaces K_derived, n_wl_classes, and both entropy components, and the reference instance no longer mentions K anywhere. The net effect is that K has been fully endogenised — it is no longer a value a human can supply, tune, or accidentally inject; it is a deterministic function of the graph's structural diversity and the strategic instability of the two-player game, and it changes only when those primitives change.

By linking operational scope entirely to structural diversity and game tension, this design accepts the deliberate risk of conservative capacity compression via the 1-WL refinement heuristic to preserve linear-time computational guarantees. It trades customizable headroom for total immunity to configuration bias, sealing the pipeline behind an immutable Design-by-Contract (DbC) execution boundary that prevents any mathematically invalid state from surviving runtime execution.

PRACTICALLY MEANS THAT YOUR TRADING BOT REVERTS TO NEAR-ZERO LOSS, OR NEAR-ZERO PROFIT WHEN IT CAN'T DETERMINE A WINNING STRATEGY.

# Degen view:

# AGOP Decision Engine — The Executive Toxic Flow Gate

A deterministic, algebraic multi-criteria operations verification pipeline and risk-routing matrix built to automate multi-legged execution alternative rankings while surviving toxic counter-party flow. It completely rejects open-ended, fuzzy statistical AI recommendations to enforce strict, un-gameable mathematical fixed points.

## 🧠 Core Mechanics
- **The Deterministic Sandwich:** Machine learning (`agop_ml.py`) is ruthlessly stripped of executive authority and contained strictly on the extreme boundaries. It runs ahead of the engine to act as a low-level data loader (normalizing raw market strings into criteria arrays) and wakes up at the back gate to map post-hoc sensitivity traces. 
- **The Game-Theoretic Vault:** The core algorithm (`agop_core.py`) is 100% free of ML. It projects execution options onto a finite two-player game payoff matrix ($U_{DM}, U_{CP}$) pitting your capital against a strategic opponent (a toxic front-running HFT desk or arbitrage competitor), analytically computing the exact mixed-strategy Nash equilibrium to choose an action that is mathematically immune to exploitation.
- **Topological Invariant Filters:** Surviving alternatives are passed through three strict deductive filters (*pain, standardisation, regulatory*), a Weisfeiler-Lehman (WL) graph isomorphism color-refinement subroutine to group structural classes, and a final $M=10,000$ Monte Carlo integration loop to verify a non-negative Lower Confidence Bound (LCB) under capacity-constrained population limits.

## ⚠️ The GIGO Pact & Disclaimer
- **Garbage In, Garbage Out:** The code behaves as a cold decision procedure. If your initial prior weights or risk metrics are wrong, the engine will flawlessly compute a highly sophisticated, mathematically pristine ranking that is completely useless.
- **Empty Regulatory Placeholder:** The repository includes a `regulatory` stopgap key gated by the API Pydantic validator. It is left entirely blank by design. It contains zero legal compliance data or geographical jurisdiction rules. You can bypass it by setting the parameters to zero, or fill it with your own specific transaction-fee or clearance constraints. 
- **Status:** Shared purely as an open-source, unmonetized thesis under the MIT License. The author assumes exactly zero responsibility for what you choose to execute with these primitives in the wild.


---

## 1. What it does

Input: a decision problem `P = (V_0, A, Θ, R)` consisting of a finite set of
alternatives, a set of decision axes (one real-valued function per axis per
alternative), a parameter vector, and a symmetric compatibility relation.

Output: a ranked list of alternatives `R = (v_(1), …, v_(k))`, each annotated
with a structural class, an outcome estimate with a lower confidence bound, and
an adoption status. Every returned alternative satisfies four admissibility
conditions:

1. it survives three deductive filters (pain, standardisation, regulatory);
2. it supports an equilibrium in the two-player market game;
3. its lower confidence bound on decision value is non-negative;
4. its adoption is supported by a capacity-constrained population equilibrium.

The procedure is deterministic under six explicit conventions (tie-breaking,
equilibrium selection, RNG seed, resonance target, ranking score, adoption
game). It does not learn parameters; it evaluates consequences of given
parameters.

---

## 2. Quick start

```bash
git clone <this-repo> agop-engine && cd agop-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:8000/> in a browser. The UI loads with the reference
instance pre-selected. Click **Run AGOP**.

---

## 3. Installation

### 3.1 Requirements

- **Python ≥ 3.10** (uses PEP 604 unions, `dataclasses`, structural pattern matching where applicable).
- **pip** and **venv** (bundled with CPython ≥ 3.4).
- No native compilation, no GPU, no external services.

### 3.2 Dependencies

Declared in `requirements.txt`:

```
numpy>=1.24
scipy>=1.10
networkx>=3.0
fastapi>=0.104
uvicorn[standard]>=0.24
matplotlib>=3.7
pydantic>=2.0
```

Install:

```bash
pip install -r requirements.txt
```

### 3.3 Editable install (optional)

If you want to `import agop_core` from other projects:

```bash
pip install -e .
```

(Requires a minimal `pyproject.toml`; not included by default because AGOP is
usable as a self-contained folder.)

### 3.4 Verifying the installation

```bash
python -c "from agop_core import reference_instance, agop; \
           s = agop(reference_instance()); \
           print(s.R)"
# ['S_12', 'S_2', 'S_1']
```

If this prints `['S_12', 'S_2', 'S_1']`, the algorithm computes correctly.

---

## 4. User guide

### 4.1 Browser UI

Start the server:

```bash
python run.py                    # default 127.0.0.1:8000
python run.py --host 0.0.0.0 --port 9000
```

Open the printed URL. The UI has three panels:

1. **Input** — choose *Reference instance* (default) or *Custom JSON*, paste a
   problem specification, click **Run AGOP**.
2. **Results** — a summary banner, the ranking, the survivor set `D*`, the
   composite scores `Φ`, and the full phase trace.
3. **Visualisation** — a 4-panel PNG chart (phase funnel, resonance scatter,
   composite bars, payoff matrix) generated by `agop_charts.py`.

The chart is regenerated on each run.

### 4.2 REST API

FastAPI serves OpenAPI 3.1 automatically:

- **Interactive docs**: <http://127.0.0.1:8000/docs>
- **ReDoc**: <http://127.0.0.1:8000/redoc>
- **OpenAPI JSON**: <http://127.0.0.1:8000/openapi.json>

Quick curl example:

```bash
curl -X POST http://127.0.0.1:8000/api/agop \
     -H "Content-Type: application/json" \
     -d '{"preset": "reference"}'
```

Returns the full JSON trace (see §5).

Chart as PNG:

```bash
curl -X POST http://127.0.0.1:8000/api/agop/chart \
     -H "Content-Type: application/json" \
     -d '{"preset": "reference"}' \
     -o agop_chart.png
```

Custom problem:

```bash
curl -X POST http://127.0.0.1:8000/api/agop \
     -H "Content-Type: application/json" \
     -d @problem.json
```

### 4.3 Python library

```python
from agop_core import reference_instance, agop

problem = reference_instance()
state   = agop(problem)

print(state.R)          # ['S_12', 'S_2', 'S_1']
print(state.D_star)     # {'S_1', 'S_2', 'S_10', 'S_12'}
print(state.Phi)        # {'S_1': 18.65, 'S_2': 22.58, 'S_12': 26.93, 'S_10': 13.11}
```

Charts from Python:

```python
from agop_core import reference_instance, agop
from agop_charts import render_chart

problem = reference_instance()
state   = agop(problem)
png_bytes = render_chart(problem, state)
open("chart.png", "wb").write(png_bytes)
```

---

## 5. API reference

### 5.1 Endpoints

| Method | Path | Description |
|---|---|---|
| GET  | `/`               | Serves `agop_ui.html`. |
| GET  | `/api/health`     | Returns `{"status": "ok"}`. |
| POST | `/api/agop`       | Runs AGOP, returns JSON trace. |
| POST | `/api/agop/chart` | Runs AGOP, returns `image/png`. |
| GET  | `/docs`           | FastAPI interactive documentation. |
| GET  | `/openapi.json`   | OpenAPI 3.1 schema. |

### 5.2 Request schema

```jsonc
{
  "preset": "reference",           // optional; "reference" is the only named preset
  "problem": {                     // optional; overrides preset if both present
    "V": ["S_1", "S_2", "..."],
    "edges": [["S_1", "S_2", 1], ["S_12", "S_3", -1]],  // (u, v, sign)
    "families": {"S_1": "B", "S_2": "A", "S_10": "C", "S_12": "C"},
    "axes": {
      "pain":         {"S_1": 0.9, "..." : 0.0},
      "integration":  {"S_1": 0.9, "..." : 0.0},
      "regulatory":   {"S_1": 0.2, "..." : 1.0},
      "readiness":    {"S_1": 0.85, "..." : 0.5},
      "barrier":      {"S_1": 0.15, "..." : 0.3}
    },
    "theta": {
      "c_b": 10.0, "c_s": 3.0, "w": 15.0, "ell": 2.0, "v_bar": 20.0,
      "theta_1": 0.5, "theta_2": 0.5, "theta_3": 0.5,
      "sigma_R": 0.2, "sigma_B": 0.2,
      "alpha_1": 0.4, "alpha_2": 0.4, "alpha_3": 0.2,
      "M": 10000, "alpha": 0.05, "seed": 42, "K": 3, "h": 3,
      "L_R": {"S_1": [30, 5], "S_2": [40, 6], "S_10": [25, 4], "S_12": [35, 5]},
      "L_C": {"S_1": [6, 2],  "S_2": [9, 3],  "S_10": [7, 2],  "S_12": [8, 2]}
    }
  }
}
```

`L_R` and `L_C` must cover all `V` (the API validator enforces this).

### 5.3 Response schema (`POST /api/agop`)

```jsonc
{
  "ranking": ["S_12", "S_2", "S_1"],
  "D_star": ["S_1", "S_2", "S_10", "S_12"],
  "convergence_depth": 1,
  "gamma": {"S_1": 0.51, "S_2": 0.85, "...": 0.0},
  "payoff": {
    "U_DM": [[5.0, -10.0], [12.0, -3.0]],
    "U_CP": [[5.0, 0.0], [3.0, 0.0]]
  },
  "nash": {
    "sigma_DM": [0.0, 1.0],
    "sigma_CP": [1.0, 0.0]
  },
  "systems": {"loop_count": 10, "reinforcing": 9, "balancing": 1, "bottleneck_size": 1},
  "topology": {"S_1": "B", "S_2": "A", "S_10": "C", "S_12": "C"},
  "adoption": {"S_1": 1, "S_2": 1, "S_10": 0, "S_12": 1},
  "monte_carlo": {
    "Y_hat": {"S_1": 24.04, "S_2": 30.97, "S_10": 17.99, "S_12": 27.02},
    "LCB":   {"S_1": 23.95, "S_2": 30.86, "S_10": 17.91, "S_12": 26.93}
  },
  "resonance": {"S_1": 0.779, "S_2": 0.732, "S_10": 0.732, "S_12": 1.0},
  "composite": {"S_1": 18.65, "S_2": 22.58, "S_10": 13.11, "S_12": 26.93}
}
```

### 5.4 Error codes

| Status | When |
|---|---|
| 400 | Body is not valid JSON, or a required `L_R`/`L_C` entry is missing, or `K > |D*|`, or a predicate threshold is out of `[0,1]`. |
| 422 | Pydantic rejected the request schema. |
| 500 | Internal error (should not occur on the reference instance). |

Errors return `{"detail": "..."}`.

---

## 6. Practical applications

AGOP applies to any decision problem with (i) finite alternatives, (ii) multiple
criteria, (iii) at least one strategic counter-party, and (iv) outcome
uncertainty. Concrete applications grouped by domain:

### 6.1 Commercial strategy

- **Market-entry sequencing.** Which SMB segments to prioritise for a software
  product, given standardisation levels, regulatory burden, and competitor
  response.
- **Product portfolio prioritisation.** Which features to build first, given
  engineering cost, customer adoption likelihood, and competitive reactions.
- **Channel-partner onboarding.** Which resellers or integrators to qualify
  first, given integration surface, revenue potential, and adverse-selection
  risk.

### 6.2 Investment and capital allocation

- **Venture-capital deal screening.** Which startups to advance to diligence,
  given founder quality, market size, competitive dynamics, and adverse
  selection in the term sheet.
- **R&D project selection.** Which research bets to fund, given scientific
  plausibility, engineering cost, competitive risk, and technical uncertainty.
- **M&A target prioritisation.** Which acquisition candidates to pursue first,
  given strategic fit, regulatory exposure, integration cost, and target-
  management game.

### 6.3 Public and non-profit

- **Grant allocation.** Which proposals to fund, given scientific merit,
  feasibility, budget, and reviewer-game dynamics.
- **Clinical-trial site selection.** Which hospital sites to activate first,
  given patient population, regulatory standing, and principal-investigator
  capacity.
- **Content production prioritisation.** Which articles, videos, or courses to
  produce, given audience size, production cost, editorial alignment, and
  competitor response.

### 6.4 Operational

- **Supplier onboarding.** Which suppliers to qualify first, given reliability,
  integration effort, contract terms, and supplier-side game.
- **Hiring prioritisation.** Which roles to fill first, given impact, cost,
  availability, and candidate-market dynamics.
- **Cloud-migration waves.** Which workloads to migrate first, given risk,
  cost, business criticality, and vendor lock-in.

### 6.5 Where AGOP is *not* the right tool

AGOP assumes the axes can be evaluated for each alternative, that a finite
two-player game captures the strategic interaction, and that the outcome
distribution can be specified a priori. It is not suitable for (a) problems
without a measurable counter-party, (b) problems where outcomes are
qualitatively incommensurable, (c) problems where the alternative set is
open-ended or continuously parameterised.

---

## 7. Architecture

```
┌──────────────┐   HTTP   ┌──────────────┐  in-proc  ┌──────────────┐
│  Browser UI  │ ───────► │  FastAPI     │ ────────► │  agop_core   │
│ (agop_ui)    │ ◄─────── │ (agop_api)   │ ◄──────── │  (algorithm) │
└──────────────┘   PNG    └──────────────┘   State   └──────────────┘
                                │                          │
                                │                          ▼
                                ▼                    ┌──────────────┐
                          ┌──────────────┐           │ agop_charts  │
                          │  Pydantic    │           │ (matplotlib) │
                          └──────────────┘           └──────────────┘
```

Files:

- `agop_core.py` — the algorithm, dataclasses, reference instance.
- `agop_charts.py` — matplotlib `Figure` + `FigureCanvasAgg`, returns PNG bytes.
- `agop_api.py` — FastAPI app, Pydantic schemas, endpoints.
- `agop_ui.html` — single-file vanilla JS UI.
- `run.py` — uvicorn launcher.

The API is **synchronous** (`def`, not `async def`); FastAPI runs synchronous
handlers in a threadpool, avoiding event-loop blocking during the CPU-bound
Monte Carlo phase.

---

## 8. Testing

```bash
python -c "from agop_core import reference_instance, agop; \
           s = agop(reference_instance()); \
           assert s.R == ['S_12', 'S_2', 'S_1']; \
           assert s.D_star == {'S_1', 'S_2', 'S_10', 'S_12'}; \
           print('PASS')"
```

Expected output: `PASS`.

The reference instance is verified against the theorems of §5 of the article:

- **Theorem 1 (well-typedness)** — dataclass annotations and runtime type
  checks.
- **Theorem 2 (fixed-point convergence)** — `convergence_depth` is `1`.
- **Theorem 3 (equilibrium existence)** — Nash subroutines raise only if the
  existence theorem is violated.
- **Theorem 4 (determinism)** — fixed seed `42`, re-running gives identical
  output.
- **Theorem 5 (admissibility)** — `LCB > 0` for every returned alternative.

---

## 9. Limitations

1. `L_R` and `L_C` are **inputs**, not learned. Different priors yield
   different rankings.
2. Weisfeiler–Lehman is a **coarsening** of graph isomorphism; two non-
   isomorphic alternatives may share a class if WL cannot distinguish them.
3. Nash existence requires **mixed strategies in general**; pure-strategy
   equilibria may not exist for arbitrary parameters.
4. Capacity `K` is **exogenous**.
5. Data validity is **audited**, not proven.
6. Determinism holds **only under conventions C1–C6**.

See §9 of the reference article for the full statement.

---

## 10. License

MIT License
