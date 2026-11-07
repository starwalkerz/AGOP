Verify it runs
bash

pip install -r requirements.txt
python run.py

Then in a browser open http://127.0.0.1:8000/.

Expected: after ~1 s, a page with:

    Summary panel: ranking S_12 > S_2 > S_1, |D*| = 4, |R| = 3, convergence depth 1, market Nash (standard, adopt).

    Results panel: the three tables populated.

    Visualisation panel: a 4-panel PNG (phase funnel, resonance scatter, composite bars, payoff matrix).

Test the API directly:
bash

curl -s -X POST http://127.0.0.1:8000/api/agop \
     -H "Content-Type: application/json" \
     -d '{"preset": "reference"}' | python -m json.tool

Expected "ranking": ["S_12", "S_2", "S_1"] and "D_star": ["S_1", "S_2", "S_10", "S_12"].

Fetch the chart as PNG:
bash

curl -s -X POST http://127.0.0.1:8000/api/agop/chart \
     -H "Content-Type: application/json" \
     -d '{"preset": "reference"}' \
     -o agop_chart.png
file agop_chart.png   # PNG image data

The OpenAPI UI at http://127.0.0.1:8000/docs lists every endpoint and lets you try requests interactively.

Three points about the deliverable. (1) The agop_core.py emitted above is identical to the reference implementation from the previous message except for one addition: phase_6_deductive now returns (D*, k*) and State carries convergence_depth, which the API and chart need. (2) The reference instance's L_R/L_C entries were extended to cover all 13 alternatives (not just the four D* members) so that the API validator set(L_R) == set(V) passes — this is required for the fast-fail schema check, not for the algorithm itself. (3) The HTML/CSS/JS is a single file with no build step and no external dependencies; it will work offline once served, except for the two POST calls to the FastAPI backend.
