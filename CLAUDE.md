# Project Rules

- **docs/solutions/** — documented detector fixes and window-detection patterns, organized by category. Relevant when debugging detection gaps or designing new recovery passes.
- Make detector fixes in code, not by hardcoding page, room, group-home, or coordinate-specific overrides.
- Treat user corrections as evidence for reusable computer-vision rules that should work across many diagrams.
- Expect multiple window symbol families over time. Keep detector logic modular so new symbol types can be added without breaking existing ones.
- Door openings, door swings, wall gaps, labels, dimensions, fixtures, and spaces between a door and a window cap are not windows.
- When a measurement or dimension line passes through a window symbol, prefer rules that recover the underlying cap/pane evidence instead of relying on uninterrupted strokes.

## Window detection — run & resume (read docs/solutions/SESSION_HANDOFF.md first)

- **State:** page-17 (6 plans) macro F1@10 0.926 / micro 0.936, FP=5. Pipeline = deterministic detector + door-swing reject + ML post-filter (reject P(window)<0.30). Branch `window-detector-accuracy-loop`; safe restore tag `loop-checkpoint-0.885`.
- **Python env:** uv venv at `/home/user/wd-venv` (repo `.venv` is a locked Windows venv). Recreate cmd in SESSION_HANDOFF.md.
- **Run server:** `WINDOW_ML_FILTER=1 PYTHONPATH=. /home/user/wd-venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000` → http://localhost:8000 (Gemini optional; deterministic-only).
- **Measure:** `WINDOW_ML_FILTER=1 PYTHONPATH=. /home/user/wd-venv/bin/python loop/harness.py --tol 10`.
- **Retrain after corrections:** `PYTHONPATH=. /home/user/wd-venv/bin/python loop/train_classifier.py` then restart server.
- **Correction convention:** real windows keep `W-NN`; rename wrong boxes with a reason (DOOR/STAIRS/no caps/not in a wall/…) → treated as hard negatives.
- **Settled:** deterministic hard rules for window-vs-not DON'T separate (overlap); user rules live as ML features. Don't re-attempt hard-rule filters. Page-17 answers do NOT auto-transfer to page-18 (different scale/position per home). Lever to higher accuracy = more corrected pages.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
