# Assignment Compliance Audit Checklist

This audit verifies the completed codebase against all assignment requirements and specifications. Every verification was conducted by direct code inspection and automated test execution.

| Requirement | Where Implemented | How Verified | Status |
| :--- | :--- | :--- | :--- |
| **Data Input & Validation** | `core/loader.py`, `core/validation.py`, `ui/data_tab.py` | Unit tests in `test_constraints.py`; AppTest in `test_app_smoke.py`; manual load checks on JSON and CSV. | **Done** |
| **Small Dataset (3 classes, 6 teachers, 4 rooms, 5×6)** | `data/small.json`, `scripts/generate_datasets.py` (`build_small`) | `scripts/generate_datasets.py` validation report; verified solvable in `test_backtracking.py`. | **Done** |
| **Large Dataset (8+ classes, 15+ teachers, 8+ rooms, 5×7)** | `data/large.json`, `scripts/generate_datasets.py` (`build_large`) | `scripts/generate_datasets.py` validation report; verified solvable in `test_backtracking.py`. | **Done** |
| **Hard Constraints H1–H9** | `core/constraints.py` (`check_h1` through `check_h9`) | 31 unit tests in `tests/test_constraints.py` with hand-crafted positive and negative fixtures. | **Done** |
| **Soft Constraints S1–S5 (at least 3 required)** | `core/constraints.py` (`penalty_s1` through `penalty_s5`) | All 5 implemented and unit-tested in `tests/test_constraints.py`. | **Done** |
| **CSP Formulation (Variables, Domains, Graph)** | `core/problem_builder.py` (`build_csp`, `build_sessions`, `build_domains`) | `tests/test_backtracking.py`, `tests/test_constraints.py`. Unary filters applied before binary search. | **Done** |
| **Cost Function ($1000 \times \text{hard} + \text{soft}$)** | `core/cost.py` (`evaluate`, `conflicts`) | 11 unit tests in `tests/test_cost.py`; re-score verifier in `experiments/runner.py`. | **Done** |
| **Backtracking (Baseline + 3 Heuristics)** | `algorithms/backtracking.py` (`BacktrackingSolver`, `BacktrackingConfig`) | `tests/test_backtracking.py` (12 tests) verifying MRV, Degree heuristic, and Forward Checking with undo trail. | **Done** |
| **Min-Conflicts Local Search** | `algorithms/min_conflicts.py` (`MinConflictsSolver`) | `tests/test_min_conflicts.py` (7 tests) verifying initial assignment, min-clash choices, walk ($p_{\text{walk}}$), restarts. | **Done** |
| **Simulated Annealing (Bonus)** | `algorithms/simulated_annealing.py`, `algorithms/neighbourhood.py` | `tests/test_simulated_annealing.py` (5 tests) verifying $T_0$ auto-calibration, geometric cooling, stagnation stop. | **Done** |
| **Hybrid Solver: Backtracking + SA (Bonus)** | `algorithms/hybrid.py` (`HybridSolver`) | `tests/test_hybrid.py` (4 tests) verifying Phase 1 feasibility (0 hard violations) and Phase 2 soft penalty optimization. | **Done** |
| **External Solver Cross-Check Only** | `tests/test_backtracking.py` (`TestPythonConstraintCrossCheck`) | Code inspection confirms `python-constraint` is only imported in test harness for cross-checking, never in solvers. | **Done** |
| **Application UI & Tabs** | `app.py`, `ui/data_tab.py`, `ui/generate_tab.py`, `ui/timetable_tab.py` | `tests/test_app_smoke.py` (`AppTest` headless verification); no UI logic in solver files. | **Done** |
| **Timetable Grids (Class, Teacher, Room)** | `services/grid_builder.py`, `ui/timetable_tab.py` | `tests/test_grid_builder.py` (7 tests) checking cell contents, lunch/break labels, and lab continuations. | **Done** |
| **CSV and Excel Export** | `services/exporter.py` (`export_grid_csv`, `export_flat_csv`, `export_timetable_excel`) | `tests/test_exporter.py` (4 tests) verifying multi-sheet openpyxl workbook and tabular session CSVs. | **Done** |
| **Manual Editing & Immediate Hard Flagging** | `services/editing.py` (`preview_move`, `apply_move`, `MoveHistory`) | `tests/test_editing.py` (7 tests) verifying H1–H5 detection, day overflow check, immutability, undo, and reset. | **Done** |
| **Experiments Suite & Convergence Graphs** | `experiments/runner.py`, `aggregate.py`, `plots.py`, `run_all.py` | `tests/test_experiments.py` (4 tests); full benchmark run producing `results/summary.md` and all plots. | **Done** |
| **Reproducibility with Fixed Seeds** | All solver modules, `experiments/config.py`, `runner.py` | `tests/test_experiments.py::test_seed_reproducibility` confirming identical outputs across runs with same seed. | **Done** |
| **Architectural Separation (Clean UI)** | `core/`, `algorithms/`, `services/`, `ui/`, `experiments/` | Verified: zero `streamlit` imports in core/algorithms/services/experiments; zero CSP logic in UI tabs. | **Done** |
| **Repository Deliverables** | Clean repo layout, `README.md`, `guide.md`, `data/`, `results/` | Verified all artifacts committed, no broken links, no hardcoded local paths. | **Done** |

