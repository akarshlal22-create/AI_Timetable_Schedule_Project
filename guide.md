# AI-Based Timetable Scheduler — Project Guide

## What the project does
This project automatically schedules a weekly college timetable using Constraint Satisfaction Problem (CSP) techniques. Given student classes, faculty, subjects, rooms, and time slots, it generates a clash-free schedule satisfying all operational hard constraints (preventing double-booking and room mismatches) while reducing soft-constraint penalties (minimising student idle periods, distributing lectures across the week, and accommodating teacher preferences).

## How it is organised
- `core/`: Pure-Python problem models, CSV/JSON loaders, input validation, CSP graph construction, constraint definitions (H1–H9, S1–S5), and cost scoring.
- `algorithms/`: Five solver implementations (Backtracking baseline, Backtracking with heuristics, Min-Conflicts, Simulated Annealing, and Hybrid).
- `services/`: Presentation logic decoupled from UI (grid transformations, manual move validation, solver registry, CSV/Excel export).
- `ui/`: Streamlit web interface divided into Data, Generate, and Timetable tabs, with minimal typography styling.
- `experiments/`: Automated benchmarking suite, metric aggregators, publication plots, and scaling experiments.
- `scripts/`: Problem generator for reproducible small, large, and scaled datasets, plus command-line solver runners.
- `data/`: Small (3 classes, 34 sessions) and large (8 classes, 112 sessions) sample problem instances in JSON format.
- `results/`: Benchmark raw logs, summary comparison tables in CSV and Markdown, and convergence plots.
- `docs/`: Assignment compliance audit checklist and application screenshot documentation.
- `tests/`: 99 unit and integration tests covering the entire codebase.

## How to run it
```bash
pip install -r requirements.txt          # Install dependencies
python scripts/generate_datasets.py      # Generate sample datasets
pytest                                   # Run all automated tests
streamlit run app.py                     # Launch the web application
python -m experiments.run_all --quick    # Run quick benchmark pass
```

## How the CSP model works
- **Variables**: Each class course meeting is a discrete variable (`Session`). A 1-period theory lecture occupies one time slot; a 2-period lab session occupies two consecutive slots $(p, p+1)$.
- **Domains**: Every legal $(day, period, room)$ triple. Domains are pruned before search by unary constraints: room type (lab vs. classroom), room capacity $\ge$ class strength, teacher availability, and break/lunch periods.
- **Constraints**: Binary relations between assigned sessions prevent shared resource conflicts (no teacher, class, or room double-booking in any period).

## Algorithms
- **Backtracking (baseline)**: Plain depth-first search that tests domain values in static order and backtracks immediately upon constraint violations.
- **Backtracking (MRV + Degree + Forward Checking)**: High-performance complete search using Minimum Remaining Values variable ordering, Degree heuristic tie-breaking, and Forward Checking with an undo trail to prune invalid future domain choices early.
- **Min-Conflicts (Local Search)**: Starts from a complete initial assignment and iteratively repairs random conflicted variables by selecting the value minimizing active clashes, using random walks and restarts to escape plateaus.
- **Simulated Annealing (Bonus)**: Stochastic metaheuristic that proposes single moves and slot swaps, accepting worsening moves based on an auto-calibrated temperature schedule ($P = \exp(-\Delta / T)$) to escape local optima.
- **Hybrid Solver (Bonus)**: Two-phase method that runs heuristic Backtracking to achieve guaranteed feasibility (0 hard violations) in milliseconds, then uses zero-clash Simulated Annealing to reduce soft penalties by 50–85%.

## Cost function
Total timetable cost is computed as:
$$\text{Cost} = 1000 \times (\text{Hard Violations}) + \sum (\text{Soft Penalties})$$
Hard constraints (H1–H9) enforce resource uniqueness, room suitability, and load limits. Soft constraints (S1–S5) penalise student timetable gaps, poor subject spread across days, split lab periods, ignored faculty preferences, and excessive consecutive lectures.

## How the app is used
1. **Data tab**: Choose a sample dataset (Small or Large) or upload custom JSON/CSV files. View solvability status and browse entity tables (read-only).
2. **Generate tab**: Select any of the five algorithms, configure random seed and time limit, and click **Generate Timetable**. View execution time, nodes expanded or iterations, and convergence plots.
3. **Timetable tab**: Inspect weekly grids across Class, Teacher, or Room views with muted breaks and highlighted clashes. Export the full schedule to CSV or multi-sheet Excel workbooks.
4. **Manual editing**: Move any session to an alternative slot with instant pre-move conflict warnings. Apply moves to observe real-time red clash highlights, with full support for 1-click Undo and Reset.
