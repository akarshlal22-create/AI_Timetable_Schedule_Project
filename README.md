# AI-Based Timetable Scheduler

An automated college timetable generator that models course scheduling as a Constraint Satisfaction Problem (CSP). Given student batches, teaching faculty, classrooms, laboratories, and weekly time grids, the system searches for conflict-free schedules that satisfy all hard constraints (preventing double-booking, room capacity overruns, room-type mismatches, and teacher unavailability) while minimising soft-constraint penalties (reducing student class gaps, spreading subjects evenly across the week, ensuring consecutive lab periods, and honouring teacher scheduling preferences).

## Features

- Complete CSP formulation with unary domain filtering and binary constraint graphs.
- Five distinct solving algorithms:
  - Backtracking search (deterministic baseline).
  - Backtracking with heuristics: Minimum Remaining Values (MRV), Degree heuristic tie-breaking, and Forward Checking with undo trail.
  - Min-Conflicts local search with random-walk escaping and random restarts.
  - Simulated Annealing with auto-calibrated initial temperature and geometric cooling (Bonus).
  - Two-phase Hybrid solver combining Backtracking feasibility with Simulated Annealing soft-penalty optimization (Bonus).
- Pure cost function evaluation ($1000 \times \text{hard violations} + \text{soft penalty}$).
- Streamlit web interface with layout-separated tabs for data input, algorithm generation, and full-width timetable grids.
- Multi-dimensional timetable views (by Class, by Teacher, and by Room) with muted breaks and visual clash highlighting.
- Real-time manual slot editor with instant hard-conflict preview, non-blocking move application, undo history, and reset.
- Data export to CSV (individual grids or flat records) and multi-sheet Excel (.xlsx) workbooks.
- Reproducible benchmarking suite with scaling analysis across varying problem sizes.

## Setup

### Requirements
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Standard virtual environment tools (`venv` or `conda`)

### Installation

```bash
# Clone the repository and navigate to the project directory
cd timetable_scheduler

# Create and activate a virtual environment
python -m venv venv
# On Windows (Command Prompt / PowerShell):
venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

## How to Run

### 1. Generate Problem Datasets
Generate the deterministic sample datasets (`data/small.json` and `data/large.json`):
```bash
python scripts/generate_datasets.py
```

### 2. Run the Web Application
Launch the interactive Streamlit user interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Automated Test Suite
Execute all unit and integration tests:
```bash
pytest
```

### 4. Run Experimental Benchmarks
Execute the full benchmark suite across all five algorithms and both datasets with scaling analysis:
```bash
python -m experiments.run_all
```
To run a fast verification pass (3 seeds, 10s limit, skipping scaling):
```bash
python -m experiments.run_all --quick
```
To run a single algorithm on a specific dataset:
```bash
python -m experiments.run_all --only "Min-Conflicts" --dataset small
```

## Input Data Format

The application accepts problem specifications via JSON or structured multi-file CSV uploads.

### JSON Format
Problem files (`data/small.json`, `data/large.json`) define:
- `time`: Grid structure (`working_days`, `periods_per_day`, `break_periods`).
- `subjects`: Course codes, full names, weekly hours, and type (`theory` or `lab`).
- `teachers`: Names, teachable subject codes, unavailable (day, period) slots, and maximum weekly teaching loads.
- `classes`: Student batches, student enrollment strength, required subject codes, and designated lunch period index.
- `rooms`: Room names, seating capacity, and room type (`classroom` or `lab`).

### CSV Format
Uploads accept a set of five related CSV files: `time_structure.csv`, `subjects.csv`, `teachers.csv`, `classes.csv`, and `rooms.csv`. Column headers must match the attributes in the corresponding dataclasses defined in `core/models.py`.

## Screenshots

Visual walkthrough of key application screens:

### Data Input & Validation
![Data Tab](docs/screenshots/data_tab.png)
*Data source selection, input validation reporting, and read-only entity inspection.*

### Algorithm Execution & Metrics
![Generate Tab](docs/screenshots/generate_result.png)
*Execution metrics summary and convergence curve for local search.*

### Timetable Grid View
![Class Grid View](docs/screenshots/class_grid.png)
*Full-width weekly schedule grid by Class, highlighting lunch periods and lab continuation slots.*

### Manual Editing & Conflict Preview
![Conflict Highlighting](docs/screenshots/conflict_highlighted.png)
*Interactive session relocation with real-time constraint violation preview and red clash highlights.*

### Experimental Comparison
![Experiment Graph](docs/screenshots/experiment_graph.png)
*Benchmark execution runtime comparison across algorithms on log scale.*

*(Note: See `docs/screenshots/README.md` for instructions on taking or updating app screenshots.)*

## Project Structure

```
timetable_scheduler/
├── app.py              # Streamlit web application entry point
├── requirements.txt    # Project Python package dependencies
├── .gitignore          # Repository ignore rules for caches and local artifacts
├── guide.md            # Concise single-page technical reference guide
├── core/               # Pure-Python CSP definitions, constraints, loader, and cost engine
├── algorithms/         # Backtracking, Min-Conflicts, Simulated Annealing, and Hybrid solvers
├── services/           # Grid builder, manual editing logic, solver registry, and Excel exporter
├── ui/                 # Streamlit UI tab views (data, generation, timetable grids) and styling
├── experiments/        # Benchmark runner, metrics aggregation, plotting, and CLI entry point
├── scripts/            # Dataset generation and standalone algorithm demonstration scripts
├── data/               # Small and large JSON problem definitions
├── results/            # Benchmark CSV summaries, comparison markdown, and visualization plots
├── docs/               # Screenshots and assignment compliance checklist
└── tests/              # Pytest test suite covering core, solvers, services, UI, and experiments
```

## Results

For detailed experimental results and comparison tables across all algorithms and datasets, refer to:
- [results/summary.md](results/summary.md): Markdown comparison table ready for report inclusion.
- [results/summary.csv](results/summary.csv): Machine-readable aggregated benchmark metrics.
- [results/scaling.csv](results/scaling.csv): Performance metrics across problem sizes (3 to 12 classes).

## References

1. **Artificial Intelligence: A Modern Approach (4th Edition)** — Stuart Russell and Peter Norvig. Chapter 6: *Constraint Satisfaction Problems* (CSP formulation, Backtracking, MRV, Degree heuristic, Forward Checking).
2. **Min-Conflicts Heuristic** — Steven Minton, Mark D. Johnston, Philips D. Laird, and Amy B. Philips (1992). *Minimizing conflicts: a heuristic repair method for constraint satisfaction and scheduling problems*. Artificial Intelligence, 58(1-3), 161-205.
3. **Simulated Annealing** — Scott Kirkpatrick, C. Daniel Gelatt, and Mario P. Vecchi (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671-680.

## Authors

- Akarsh Lal (2025KUAD3009) — Indian Institute of Information Technology Kota
