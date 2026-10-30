"""scripts/generate_datasets.py — Generate small.json and large.json deterministically.

Run:
    python scripts/generate_datasets.py

Both datasets are validated and a solvability summary is printed.
The script exits with code 1 if any dataset fails validation.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.loader import save_problem_to_json
from core.models import (
    ClassBatch, Problem, Room, Subject, Teacher, TimeStructure,
)
from core.validation import validate_problem

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _pick_unavailable(rng: random.Random, days: int, periods: int,
                      breaks: set, n: int = 2) -> frozenset:
    """Pick n random (day, period) pairs that are not break periods."""
    pool = [(d, p) for d in range(days) for p in range(periods) if p not in breaks]
    return frozenset(rng.sample(pool, min(n, len(pool))))


# ---------------------------------------------------------------------------
# Small dataset: 3 classes, 6 teachers, 4 rooms, 5 days × 6 periods
# ---------------------------------------------------------------------------

def build_small(seed: int = SEED) -> Problem:
    rng = _rng(seed)

    time = TimeStructure(
        working_days=5,
        periods_per_day=6,
        break_periods=frozenset({3}),   # period index 3 is the mid-day break
    )

    subjects = {
        "MATH201": Subject("MATH201", "Mathematics II",      weekly_hours=3, type="theory"),
        "PHY201":  Subject("PHY201",  "Engineering Physics", weekly_hours=2, type="theory"),
        "CS201":   Subject("CS201",   "Data Structures",     weekly_hours=3, type="theory"),
        "CS202":   Subject("CS202",   "OOP with Java",       weekly_hours=2, type="theory"),
        "EE201":   Subject("EE201",   "Basic Electronics",   weekly_hours=2, type="theory"),
        "PHY_LAB": Subject("PHY_LAB", "Physics Lab",         weekly_hours=1, type="lab"),
        "CS_LAB":  Subject("CS_LAB",  "Programming Lab",     weekly_hours=1, type="lab"),
    }

    # 6 teachers — each covers ≥1 subject, load balanced
    teachers = {
        "Dr. Sharma": Teacher(
            name="Dr. Sharma",
            teachable_subjects=frozenset({"MATH201"}),
            unavailable_slots=_pick_unavailable(rng, 5, 6, {3}, n=1),
            max_weekly_load=12,
            preferences={"no_first_period": True},
        ),
        "Dr. Patel": Teacher(
            name="Dr. Patel",
            teachable_subjects=frozenset({"PHY201"}),
            unavailable_slots=_pick_unavailable(rng, 5, 6, {3}, n=1),
            max_weekly_load=10,
        ),
        "Dr. Rao": Teacher(
            name="Dr. Rao",
            teachable_subjects=frozenset({"CS201", "CS202"}),
            unavailable_slots=_pick_unavailable(rng, 5, 6, {3}, n=2),
            max_weekly_load=14,
            preferences={"no_first_period": True},
        ),
        "Prof. Singh": Teacher(
            name="Prof. Singh",
            teachable_subjects=frozenset({"EE201", "CS202"}),
            unavailable_slots=_pick_unavailable(rng, 5, 6, {3}, n=1),
            max_weekly_load=12,
        ),
        "Dr. Mehta": Teacher(
            name="Dr. Mehta",
            teachable_subjects=frozenset({"PHY_LAB", "PHY201"}),
            unavailable_slots=frozenset(),
            max_weekly_load=8,
        ),
        "Prof. Kumar": Teacher(
            name="Prof. Kumar",
            teachable_subjects=frozenset({"CS_LAB", "CS201"}),
            unavailable_slots=_pick_unavailable(rng, 5, 6, {3}, n=1),
            max_weekly_load=8,
        ),
    }

    # 3 classes (batches), each ~30–40 students
    classes = {
        "CS-A": ClassBatch(
            name="CS-A",
            strength=38,
            subject_codes=("MATH201", "PHY201", "CS201", "CS202", "PHY_LAB", "CS_LAB"),
            lunch_slot=3,
        ),
        "CS-B": ClassBatch(
            name="CS-B",
            strength=35,
            subject_codes=("MATH201", "PHY201", "CS201", "EE201", "PHY_LAB", "CS_LAB"),
            lunch_slot=3,
        ),
        "EC-A": ClassBatch(
            name="EC-A",
            strength=40,
            subject_codes=("MATH201", "PHY201", "EE201", "CS202", "PHY_LAB"),
            lunch_slot=3,
        ),
    }

    rooms = {
        "LH-101": Room(name="LH-101", capacity=60, type="classroom"),
        "LH-102": Room(name="LH-102", capacity=50, type="classroom"),
        "CS-LAB": Room(name="CS-LAB", capacity=45, type="lab"),
        "PH-LAB": Room(name="PH-LAB", capacity=45, type="lab"),
    }

    return Problem(
        name="small",
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Large dataset: 8 classes, 15 teachers, 8 rooms, 5 days × 7 periods
# Target: ~60–75 % room-slot utilisation
# ---------------------------------------------------------------------------

def build_large(seed: int = SEED) -> Problem:  # noqa: C901
    rng = _rng(seed + 1)

    time = TimeStructure(
        working_days=5,
        periods_per_day=7,
        break_periods=frozenset({3}),   # lunch break
    )

    subjects = {
        # Theory subjects
        "MATH201": Subject("MATH201", "Mathematics II",        weekly_hours=3, type="theory"),
        "MATH202": Subject("MATH202", "Discrete Mathematics",  weekly_hours=2, type="theory"),
        "PHY201":  Subject("PHY201",  "Engineering Physics",   weekly_hours=2, type="theory"),
        "CS201":   Subject("CS201",   "Data Structures",       weekly_hours=3, type="theory"),
        "CS202":   Subject("CS202",   "OOP with Java",         weekly_hours=2, type="theory"),
        "CS203":   Subject("CS203",   "Computer Architecture", weekly_hours=2, type="theory"),
        "CS204":   Subject("CS204",   "Operating Systems",     weekly_hours=2, type="theory"),
        "EE201":   Subject("EE201",   "Basic Electronics",     weekly_hours=2, type="theory"),
        "EE202":   Subject("EE202",   "Circuit Theory",        weekly_hours=2, type="theory"),
        "AI201":   Subject("AI201",   "Introduction to AI",    weekly_hours=2, type="theory"),
        "AI202":   Subject("AI202",   "Machine Learning",      weekly_hours=2, type="theory"),
        # Lab subjects
        "PHY_LAB": Subject("PHY_LAB", "Physics Lab",           weekly_hours=1, type="lab"),
        "CS_LAB1": Subject("CS_LAB1", "Programming Lab I",     weekly_hours=1, type="lab"),
        "CS_LAB2": Subject("CS_LAB2", "DS Lab",                weekly_hours=1, type="lab"),
        "EE_LAB":  Subject("EE_LAB",  "Electronics Lab",       weekly_hours=1, type="lab"),
        "AI_LAB":  Subject("AI_LAB",  "AI Lab",                weekly_hours=1, type="lab"),
    }

    def unavail(n: int) -> frozenset:
        return _pick_unavailable(rng, 5, 7, {3}, n=n)

    teachers = {
        "Dr. Sharma":    Teacher("Dr. Sharma",    frozenset({"MATH201", "MATH202"}), unavail(2), 18, {"no_first_period": True}),
        "Dr. Patel":     Teacher("Dr. Patel",     frozenset({"PHY201"}),             unavail(1), 14),
        "Dr. Rao":       Teacher("Dr. Rao",       frozenset({"CS201", "CS202"}),     unavail(2), 18, {"no_first_period": True}),
        "Prof. Singh":   Teacher("Prof. Singh",   frozenset({"EE201", "EE202"}),     unavail(1), 16),
        "Dr. Mehta":     Teacher("Dr. Mehta",     frozenset({"PHY_LAB", "PHY201"}),  frozenset(), 12),
        "Prof. Kumar":   Teacher("Prof. Kumar",   frozenset({"CS_LAB1", "CS201"}),   unavail(1), 16),
        "Dr. Verma":     Teacher("Dr. Verma",     frozenset({"CS203", "CS204"}),     unavail(2), 18),
        "Dr. Gupta":     Teacher("Dr. Gupta",     frozenset({"AI201", "AI202"}),     unavail(1), 18, {"no_first_period": True}),
        "Prof. Joshi":   Teacher("Prof. Joshi",   frozenset({"MATH202", "MATH201"}), unavail(2), 16),
        "Dr. Bhat":      Teacher("Dr. Bhat",      frozenset({"EE_LAB", "EE201"}),    frozenset(), 12),
        "Prof. Iyer":    Teacher("Prof. Iyer",    frozenset({"CS_LAB2", "CS202"}),   unavail(1), 16),
        "Dr. Nair":      Teacher("Dr. Nair",      frozenset({"AI_LAB",  "AI201"}),   frozenset(), 10),
        "Prof. Reddy":   Teacher("Prof. Reddy",   frozenset({"CS204", "AI202"}),     unavail(1), 16),
        "Dr. Pillai":    Teacher("Dr. Pillai",    frozenset({"CS203", "EE202"}),     unavail(2), 14),
        "Prof. Das":     Teacher("Prof. Das",     frozenset({"MATH201", "AI201"}),   unavail(1), 14),
    }

    # 8 classes: 4 CS batches + 2 EC batches + 2 AI batches
    # Lunch slot alternates to avoid all batches sharing the same slot
    cs_subjects    = ("MATH201", "PHY201", "CS201", "CS202", "CS203", "PHY_LAB", "CS_LAB1", "CS_LAB2")
    ec_subjects    = ("MATH201", "PHY201", "EE201", "EE202", "CS202", "PHY_LAB", "EE_LAB")
    ai_subjects    = ("MATH202", "CS201", "CS204", "AI201", "AI202", "CS_LAB2", "AI_LAB")

    classes = {
        "CS-A": ClassBatch("CS-A", 42, cs_subjects, lunch_slot=3),
        "CS-B": ClassBatch("CS-B", 38, cs_subjects, lunch_slot=3),
        "CS-C": ClassBatch("CS-C", 40, cs_subjects, lunch_slot=3),
        "CS-D": ClassBatch("CS-D", 35, cs_subjects, lunch_slot=3),
        "EC-A": ClassBatch("EC-A", 44, ec_subjects, lunch_slot=3),
        "EC-B": ClassBatch("EC-B", 36, ec_subjects, lunch_slot=3),
        "AI-A": ClassBatch("AI-A", 38, ai_subjects, lunch_slot=3),
        "AI-B": ClassBatch("AI-B", 34, ai_subjects, lunch_slot=3),
    }

    rooms = {
        "LH-101": Room("LH-101", 60, "classroom"),
        "LH-102": Room("LH-102", 55, "classroom"),
        "LH-103": Room("LH-103", 50, "classroom"),
        "LH-104": Room("LH-104", 50, "classroom"),
        "CS-LAB1": Room("CS-LAB1", 48, "lab"),
        "CS-LAB2": Room("CS-LAB2", 48, "lab"),
        "PH-LAB":  Room("PH-LAB",  48, "lab"),
        "EE-LAB":  Room("EE-LAB",  48, "lab"),
    }

    return Problem(
        name="large",
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
        seed=seed,
    )


def build_scaled(n_classes: int, seed: int = SEED) -> Problem:
    """Generate a solvable Problem instance scaled to *n_classes* classes.

    Used by the scaling experiment to evaluate performance as problem size grows.
    """
    if n_classes == 3:
        return build_small(seed=seed)
    if n_classes == 8:
        return build_large(seed=seed)

    rng = _rng(seed)
    periods = 7
    days = 5
    time = TimeStructure(working_days=days, periods_per_day=periods, break_periods=frozenset({3}))

    subjects = {
        "MATH201": Subject("MATH201", "Mathematics II", 3, type="theory"),
        "MATH202": Subject("MATH202", "Discrete Mathematics", 2, type="theory"),
        "PHY201": Subject("PHY201", "Engineering Physics", 2, type="theory"),
        "CS201": Subject("CS201", "Data Structures", 3, type="theory"),
        "CS202": Subject("CS202", "OOP with Java", 2, type="theory"),
        "CS203": Subject("CS203", "Computer Architecture", 2, type="theory"),
        "EE201": Subject("EE201", "Basic Electronics", 2, type="theory"),
        "AI201": Subject("AI201", "Intro to AI", 2, type="theory"),
        "PHY_LAB": Subject("PHY_LAB", "Physics Lab", 1, type="lab"),
        "CS_LAB1": Subject("CS_LAB1", "Programming Lab", 1, type="lab"),
        "EE_LAB": Subject("EE_LAB", "Electronics Lab", 1, type="lab"),
    }

    class_names = [f"CLS-{i+1}" for i in range(n_classes)]
    sub_sets = [
        ("MATH201", "PHY201", "CS201", "CS202", "PHY_LAB", "CS_LAB1"),
        ("MATH201", "CS201", "CS203", "EE201", "PHY_LAB", "EE_LAB"),
        ("MATH202", "PHY201", "CS202", "AI201", "CS_LAB1", "EE_LAB"),
    ]
    classes = {}
    for i, cname in enumerate(class_names):
        classes[cname] = ClassBatch(
            name=cname,
            strength=36 + (i % 6),
            subject_codes=sub_sets[i % len(sub_sets)],
            lunch_slot=3,
        )

    n_classrooms = max(2, int(round(n_classes * 0.55)))
    n_labs = max(2, int(round(n_classes * 0.55)))
    rooms = {}
    for i in range(n_classrooms):
        rooms[f"LH-{101+i}"] = Room(f"LH-{101+i}", 55, "classroom")
    for i in range(n_labs):
        rooms[f"LAB-{1+i}"] = Room(f"LAB-{1+i}", 45, "lab")

    n_teachers = max(6, int(round(n_classes * 1.9)))
    subject_keys = list(subjects.keys())
    teacher_subs = {f"Dr. Teacher_{i+1}": set() for i in range(n_teachers)}

    for s_idx, code in enumerate(subject_keys):
        t1 = f"Dr. Teacher_{(s_idx % n_teachers) + 1}"
        t2 = f"Dr. Teacher_{((s_idx + 1) % n_teachers) + 1}"
        teacher_subs[t1].add(code)
        teacher_subs[t2].add(code)

    teachers = {}
    for i in range(n_teachers):
        tname = f"Dr. Teacher_{i+1}"
        tsubs = frozenset(teacher_subs[tname])
        unavail = frozenset(rng.sample([(d, p) for d in range(days) for p in range(periods) if p != 3], min(2, i % 3)))
        teachers[tname] = Teacher(
            name=tname,
            teachable_subjects=tsubs,
            unavailable_slots=unavail,
            max_weekly_load=18,
        )

    return Problem(
        name=f"scaled_{n_classes}",
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Utilisation report
# ---------------------------------------------------------------------------

def utilisation_report(problem: Problem) -> str:
    t = problem.time
    teaching_periods = [p for p in range(t.periods_per_day) if p not in t.break_periods]
    teaching_set = set(teaching_periods)

    classroom_rooms = [r for r in problem.rooms.values() if r.type == "classroom"]
    lab_rooms       = [r for r in problem.rooms.values() if r.type == "lab"]

    total_classroom_slots = len(classroom_rooms) * t.working_days * len(teaching_periods)
    lab_starts            = [p for p in teaching_periods if (p + 1) in teaching_set]
    total_lab_slots       = len(lab_rooms) * t.working_days * len(lab_starts)

    req_theory = req_lab = 0
    for cls in problem.classes.values():
        for code in cls.subject_codes:
            if code not in problem.subjects:
                continue
            subj = problem.subjects[code]
            if subj.type == "theory":
                req_theory += subj.weekly_hours
            else:
                req_lab += subj.weekly_hours

    theory_util = req_theory / total_classroom_slots * 100 if total_classroom_slots else 0
    lab_util    = req_lab    / total_lab_slots       * 100 if total_lab_slots       else 0

    return (
        f"  Theory:  {req_theory:3d} sessions / {total_classroom_slots:3d} classroom-slots  "
        f"= {theory_util:5.1f}% utilisation\n"
        f"  Lab:     {req_lab:3d} sessions / {total_lab_slots:3d} lab-slots        "
        f"= {lab_util:5.1f}% utilisation"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("small", build_small, data_dir / "small.json"),
        ("large", build_large, data_dir / "large.json"),
    ]

    all_ok = True
    for label, builder, out_path in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {label.upper()}")
        print(f"{'='*60}")
        problem = builder(seed=SEED)

        errors = validate_problem(problem)
        if errors:
            print(f"  [FAIL] Validation errors:")
            for e in errors:
                print(f"    • {e}")
            all_ok = False
        else:
            print("  [OK] Validation passed.")

        print(utilisation_report(problem))

        save_problem_to_json(problem, out_path)
        print(f"  Saved -> {out_path}")

    print()
    if not all_ok:
        print("Some datasets failed validation. Fix errors before proceeding.")
        sys.exit(1)
    else:
        print("All datasets generated and validated successfully.")


if __name__ == "__main__":
    main()
