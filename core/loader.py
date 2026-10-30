"""core/loader.py — Load and save Problem instances from/to JSON and CSV.

JSON schema mirrors the Problem dataclass hierarchy.
CSV mode expects one file per entity type (subjects.csv, teachers.csv, …).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Union

from core.models import (
    ClassBatch, Problem, Room, Subject, Teacher, TimeStructure,
)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_problem_from_json(path: Union[str, Path]) -> Problem:
    """Load a Problem from a JSON file produced by :func:`save_problem_to_json`.

    Args:
        path: Path to the ``.json`` file.

    Returns:
        A fully-populated :class:`Problem` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError / ValueError: If the JSON is malformed.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return _problem_from_dict(data)


def save_problem_to_json(problem: Problem, path: Union[str, Path]) -> None:
    """Serialise *problem* to a pretty-printed JSON file.

    Args:
        problem: The :class:`Problem` to save.
        path:    Destination file path (parent dirs must exist).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_problem_to_dict(problem), fh, indent=2)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_problem_from_csv(directory: Union[str, Path], name: str = "problem") -> Problem:
    """Load a Problem from a directory containing CSV files.

    Expected files: ``subjects.csv``, ``teachers.csv``, ``classes.csv``,
    ``rooms.csv``, ``time_structure.csv``.

    Args:
        directory: Path to the directory holding the CSV files.
        name:      Human-readable label for the resulting Problem.

    Returns:
        A :class:`Problem` instance.
    """
    d = Path(directory)

    subjects = _load_subjects_csv(d / "subjects.csv")
    teachers = _load_teachers_csv(d / "teachers.csv")
    classes = _load_classes_csv(d / "classes.csv")
    rooms = _load_rooms_csv(d / "rooms.csv")
    time = _load_time_csv(d / "time_structure.csv")

    return Problem(
        name=name,
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
    )


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------

def _problem_to_dict(p: Problem) -> dict:
    return {
        "name": p.name,
        "seed": p.seed,
        "time": {
            "working_days": p.time.working_days,
            "periods_per_day": p.time.periods_per_day,
            "break_periods": sorted(p.time.break_periods),
        },
        "subjects": [
            {
                "code": s.code,
                "name": s.name,
                "weekly_hours": s.weekly_hours,
                "type": s.type,
            }
            for s in p.subjects.values()
        ],
        "teachers": [
            {
                "name": t.name,
                "teachable_subjects": sorted(t.teachable_subjects),
                "unavailable_slots": [list(sl) for sl in sorted(t.unavailable_slots)],
                "max_weekly_load": t.max_weekly_load,
                "preferences": t.preferences,
            }
            for t in p.teachers.values()
        ],
        "classes": [
            {
                "name": c.name,
                "strength": c.strength,
                "subject_codes": list(c.subject_codes),
                "lunch_slot": c.lunch_slot,
            }
            for c in p.classes.values()
        ],
        "rooms": [
            {
                "name": r.name,
                "capacity": r.capacity,
                "type": r.type,
            }
            for r in p.rooms.values()
        ],
    }


def _problem_from_dict(data: dict) -> Problem:
    """Parse a raw JSON dict into a :class:`Problem`.

    Accepts both list-of-objects and dict-of-objects for entity sections and
    normalises common alternative key names so that user-authored files do not
    need to match the internal schema exactly.  Raises :class:`ValueError` with
    an actionable message when a required field is absent.
    """
    # --- time ---
    if "time" not in data:
        raise ValueError(
            "Missing top-level key 'time'. Expected: "
            "{\"time\": {\"working_days\": 5, \"periods_per_day\": 6, \"break_periods\": [3]}, ...}"
        )
    time_data = data["time"]

    # Accept "days" as a list of day-names or an integer count.
    if "working_days" not in time_data:
        days_raw = time_data.get("days") or time_data.get("num_days")
        if days_raw is None:
            raise ValueError(
                "Missing key 'working_days' inside 'time'. "
                "Expected keys: working_days (int), periods_per_day (int), break_periods (list)."
            )
        working_days = len(days_raw) if isinstance(days_raw, list) else int(days_raw)
    else:
        working_days = int(time_data["working_days"])

    if "periods_per_day" not in time_data:
        raise ValueError(
            "Missing key 'periods_per_day' inside 'time'. "
            "Expected keys: working_days, periods_per_day, break_periods."
        )

    time = TimeStructure(
        working_days=working_days,
        periods_per_day=int(time_data["periods_per_day"]),
        break_periods=frozenset(time_data.get("break_periods", [])),
    )

    # --- helper: normalise list-or-dict sections ---
    def _as_list(section_name: str) -> list:
        raw = data.get(section_name)
        if raw is None:
            raise ValueError(f"Missing top-level key '{section_name}'.")
        if isinstance(raw, dict):
            return list(raw.values())
        if isinstance(raw, list):
            return raw
        raise ValueError(
            f"'{section_name}' must be a list or an object, got {type(raw).__name__}."
        )

    # --- subjects ---
    subjects: Dict[str, Subject] = {}
    for s in _as_list("subjects"):
        subj = Subject(
            code=s["code"],
            name=s["name"],
            weekly_hours=int(s["weekly_hours"]),
            type=s["type"],
        )
        subjects[subj.code] = subj

    # --- teachers ---
    # Accepts: teachable_subjects | subjects
    # Accepts: unavailable_slots  | unavailable
    # Accepts: max_weekly_load    | max_load
    teachers: Dict[str, Teacher] = {}
    for t in _as_list("teachers"):
        teachable = t.get("teachable_subjects") or t.get("subjects") or []
        unavailable = t.get("unavailable_slots") or t.get("unavailable") or []
        max_load = int(t.get("max_weekly_load") or t.get("max_load") or 20)
        teacher = Teacher(
            name=t["name"],
            teachable_subjects=frozenset(teachable),
            unavailable_slots=frozenset(tuple(sl) for sl in unavailable),
            max_weekly_load=max_load,
            preferences=t.get("preferences", {}),
        )
        teachers[teacher.name] = teacher

    # --- classes ---
    # Accepts: subject_codes | subjects
    # Accepts: lunch_slot    | lunch_period
    # Accepts: teacher_for   (optional explicit teacher mapping)
    classes: Dict[str, ClassBatch] = {}
    for c in _as_list("classes"):
        subject_codes = c.get("subject_codes") or c.get("subjects") or []
        lunch_slot = int(c.get("lunch_slot") if c.get("lunch_slot") is not None
                        else c.get("lunch_period", 0))
        batch = ClassBatch(
            name=c["name"],
            strength=int(c["strength"]),
            subject_codes=tuple(subject_codes),
            lunch_slot=lunch_slot,
            teacher_for=dict(c.get("teacher_for", {})),
        )
        classes[batch.name] = batch

    # --- rooms ---
    rooms: Dict[str, Room] = {}
    for r in _as_list("rooms"):
        room = Room(
            name=r["name"],
            capacity=int(r["capacity"]),
            type=r["type"],
        )
        rooms[room.name] = room

    return Problem(
        name=data.get("name", "problem"),
        seed=int(data.get("seed", 42)),
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
    )


# ---------------------------------------------------------------------------
# CSV loaders (one per entity type)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Day name mapping helper for CSVs
# ---------------------------------------------------------------------------

_DAY_NAME_TO_INDEX = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def _parse_slot_pair(token: str) -> Tuple[int, int]:
    """Parse '0,1' or 'Mon:1' or 'Monday,2' into a (day, period) tuple."""
    token = token.strip()
    sep = ":" if ":" in token else ","
    parts = token.split(sep, 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid slot format: '{token}'. Expected 'day,period' or 'Day:period'.")
    day_part, period_part = parts[0].strip(), parts[1].strip()

    # Parse day part
    if day_part.lower() in _DAY_NAME_TO_INDEX:
        day_idx = _DAY_NAME_TO_INDEX[day_part.lower()]
    else:
        day_idx = int(day_part)

    period_idx = int(period_part)
    return day_idx, period_idx


def _load_subjects_csv(path: Path) -> Dict[str, Subject]:
    subjects: Dict[str, Subject] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            # Normalise keys (lowercase and stripped)
            r = {k.strip().lower(): v for k, v in row.items() if k}
            code = r.get("code", "").strip()
            name = r.get("name", code).strip()
            hours = int(r.get("weekly_hours") or r.get("hours", 1))
            stype = r.get("type", "theory").strip().lower()
            s = Subject(code=code, name=name, weekly_hours=hours, type=stype)
            subjects[s.code] = s
    return subjects


def _load_teachers_csv(path: Path) -> Dict[str, Teacher]:
    teachers: Dict[str, Teacher] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            r = {k.strip().lower(): v for k, v in row.items() if k}
            name = r.get("name", "").strip()

            # subjects / teachable_subjects
            subj_raw = r.get("teachable_subjects") or r.get("subjects") or ""
            codes = frozenset(c.strip() for c in subj_raw.split("|") if c.strip())

            # unavailable slots: handles both "0,1|1,2" and "Mon:1|Fri:5"
            raw_slots = r.get("unavailable_slots") or r.get("unavailable") or ""
            slots: List[Tuple[int, int]] = []
            if raw_slots.strip():
                for pair in raw_slots.split("|"):
                    pair = pair.strip()
                    if pair:
                        slots.append(_parse_slot_pair(pair))

            # max weekly load
            max_load_val = r.get("max_weekly_load") or r.get("max_load") or 20
            max_load = int(max_load_val)

            # preferences: handles "k=v|k2=v2" or individual columns like no_first_period
            prefs: dict = {}
            prefs_raw = r.get("preferences", "").strip()
            if prefs_raw:
                for item in prefs_raw.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        prefs[k.strip()] = v.strip()

            # check explicit preference flags in columns (e.g. no_first_period)
            for pref_col in ("no_first_period", "avoid_periods"):
                if pref_col in r and r[pref_col].strip():
                    val = r[pref_col].strip()
                    if val.lower() in ("yes", "true", "1"):
                        prefs[pref_col] = True
                    elif val.lower() not in ("no", "false", "0"):
                        prefs[pref_col] = val

            t = Teacher(
                name=name,
                teachable_subjects=codes,
                unavailable_slots=frozenset(slots),
                max_weekly_load=max_load,
                preferences=prefs,
            )
            teachers[t.name] = t
    return teachers


def _load_classes_csv(path: Path) -> Dict[str, ClassBatch]:
    classes: Dict[str, ClassBatch] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            r = {k.strip().lower(): v for k, v in row.items() if k}
            name = r.get("name", "").strip()
            strength = int(r.get("strength", 30))

            # subject_codes or subjects
            subj_raw = r.get("subject_codes") or r.get("subjects") or ""
            codes = tuple(c.strip() for c in subj_raw.split("|") if c.strip())

            # lunch_slot or lunch_period
            lunch_val = r.get("lunch_slot") if r.get("lunch_slot") is not None else r.get("lunch_period", 0)
            lunch_slot = int(lunch_val)

            c = ClassBatch(
                name=name,
                strength=strength,
                subject_codes=codes,
                lunch_slot=lunch_slot,
            )
            classes[c.name] = c
    return classes


def _load_rooms_csv(path: Path) -> Dict[str, Room]:
    rooms: Dict[str, Room] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            r = {k.strip().lower(): v for k, v in row.items() if k}
            name = r.get("name", "").strip()
            capacity = int(r.get("capacity", 40))
            rtype = r.get("type", "classroom").strip().lower()
            room = Room(name=name, capacity=capacity, type=rtype)
            rooms[room.name] = room
    return rooms


def _load_time_csv(path: Path) -> TimeStructure:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("time_structure.csv is empty")
    row = {k.strip().lower(): v for k, v in rows[0].items() if k}

    # working_days or days
    if "working_days" in row and row["working_days"].strip():
        working_days = int(row["working_days"])
    elif "days" in row and row["days"].strip():
        days_val = row["days"].strip()
        if "|" in days_val:
            working_days = len([d for d in days_val.split("|") if d.strip()])
        elif days_val.isdigit():
            working_days = int(days_val)
        else:
            working_days = 5
    else:
        working_days = 5

    periods_per_day = int(row.get("periods_per_day", 6))

    breaks_raw = row.get("break_periods", "").strip()
    break_periods = (
        frozenset(int(x) for x in breaks_raw.split("|") if x.strip())
        if breaks_raw else frozenset()
    )
    return TimeStructure(
        working_days=working_days,
        periods_per_day=periods_per_day,
        break_periods=break_periods,
    )
