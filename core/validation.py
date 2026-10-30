"""core/validation.py — Input validation with human-readable error messages.

All checks are collected into a list of strings so callers can display all
problems at once rather than raising on the first one.
"""

from __future__ import annotations

from typing import List

from core.models import Problem


def validate_problem(problem: Problem) -> List[str]:
    """Run all validation checks on *problem* and return a list of error strings.

    An empty list means the problem is valid.

    Args:
        problem: The :class:`~core.models.Problem` to validate.

    Returns:
        A (possibly empty) list of human-readable error descriptions.
    """
    errors: List[str] = []
    errors.extend(_check_non_empty(problem))
    errors.extend(_check_duplicate_names(problem))
    errors.extend(_check_unknown_subject_codes(problem))
    errors.extend(_check_teacher_coverage(problem))
    errors.extend(_check_room_capacity(problem))
    errors.extend(_check_time_structure(problem))
    errors.extend(_check_solvability(problem))
    return errors


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_non_empty(p: Problem) -> List[str]:
    errors = []
    if not p.subjects:
        errors.append("No subjects defined.")
    if not p.teachers:
        errors.append("No teachers defined.")
    if not p.classes:
        errors.append("No classes defined.")
    if not p.rooms:
        errors.append("No rooms defined.")
    return errors


def _check_duplicate_names(p: Problem) -> List[str]:
    """Check that each entity type has unique identifiers."""
    errors: List[str] = []

    seen_subjects: set = set()
    for code in p.subjects:
        if code in seen_subjects:
            errors.append(f"Duplicate subject code: '{code}'.")
        seen_subjects.add(code)

    seen_teachers: set = set()
    for name in p.teachers:
        if name in seen_teachers:
            errors.append(f"Duplicate teacher name: '{name}'.")
        seen_teachers.add(name)

    seen_classes: set = set()
    for name in p.classes:
        if name in seen_classes:
            errors.append(f"Duplicate class name: '{name}'.")
        seen_classes.add(name)

    seen_rooms: set = set()
    for name in p.rooms:
        if name in seen_rooms:
            errors.append(f"Duplicate room name: '{name}'.")
        seen_rooms.add(name)

    return errors


def _check_unknown_subject_codes(p: Problem) -> List[str]:
    """Verify that every subject code referenced by classes and teachers exists."""
    errors: List[str] = []
    known = set(p.subjects)

    for cls in p.classes.values():
        for code in cls.subject_codes:
            if code not in known:
                errors.append(
                    f"Class '{cls.name}' references unknown subject code '{code}'."
                )

    for teacher in p.teachers.values():
        for code in teacher.teachable_subjects:
            if code not in known:
                errors.append(
                    f"Teacher '{teacher.name}' lists unknown subject code '{code}'."
                )

    return errors


def _check_teacher_coverage(p: Problem) -> List[str]:
    """Every subject required by at least one class must have ≥1 eligible teacher."""
    errors: List[str] = []
    required_codes = {code for cls in p.classes.values() for code in cls.subject_codes}

    for code in required_codes:
        eligible = [
            t.name for t in p.teachers.values() if code in t.teachable_subjects
        ]
        if not eligible:
            subj_name = p.subjects[code].name if code in p.subjects else code
            errors.append(
                f"Subject '{subj_name}' ({code}) has no eligible teacher."
            )
    return errors


def _check_room_capacity(p: Problem) -> List[str]:
    """Every class must fit in at least one room of the appropriate type."""
    errors: List[str] = []
    for cls in p.classes.values():
        # Check theory rooms
        theory_rooms = [r for r in p.rooms.values() if r.type == "classroom"]
        if theory_rooms and cls.strength > max(r.capacity for r in theory_rooms):
            errors.append(
                f"Class '{cls.name}' (strength {cls.strength}) is larger than "
                f"every classroom. Largest classroom: "
                f"{max(r.capacity for r in theory_rooms)} seats."
            )
        # Check lab rooms (only if the class has lab subjects)
        has_lab = any(
            p.subjects[code].type == "lab"
            for code in cls.subject_codes
            if code in p.subjects
        )
        if has_lab:
            lab_rooms = [r for r in p.rooms.values() if r.type == "lab"]
            if lab_rooms and cls.strength > max(r.capacity for r in lab_rooms):
                errors.append(
                    f"Class '{cls.name}' (strength {cls.strength}) is larger than "
                    f"every lab room. Largest lab: "
                    f"{max(r.capacity for r in lab_rooms)} seats."
                )
    return errors


def _check_time_structure(p: Problem) -> List[str]:
    """Sanity-check the TimeStructure values."""
    errors: List[str] = []
    t = p.time
    if t.working_days < 1:
        errors.append("working_days must be at least 1.")
    if t.periods_per_day < 1:
        errors.append("periods_per_day must be at least 1.")
    for bp in t.break_periods:
        if bp < 0 or bp >= t.periods_per_day:
            errors.append(
                f"Break period {bp} is out of range "
                f"[0, {t.periods_per_day - 1}]."
            )
    for cls in p.classes.values():
        if cls.lunch_slot < 0 or cls.lunch_slot >= t.periods_per_day:
            errors.append(
                f"Class '{cls.name}' lunch_slot {cls.lunch_slot} is out of range "
                f"[0, {t.periods_per_day - 1}]."
            )
    return errors


def _check_solvability(p: Problem) -> List[str]:
    """Coarse feasibility: total required sessions ≤ available room-slots.

    Checked separately for theory (classroom) and lab room types.
    This is a necessary (not sufficient) condition for solvability.
    """
    errors: List[str] = []
    t = p.time
    teaching_periods = [
        per for per in range(t.periods_per_day) if per not in t.break_periods
    ]
    # For labs, two consecutive periods are needed, so usable lab starts are
    # periods where period+1 is also a teaching period (and not a break).
    teaching_set = set(teaching_periods)

    # Count available slots per room type
    classroom_rooms = [r for r in p.rooms.values() if r.type == "classroom"]
    lab_rooms = [r for r in p.rooms.values() if r.type == "lab"]

    classroom_slots = len(classroom_rooms) * t.working_days * len(teaching_periods)
    lab_slot_starts = [
        per for per in teaching_periods if (per + 1) in teaching_set
    ]
    lab_slots = len(lab_rooms) * t.working_days * len(lab_slot_starts)

    # Count required sessions
    required_theory = 0
    required_lab = 0
    for cls in p.classes.values():
        for code in cls.subject_codes:
            if code not in p.subjects:
                continue
            subj = p.subjects[code]
            if subj.type == "theory":
                required_theory += subj.weekly_hours
            else:
                required_lab += subj.weekly_hours

    if required_theory > classroom_slots:
        errors.append(
            f"Solvability check failed: {required_theory} theory sessions required "
            f"but only {classroom_slots} classroom-slots available "
            f"({len(classroom_rooms)} rooms × {t.working_days} days × "
            f"{len(teaching_periods)} teaching periods)."
        )
    if required_lab > lab_slots:
        errors.append(
            f"Solvability check failed: {required_lab} lab sessions required "
            f"but only {lab_slots} lab-slots available "
            f"({len(lab_rooms)} rooms × {t.working_days} days × "
            f"{len(lab_slot_starts)} valid lab-start periods)."
        )
    return errors
