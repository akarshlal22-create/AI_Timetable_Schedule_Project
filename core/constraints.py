"""core/constraints.py — Hard and soft constraint checks for a Timetable.

Each constraint is a small standalone function that accepts a Timetable and
returns True (satisfied) or False (violated), or a penalty integer for soft
constraints. Functions are intentionally short so violations can be counted
and explained individually.

Hard constraints H1–H9:
  H1  No teacher clash
  H2  No class clash
  H3  No room clash
  H4  Room capacity ≥ class strength
  H5  Room type matches subject type
  H6  Teacher not scheduled in unavailable slot
  H7  Every subject gets exactly its weekly_hours sessions
  H8  Teacher total load ≤ max_weekly_load
  H9  Nothing scheduled in a class's lunch slot

Soft constraints S1–S5:
  S1  Minimise gaps in a class's day
  S2  Spread subject sessions across different days (labs exempt)
  S3  Lab sessions must occupy two consecutive periods (penalty if not satisfied)
  S4  Respect teacher preferences (e.g. no_first_period)
  S5  No more than 2 consecutive lectures for a teacher
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from core.models import Session, Timetable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slots_of(timetable: Timetable, session: Session) -> List[Tuple[int, int]]:
    """Return all (day, period) pairs occupied by *session*.

    Labs occupy (day, period) and (day, period+1).
    Returns [] when the session is unassigned.
    """
    return timetable.slots_occupied_by(session)


# ---------------------------------------------------------------------------
# Hard constraints (return True = OK, False = violated)
# ---------------------------------------------------------------------------

def check_h1_teacher_clash(timetable: Timetable) -> bool:
    """H1: A teacher must not teach two sessions at the same (day, period)."""
    by_teacher: dict = defaultdict(list)
    for sess, (day, period, _) in timetable.assignment.items():
        for slot in timetable.slots_occupied_by(sess):
            by_teacher[sess.teacher_name].append(slot)
    for teacher, slots in by_teacher.items():
        if len(slots) != len(set(slots)):
            return False
    return True


def check_h2_class_clash(timetable: Timetable) -> bool:
    """H2: A class must not have two sessions at the same (day, period)."""
    by_class: dict = defaultdict(list)
    for sess in timetable.assignment:
        for slot in timetable.slots_occupied_by(sess):
            by_class[sess.class_name].append(slot)
    for cls, slots in by_class.items():
        if len(slots) != len(set(slots)):
            return False
    return True


def check_h3_room_clash(timetable: Timetable) -> bool:
    """H3: A room must not host two sessions at the same (day, period)."""
    by_room: dict = defaultdict(list)
    for sess, (day, period, room) in timetable.assignment.items():
        for slot in timetable.slots_occupied_by(sess):
            by_room[room].append(slot)
    for room, slots in by_room.items():
        if len(slots) != len(set(slots)):
            return False
    return True


def check_h4_room_capacity(timetable: Timetable) -> bool:
    """H4: Every assigned room must have capacity ≥ class strength."""
    p = timetable.problem
    for sess, (_, _, room_name) in timetable.assignment.items():
        cls = p.classes[sess.class_name]
        room = p.rooms[room_name]
        if room.capacity < cls.strength:
            return False
    return True


def check_h5_room_type(timetable: Timetable) -> bool:
    """H5: Room type must match subject type (classroom↔theory, lab↔lab)."""
    p = timetable.problem
    for sess, (_, _, room_name) in timetable.assignment.items():
        subj = p.subjects[sess.subject_code]
        room = p.rooms[room_name]
        expected = "lab" if subj.type == "lab" else "classroom"
        if room.type != expected:
            return False
    return True


def check_h6_teacher_availability(timetable: Timetable) -> bool:
    """H6: No session placed in a teacher's unavailable slot."""
    p = timetable.problem
    for sess in timetable.assignment:
        teacher = p.teachers[sess.teacher_name]
        for slot in timetable.slots_occupied_by(sess):
            if slot in teacher.unavailable_slots:
                return False
    return True


def check_h7_subject_hours(timetable: Timetable) -> bool:
    """H7: Each subject for each class is scheduled exactly weekly_hours times."""
    p = timetable.problem
    # count assigned sessions per (class, subject)
    counts: dict = defaultdict(int)
    for sess in timetable.assignment:
        counts[(sess.class_name, sess.subject_code)] += 1
    for cls in p.classes.values():
        for code in cls.subject_codes:
            if code not in p.subjects:
                continue
            required = p.subjects[code].weekly_hours
            actual = counts.get((cls.name, code), 0)
            if actual != required:
                return False
    return True


def check_h8_teacher_load(timetable: Timetable) -> bool:
    """H8: No teacher exceeds their max_weekly_load."""
    p = timetable.problem
    load: dict = defaultdict(int)
    for sess in timetable.assignment:
        load[sess.teacher_name] += 1
    for teacher_name, count in load.items():
        if teacher_name not in p.teachers:
            continue
        if count > p.teachers[teacher_name].max_weekly_load:
            return False
    return True


def check_h9_lunch_slot(timetable: Timetable) -> bool:
    """H9: No session overlaps a class's lunch slot."""
    p = timetable.problem
    for sess in timetable.assignment:
        cls = p.classes[sess.class_name]
        for (day, period) in timetable.slots_occupied_by(sess):
            if period == cls.lunch_slot:
                return False
    return True


# ---------------------------------------------------------------------------
# Soft constraints (return int penalty; 0 = no violation)
# ---------------------------------------------------------------------------

def penalty_s1_class_gaps(timetable: Timetable) -> int:
    """S1: Penalty for idle gaps within a class's teaching day.

    A gap is a period between the first and last session of a day where
    the class has no teaching. Each gap period costs 1 penalty point.
    """
    p = timetable.problem
    total = 0
    for cls in p.classes.values():
        # Collect occupied periods per day for this class
        day_periods: dict = defaultdict(set)
        for sess, (day, period, _) in timetable.assignment.items():
            if sess.class_name != cls.name:
                continue
            for (d, per) in timetable.slots_occupied_by(sess):
                day_periods[d].add(per)
        for day, periods in day_periods.items():
            if not periods:
                continue
            first, last = min(periods), max(periods)
            # Gap = periods between first and last that are not occupied
            gap = (last - first + 1) - len(periods)
            total += max(0, gap)
    return total


def penalty_s2_subject_spread(timetable: Timetable) -> int:
    """S2: Penalty when a subject's sessions cluster on the same day.

    Theory subjects should appear on different days.  Labs are exempt.
    Cost = number of (class, subject) pairs where two sessions share a day.
    """
    p = timetable.problem
    total = 0
    # group by (class, subject)
    groups: dict = defaultdict(list)
    for sess, (day, period, _) in timetable.assignment.items():
        if sess.is_lab:
            continue
        groups[(sess.class_name, sess.subject_code)].append(day)
    for days in groups.values():
        # count duplicate day occurrences
        seen: set = set()
        for d in days:
            if d in seen:
                total += 1
            seen.add(d)
    return total


def penalty_s3_lab_consecutive(timetable: Timetable) -> int:
    """S3: Penalty when a lab session's two periods are not consecutive.

    Under the current domain construction labs already have consecutive
    periods enforced, so this serves as a safety net for manually-built
    or repaired timetables.  Each non-consecutive lab costs 5 points.
    """
    total = 0
    for sess, (day, period, _) in timetable.assignment.items():
        if not sess.is_lab:
            continue
        slots = timetable.slots_occupied_by(sess)
        if len(slots) < 2:
            total += 5
            continue
        (d1, p1), (d2, p2) = slots[0], slots[1]
        if d1 != d2 or p2 != p1 + 1:
            total += 5
    return total


def penalty_s4_teacher_preferences(timetable: Timetable) -> int:
    """S4: Penalty for violating optional teacher preferences.

    Supported preference keys:
    - ``no_first_period`` (bool): teacher prefers not to teach period 0.
    - ``preferred_days``  (list[int]): teacher prefers these day indices.
    Each violation costs 2 points.
    """
    p = timetable.problem
    total = 0
    for sess, (day, period, _) in timetable.assignment.items():
        teacher = p.teachers.get(sess.teacher_name)
        if teacher is None or not teacher.preferences:
            continue
        prefs = teacher.preferences
        if prefs.get("no_first_period") in (True, "True", "true", "1"):
            for (_, per) in timetable.slots_occupied_by(sess):
                if per == 0:
                    total += 2
        if "preferred_days" in prefs:
            raw = prefs["preferred_days"]
            pref_days = raw if isinstance(raw, list) else [int(x) for x in str(raw).split(",")]
            if day not in pref_days:
                total += 2
    return total


def penalty_s5_teacher_consecutive(timetable: Timetable) -> int:
    """S5: Penalty for more than 2 consecutive teaching periods for a teacher.

    Each period exceeding the run of 2 costs 3 penalty points.
    """
    p = timetable.problem
    total = 0
    by_teacher: dict = defaultdict(lambda: defaultdict(set))
    for sess in timetable.assignment:
        for (day, period) in timetable.slots_occupied_by(sess):
            by_teacher[sess.teacher_name][day].add(period)
    for teacher_name, days in by_teacher.items():
        for day, periods in days.items():
            sorted_periods = sorted(periods)
            run = 1
            for i in range(1, len(sorted_periods)):
                if sorted_periods[i] == sorted_periods[i - 1] + 1:
                    run += 1
                    if run > 2:
                        total += 3
                else:
                    run = 1
    return total
