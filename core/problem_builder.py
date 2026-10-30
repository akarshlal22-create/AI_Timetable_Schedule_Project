"""core/problem_builder.py — Convert a Problem into a CSP representation.

Provides:
  - ``build_sessions``:      create all Session variables.
  - ``build_domains``:       compute valid (day, period, room) triples per session.
  - ``assign_teachers``:     balance teacher load across eligible teachers.
  - ``build_constraint_graph``: map each session to its neighbours (shared
                               teacher, class, or room domain).
  - ``build_csp``:           convenience wrapper returning everything at once.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

from core.models import (
    AssignmentTuple, ClassBatch, Problem, Room, Session, SlotTuple, Subject,
    Teacher,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sessions(problem: Problem) -> List[Session]:
    """Create one Session variable per required lecture/lab meeting.

    Teacher assignment is deferred — use :func:`assign_teachers` afterwards.
    This function returns sessions with a placeholder teacher name of ``""``.

    Args:
        problem: A validated :class:`~core.models.Problem`.

    Returns:
        Ordered list of :class:`~core.models.Session` objects.
    """
    sessions: List[Session] = []
    for cls in problem.classes.values():
        for code in cls.subject_codes:
            if code not in problem.subjects:
                continue
            subj = problem.subjects[code]
            is_lab = subj.type == "lab"
            for idx in range(subj.weekly_hours):
                sessions.append(
                    Session(
                        class_name=cls.name,
                        subject_code=code,
                        session_index=idx,
                        teacher_name="",   # filled by assign_teachers
                        is_lab=is_lab,
                    )
                )
    return sessions


def assign_teachers(
    sessions: List[Session],
    problem: Problem,
    rng: random.Random | None = None,
) -> List[Session]:
    """Fix one teacher per session, balancing load across eligible teachers.

    Within each (class, subject) group, the same teacher is always used so
    that a class does not see multiple teachers for the same subject.
    Load is balanced by choosing the eligible teacher with the lowest current
    assignment count.

    Args:
        sessions: Sessions with ``teacher_name == ""``.
        problem:  The Problem (used to look up eligible teachers).
        rng:      Optional seeded :class:`random.Random` for tie-breaking.

    Returns:
        New list of :class:`~core.models.Session` with ``teacher_name`` filled.

    Raises:
        ValueError: If no eligible teacher exists for a session's subject.
    """
    if rng is None:
        rng = random.Random(problem.seed)

    # teacher_load tracks how many sessions each teacher has been assigned
    teacher_load: Dict[str, int] = {name: 0 for name in problem.teachers}

    # Cache: (class_name, subject_code) -> chosen teacher name
    chosen_teacher: Dict[Tuple[str, str], str] = {}

    result: List[Session] = []
    for sess in sessions:
        key = (sess.class_name, sess.subject_code)
        if key not in chosen_teacher:
            # Check if class explicitly specifies teacher_for
            class_obj = problem.classes.get(sess.class_name)
            explicit_teacher = class_obj.teacher_for.get(sess.subject_code) if class_obj else None

            if explicit_teacher and explicit_teacher in problem.teachers:
                chosen_teacher[key] = explicit_teacher
            else:
                eligible = [
                    t for t in problem.teachers.values()
                    if sess.subject_code in t.teachable_subjects
                ]
                if not eligible:
                    raise ValueError(
                        f"No eligible teacher for subject '{sess.subject_code}' "
                        f"in class '{sess.class_name}'."
                    )
                # Sort by load then name for determinism, shuffle ties
                min_load = min(teacher_load[t.name] for t in eligible)
                candidates = [t for t in eligible if teacher_load[t.name] == min_load]
                rng.shuffle(candidates)
                chosen = candidates[0]
                chosen_teacher[key] = chosen.name

        teacher_name = chosen_teacher[key]
        teacher_load[teacher_name] += 1

        result.append(
            Session(
                class_name=sess.class_name,
                subject_code=sess.subject_code,
                session_index=sess.session_index,
                teacher_name=teacher_name,
                is_lab=sess.is_lab,
            )
        )
    return result


def build_domains(
    sessions: List[Session],
    problem: Problem,
) -> Dict[Session, List[AssignmentTuple]]:
    """Compute the domain for each session after applying unary constraints.

    Unary filters applied:
    - **H5** Room type must match subject type (classroom ↔ theory, lab ↔ lab).
    - **H4** Room capacity ≥ class strength.
    - **H6** Slot is not in teacher's unavailable_slots.
    - **H9** Slot is not the class's lunch_slot.
    - Break periods are excluded globally.
    - For lab sessions, the slot ``(day, period)`` is valid only when
      ``period + 1`` is also a teaching period (not a break, not out of range).

    Args:
        sessions: Sessions with teachers already assigned.
        problem:  The Problem supplying rooms, time, classes, teachers.

    Returns:
        Dict mapping each :class:`~core.models.Session` to a list of
        ``(day, period, room_name)`` triples.
    """
    t = problem.time
    teaching_periods: List[int] = t.teaching_periods()
    teaching_set: Set[int] = set(teaching_periods)

    domains: Dict[Session, List[AssignmentTuple]] = {}

    for sess in sessions:
        cls = problem.classes[sess.class_name]
        subj = problem.subjects[sess.subject_code]
        teacher = problem.teachers[sess.teacher_name]
        room_type_needed = "lab" if sess.is_lab else "classroom"

        eligible_rooms = [
            r for r in problem.rooms.values()
            if r.type == room_type_needed and r.capacity >= cls.strength
        ]

        valid: List[AssignmentTuple] = []
        for day in range(t.working_days):
            for period in teaching_periods:
                # Lunch slot check (H9)
                if period == cls.lunch_slot:
                    continue
                # Teacher availability (H6)
                if (day, period) in teacher.unavailable_slots:
                    continue
                # Lab needs period+1 to be a valid teaching period (not break, not overflow)
                if sess.is_lab:
                    if (period + 1) not in teaching_set or (period + 1) >= t.periods_per_day:
                        continue
                    # Also check lunch and teacher unavailability for the second period
                    if (period + 1) == cls.lunch_slot:
                        continue
                    if (day, period + 1) in teacher.unavailable_slots:
                        continue
                for room in eligible_rooms:
                    valid.append((day, period, room.name))

        domains[sess] = valid

    return domains


def build_constraint_graph(
    sessions: List[Session],
) -> Dict[Session, List[Session]]:
    """Return the constraint graph: session → list of sessions it conflicts with.

    Two sessions conflict (are neighbours) when they share a teacher, a class,
    or a room domain (same room type — actual room clash is checked at runtime).
    The graph is symmetric.

    Args:
        sessions: All :class:`~core.models.Session` objects in the problem.

    Returns:
        Dict mapping each session to its list of neighbour sessions.
    """
    by_class: Dict[str, List[Session]] = defaultdict(list)
    by_teacher: Dict[str, List[Session]] = defaultdict(list)

    for sess in sessions:
        by_class[sess.class_name].append(sess)
        by_teacher[sess.teacher_name].append(sess)

    neighbours: Dict[Session, Set[Session]] = {s: set() for s in sessions}

    for group in by_class.values():
        for i, s1 in enumerate(group):
            for s2 in group[i + 1 :]:
                neighbours[s1].add(s2)
                neighbours[s2].add(s1)

    for group in by_teacher.values():
        for i, s1 in enumerate(group):
            for s2 in group[i + 1 :]:
                neighbours[s1].add(s2)
                neighbours[s2].add(s1)

    return {s: list(nb) for s, nb in neighbours.items()}


def build_csp(
    problem: Problem,
    rng: random.Random | None = None,
) -> Tuple[
    List[Session],
    Dict[Session, List[AssignmentTuple]],
    Dict[Session, List[Session]],
]:
    """Build the full CSP from a Problem.

    Convenience wrapper that calls :func:`build_sessions`,
    :func:`assign_teachers`, :func:`build_domains`, and
    :func:`build_constraint_graph` in the correct order.

    Args:
        problem: A validated :class:`~core.models.Problem`.
        rng:     Optional seeded :class:`random.Random`.

    Returns:
        ``(sessions, domains, constraint_graph)`` tuple ready for a solver.
    """
    sessions = build_sessions(problem)
    sessions = assign_teachers(sessions, problem, rng=rng)
    domains = build_domains(sessions, problem)
    graph = build_constraint_graph(sessions)
    return sessions, domains, graph
