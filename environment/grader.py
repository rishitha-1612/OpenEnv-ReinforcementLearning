from __future__ import annotations

from typing import Iterable

from environment.models import ActionType
from environment.tasks import TASKS


HARMFUL_ACTIONS: dict[str, set[ActionType]] = {
    "cardiac_arrest_easy": {
        ActionType.WAIT,
        ActionType.PLACE_RECOVERY_POSITION,
        ActionType.APPLY_PRESSURE,
    },
    "severe_bleeding_medium": {
        ActionType.START_CPR,
        ActionType.USE_AED,
        ActionType.WAIT,
    },
    "road_accident_hard": {
        ActionType.WAIT,
        ActionType.PLACE_RECOVERY_POSITION,
        ActionType.USE_AED,
    },
}


class EmergencyTaskGrader:
    def grade_task(self, task_id: str, actions: Iterable[ActionType | str]) -> float:
        task = TASKS[task_id]
        normalized = [action if isinstance(action, ActionType) else ActionType(action) for action in actions]
        optimal = task.optimal_sequence
        if not normalized:
            return 0.0

        matched_positions = sum(
            1 for index, action in enumerate(normalized[: len(optimal)]) if action == optimal[index]
        )
        unique_critical_hits = len(set(normalized).intersection(set(optimal)))
        harmful_count = sum(1 for action in normalized if action in HARMFUL_ACTIONS[task_id])
        duplicate_count = max(0, len(normalized) - len(set(normalized)))

        correctness = 0.6 * (matched_positions / len(optimal)) + 0.2 * (
            unique_critical_hits / len(optimal)
        )
        efficiency = 0.15 * min(1.0, len(optimal) / len(normalized))
        safety = max(0.0, 0.05 - 0.03 * harmful_count - 0.01 * duplicate_count)

        return round(max(0.0, min(1.0, correctness + efficiency + safety)), 4)

    def grade_easy(self, actions: Iterable[ActionType | str]) -> float:
        return self.grade_task("cardiac_arrest_easy", actions)

    def grade_medium(self, actions: Iterable[ActionType | str]) -> float:
        return self.grade_task("severe_bleeding_medium", actions)

    def grade_hard(self, actions: Iterable[ActionType | str]) -> float:
        return self.grade_task("road_accident_hard", actions)
