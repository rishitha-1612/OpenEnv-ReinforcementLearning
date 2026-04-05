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
    "anaphylaxis_medium": {
        ActionType.WAIT,
        ActionType.START_CPR,
        ActionType.USE_AED,
    },
    "choking_easy": {
        ActionType.WAIT,
        ActionType.APPLY_PRESSURE,
        ActionType.USE_AED,
    },
}


class EmergencyTaskGrader:
    def grade_task(self, task_id: str, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        if task_id == "cardiac_arrest_easy":
            return self.grade_easy(normalized)
        if task_id == "severe_bleeding_medium":
            return self.grade_medium(normalized)
        if task_id == "road_accident_hard":
            return self.grade_hard(normalized)
        if task_id == "anaphylaxis_medium":
            return self.grade_anaphylaxis(normalized)
        if task_id == "choking_easy":
            return self.grade_choking(normalized)
        raise KeyError(f"Unknown task_id: {task_id}")

    def grade_easy(self, actions: Iterable[ActionType | str]) -> float:
        return self._base_grade("cardiac_arrest_easy", self._normalize(actions))

    def grade_medium(self, actions: Iterable[ActionType | str]) -> float:
        return self._base_grade("severe_bleeding_medium", self._normalize(actions))

    def grade_hard(self, actions: Iterable[ActionType | str]) -> float:
        return self._base_grade("road_accident_hard", self._normalize(actions))

    def grade_anaphylaxis(self, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        score = self._base_grade("anaphylaxis_medium", normalized)
        if ActionType.CONTROL_AIRWAY in normalized:
            airway_index = normalized.index(ActionType.CONTROL_AIRWAY)
            if ActionType.CHECK_BREATHING not in normalized[:airway_index]:
                score -= 0.2
        return round(max(0.0, min(1.0, score)), 4)

    def grade_choking(self, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        score = self._base_grade("choking_easy", normalized)
        wait_count = sum(1 for action in normalized if action == ActionType.WAIT)
        score -= 0.12 * wait_count
        return round(max(0.0, min(1.0, score)), 4)

    def _base_grade(self, task_id: str, actions: list[ActionType]) -> float:
        task = TASKS[task_id]
        optimal = task.optimal_sequence
        if not actions:
            return 0.0

        matched_positions = sum(
            1 for index, action in enumerate(actions[: len(optimal)]) if action == optimal[index]
        )
        unique_critical_hits = len(set(actions).intersection(set(optimal)))
        harmful_count = sum(1 for action in actions if action in HARMFUL_ACTIONS[task_id])
        duplicate_count = max(0, len(actions) - len(set(actions)))

        correctness = 0.6 * (matched_positions / len(optimal)) + 0.2 * (
            unique_critical_hits / len(optimal)
        )
        efficiency = 0.15 * min(1.0, len(optimal) / len(actions))
        safety = max(0.0, 0.05 - 0.03 * harmful_count - 0.01 * duplicate_count)

        return round(max(0.0, min(1.0, correctness + efficiency + safety)), 4)

    def _normalize(self, actions: Iterable[ActionType | str]) -> list[ActionType]:
        return [action if isinstance(action, ActionType) else ActionType(action) for action in actions]
