from __future__ import annotations

from copy import deepcopy
from typing import Any

from environment.grader import EmergencyTaskGrader
from environment.models import (
    Action,
    ActionType,
    AirwayStatus,
    BleedingSeverity,
    BreathingStatus,
    ConsciousStatus,
    InternalState,
    Observation,
    PulseStatus,
    RewardSignal,
)
from environment.tasks import DEFAULT_TASK_ID, TASKS


class EmergencyFirstResponseDecisionEngine:
    def __init__(self, default_task_id: str = DEFAULT_TASK_ID) -> None:
        self._grader = EmergencyTaskGrader()
        self._default_task_id = default_task_id
        self._state = self._build_state(default_task_id)

    def reset(self, task_id: str | None = None) -> Observation:
        selected_task = task_id or self._default_task_id
        if selected_task not in TASKS:
            raise ValueError(f"Unknown task_id: {selected_task}")
        self._state = self._build_state(selected_task)
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._state.done:
            observation = self._build_observation()
            return observation, 0.0, True, self._build_info(termination_locked=True)

        reward = -0.2
        reward_components: dict[str, float] = {"base_step_cost": -0.2}
        effect_messages: list[str] = []
        chosen_action = action.action_type
        prior_actions = list(self._state.actions_taken)

        if prior_actions and prior_actions[-1] == chosen_action:
            reward -= 0.6
            reward_components["repeat_penalty"] = -0.6
            effect_messages.append("Action repeated without reassessment.")

        action_delta = self._apply_action(chosen_action, effect_messages)
        reward += action_delta
        reward_components["action_delta"] = action_delta
        self._state.actions_taken.append(chosen_action)
        self._state.time_elapsed += 1

        progression_delta = self._apply_progression(effect_messages)
        reward += progression_delta
        reward_components["progression_delta"] = progression_delta
        self._update_observed_condition()
        self._check_termination()

        if self._state.done:
            terminal_delta = 5.0 if self._state.success else -5.0
            reward += terminal_delta
            reward_components["terminal_delta"] = terminal_delta

        self._state.last_action_effect = " ".join(effect_messages) if effect_messages else "No meaningful change."
        observation = self._build_observation()
        info = self._build_info(
            reward_signal=RewardSignal(
                value=round(reward, 3),
                components={key: round(value, 3) for key, value in reward_components.items()},
                rationale=self._state.last_action_effect,
            )
        )
        return observation, round(reward, 3), self._state.done, info

    def state(self) -> InternalState:
        return self._state.model_copy(deep=True)

    def _build_state(self, task_id: str) -> InternalState:
        task = TASKS[task_id]
        return InternalState(
            task_id=task.task_id,
            difficulty=task.difficulty,
            scenario_summary=task.scenario_summary,
            true_condition=task.initial_true_condition.model_copy(deep=True),
            observed_condition=task.initial_observed_condition.model_copy(deep=True),
            environment_context=task.environment_context.model_copy(deep=True),
            max_steps=task.max_steps,
            optimal_sequence=deepcopy(task.optimal_sequence),
            hidden_notes=deepcopy(task.hidden_notes),
        )

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            scenario_summary=self._state.scenario_summary,
            patient_condition=self._state.observed_condition.model_copy(deep=True),
            time_elapsed=self._state.time_elapsed,
            actions_taken=list(self._state.actions_taken),
            environment_context=self._state.environment_context.model_copy(deep=True),
            available_actions=list(ActionType),
            last_action_effect=self._state.last_action_effect,
            risk_level=self._risk_level(),
        )

    def _build_info(
        self,
        termination_locked: bool = False,
        reward_signal: RewardSignal | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty.value,
            "termination_reason": self._state.termination_reason,
            "success": self._state.success,
            "score_so_far": self._grader.grade_task(self._state.task_id, self._state.actions_taken),
            "optimal_sequence": [action.value for action in self._state.optimal_sequence],
            "termination_locked": termination_locked,
            "reward_signal": reward_signal.model_dump() if reward_signal else None,
        }

    def _apply_action(self, action: ActionType, effects: list[str]) -> float:
        task_id = self._state.task_id
        reward = 0.0

        if action == ActionType.CHECK_SCENE_SAFETY:
            if self._state.environment_context.hazards:
                effects.append("Hazards identified and managed as well as possible.")
                reward += 1.2
            else:
                effects.append("Scene appears safe.")
                reward += 0.4
        elif action == ActionType.CALL_EMERGENCY:
            if not self._state.emergency_called:
                self._state.emergency_called = True
                effects.append("Emergency services alerted.")
                reward += 2.0
            else:
                effects.append("Emergency services were already contacted.")
                reward -= 0.3
        elif action == ActionType.CHECK_RESPONSIVENESS:
            self._state.responsiveness_checked = True
            self._state.observed_condition.conscious_status = self._state.true_condition.conscious_status
            effects.append("Responsiveness assessed.")
            reward += 0.7
        elif action == ActionType.CHECK_BREATHING:
            self._state.breathing_checked = True
            self._state.observed_condition.breathing_status = self._state.true_condition.breathing_status
            effects.append("Breathing assessed.")
            reward += 1.0
        elif action == ActionType.CHECK_PULSE:
            self._state.pulse_checked = True
            self._state.observed_condition.pulse_status = self._state.true_condition.pulse_status
            effects.append("Pulse assessed.")
            reward += 0.8
        elif action == ActionType.START_CPR:
            if task_id == "cardiac_arrest_easy":
                if self._state.true_condition.breathing_status == BreathingStatus.ABSENT:
                    self._state.cpr_started = True
                    effects.append("CPR started promptly.")
                    reward += 2.5
                else:
                    effects.append("CPR is not indicated while the patient is breathing.")
                    reward -= 1.6
            else:
                if self._state.true_condition.breathing_status == BreathingStatus.ABSENT and self._state.true_condition.pulse_status == PulseStatus.ABSENT:
                    self._state.cpr_started = True
                    effects.append("CPR started after recognizing cardiac arrest.")
                    reward += 2.0
                else:
                    effects.append("CPR would be unsafe in the current state.")
                    reward -= 1.8
        elif action == ActionType.USE_AED:
            if task_id == "cardiac_arrest_easy" and self._state.cpr_started:
                self._state.aed_used = True
                self._state.true_condition.pulse_status = PulseStatus.WEAK
                effects.append("AED applied after CPR began.")
                reward += 2.3
            else:
                effects.append("AED use is ineffective or premature here.")
                reward -= 1.4
        elif action == ActionType.APPLY_PRESSURE:
            if self._state.true_condition.bleeding_severity in {BleedingSeverity.SEVERE, BleedingSeverity.CRITICAL}:
                self._state.pressure_applied = True
                self._state.true_condition.bleeding_severity = BleedingSeverity.MODERATE
                effects.append("Direct pressure reduces bleeding.")
                reward += 2.2
            elif self._state.true_condition.bleeding_severity == BleedingSeverity.MODERATE:
                self._state.pressure_applied = True
                self._state.true_condition.bleeding_severity = BleedingSeverity.MILD
                effects.append("Bleeding further reduced.")
                reward += 1.0
            else:
                effects.append("No major external bleeding to compress.")
                reward -= 0.8
        elif action == ActionType.CONTROL_AIRWAY:
            if self._state.true_condition.airway_status == AirwayStatus.COMPROMISED:
                self._state.airway_controlled = True
                self._state.true_condition.airway_status = AirwayStatus.CLEAR
                if self._state.true_condition.breathing_status == BreathingStatus.ABSENT:
                    self._state.true_condition.breathing_status = BreathingStatus.SHALLOW
                elif self._state.true_condition.breathing_status == BreathingStatus.SHALLOW:
                    self._state.true_condition.breathing_status = BreathingStatus.NORMAL
                effects.append("Airway supported and breathing improves.")
                reward += 2.0
            else:
                effects.append("Airway support provided, but airway was already clear.")
                reward -= 0.2
        elif action == ActionType.PLACE_RECOVERY_POSITION:
            if (
                self._state.true_condition.breathing_status != BreathingStatus.ABSENT
                and self._state.true_condition.conscious_status == ConsciousStatus.UNCONSCIOUS
                and self._state.task_id != "road_accident_hard"
            ):
                effects.append("Recovery position protects the airway.")
                reward += 1.0
            else:
                effects.append("Recovery position is unsafe or not indicated.")
                reward -= 1.5
        elif action == ActionType.MONITOR_PATIENT:
            effects.append("Patient reassessed and trends monitored.")
            reward += 0.9 if self._state.emergency_called else 0.2
        elif action == ActionType.WAIT:
            effects.append("No intervention performed while the condition evolves.")
            reward -= 1.2

        return reward

    def _apply_progression(self, effects: list[str]) -> float:
        reward = 0.0
        task_id = self._state.task_id

        if task_id == "cardiac_arrest_easy":
            if not self._state.cpr_started:
                reward -= 1.8
                effects.append("Ongoing cardiac arrest without CPR worsens survival odds.")
            if self._state.cpr_started and not self._state.aed_used:
                reward += 0.5
                effects.append("CPR is maintaining some circulation while waiting for defibrillation.")
            if self._state.aed_used:
                self._state.true_condition.breathing_status = BreathingStatus.SHALLOW
                self._state.true_condition.pulse_status = PulseStatus.WEAK
                effects.append("After defibrillation, the patient shows signs of circulation.")
        elif task_id == "severe_bleeding_medium":
            if not self._state.pressure_applied:
                reward -= 1.4
                if self._state.time_elapsed >= 2:
                    self._state.true_condition.bleeding_severity = BleedingSeverity.CRITICAL
                    self._state.true_condition.pulse_status = PulseStatus.WEAK
                    effects.append("Bleeding remains uncontrolled and worsens.")
                if self._state.time_elapsed >= 4:
                    self._state.true_condition.conscious_status = ConsciousStatus.CONFUSED
                    effects.append("Signs of shock are developing.")
            else:
                if self._state.true_condition.bleeding_severity == BleedingSeverity.MODERATE:
                    self._state.true_condition.pulse_status = PulseStatus.NORMAL
                reward += 0.5
                effects.append("Bleeding control is preserving circulation.")
        elif task_id == "road_accident_hard":
            if not self._state.pressure_applied:
                reward -= 1.3
                if self._state.time_elapsed >= 2:
                    self._state.true_condition.bleeding_severity = BleedingSeverity.CRITICAL
                    effects.append("Roadside hemorrhage is worsening.")
            else:
                if self._state.true_condition.bleeding_severity == BleedingSeverity.MODERATE:
                    reward += 0.4

            if not self._state.airway_controlled:
                if self._state.time_elapsed >= 3:
                    self._state.true_condition.airway_status = AirwayStatus.COMPROMISED
                    self._state.true_condition.breathing_status = BreathingStatus.ABSENT
                    self._state.true_condition.conscious_status = ConsciousStatus.UNCONSCIOUS
                    reward -= 1.5
                    effects.append("Airway compromise progresses to respiratory failure.")
                else:
                    reward -= 0.6
            else:
                if self._state.true_condition.pulse_status == PulseStatus.WEAK:
                    self._state.true_condition.pulse_status = PulseStatus.NORMAL
                reward += 0.7
                effects.append("Airway control improves oxygenation.")

            if self._state.true_condition.breathing_status == BreathingStatus.ABSENT and not self._state.cpr_started:
                if self._state.time_elapsed >= 5:
                    self._state.true_condition.pulse_status = PulseStatus.ABSENT
                    effects.append("Prolonged respiratory failure progresses to cardiac arrest.")
                    reward -= 2.0

        return reward

    def _update_observed_condition(self) -> None:
        self._state.observed_condition.conscious_status = self._state.true_condition.conscious_status
        self._state.observed_condition.bleeding_severity = self._state.true_condition.bleeding_severity

        if self._state.breathing_checked or self._state.task_id == "cardiac_arrest_easy":
            self._state.observed_condition.breathing_status = self._state.true_condition.breathing_status
        if self._state.pulse_checked:
            self._state.observed_condition.pulse_status = self._state.true_condition.pulse_status
        if self._state.airway_controlled or self._state.task_id == "severe_bleeding_medium":
            self._state.observed_condition.airway_status = self._state.true_condition.airway_status

    def _check_termination(self) -> None:
        if self._state.time_elapsed >= self._state.max_steps:
            self._state.done = True
            self._state.termination_reason = "max_steps_reached"
            self._state.success = False
            return

        if self._state.task_id == "cardiac_arrest_easy":
            if (
                self._state.emergency_called
                and self._state.cpr_started
                and self._state.aed_used
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif self._state.time_elapsed >= 4 and not self._state.cpr_started:
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False
        elif self._state.task_id == "severe_bleeding_medium":
            if (
                self._state.emergency_called
                and self._state.pressure_applied
                and ActionType.CHECK_PULSE in self._state.actions_taken
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
                and self._state.time_elapsed >= 5
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif (
                self._state.true_condition.bleeding_severity == BleedingSeverity.CRITICAL
                and self._state.time_elapsed >= 5
                and not self._state.pressure_applied
            ):
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False
        elif self._state.task_id == "road_accident_hard":
            if (
                self._state.emergency_called
                and self._state.pressure_applied
                and self._state.airway_controlled
                and ActionType.CHECK_PULSE in self._state.actions_taken
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
                and self._state.time_elapsed >= 7
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif self._state.true_condition.pulse_status == PulseStatus.ABSENT:
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False

    def _risk_level(self) -> str:
        condition = self._state.true_condition
        if (
            condition.pulse_status == PulseStatus.ABSENT
            or condition.breathing_status == BreathingStatus.ABSENT
            or condition.bleeding_severity == BleedingSeverity.CRITICAL
        ):
            return "critical"
        if (
            condition.bleeding_severity == BleedingSeverity.SEVERE
            or condition.breathing_status == BreathingStatus.SHALLOW
            or condition.pulse_status == PulseStatus.WEAK
        ):
            return "high"
        return "moderate"
