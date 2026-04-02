from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from environment.env import EmergencyFirstResponseDecisionEngine
from environment.models import Action, ActionType, Observation
from environment.tasks import TASKS


ALL_ACTIONS = list(ActionType)
DEFAULT_Q_TABLE_PATH = Path(__file__).parent / "artifacts" / "q_table.json"


def encode_observation(observation: Observation) -> str:
    condition = observation.patient_condition
    actions_taken = set(observation.actions_taken)
    key = {
        "task_id": observation.task_id,
        "time_elapsed": observation.time_elapsed,
        "risk_level": observation.risk_level,
        "conscious_status": condition.conscious_status.value,
        "breathing_status": condition.breathing_status.value,
        "bleeding_severity": condition.bleeding_severity.value,
        "pulse_status": condition.pulse_status.value,
        "airway_status": condition.airway_status.value,
        "emergency_called": ActionType.CALL_EMERGENCY in actions_taken,
        "scene_checked": ActionType.CHECK_SCENE_SAFETY in actions_taken,
        "breathing_checked": ActionType.CHECK_BREATHING in actions_taken,
        "pulse_checked": ActionType.CHECK_PULSE in actions_taken,
        "cpr_started": ActionType.START_CPR in actions_taken,
        "aed_used": ActionType.USE_AED in actions_taken,
        "pressure_applied": ActionType.APPLY_PRESSURE in actions_taken,
        "airway_controlled": ActionType.CONTROL_AIRWAY in actions_taken,
        "monitored": ActionType.MONITOR_PATIENT in actions_taken,
        "action_history": [action.value for action in observation.actions_taken],
    }
    return json.dumps(key, sort_keys=True)


class QLearningEmergencyAgent:
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.92,
        epsilon: float = 0.2,
        seed: int = 7,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._q_table: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {action.value: 0.0 for action in ALL_ACTIONS}
        )

    def choose_action(self, observation: Observation, greedy: bool = False) -> ActionType:
        state_key = encode_observation(observation)
        action_values = self._q_table[state_key]

        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(ALL_ACTIONS)

        best_value = max(action_values.values())
        best_actions = [
            ActionType(action_name)
            for action_name, value in action_values.items()
            if value == best_value
        ]
        best_actions.sort(key=lambda action: action.value)
        return best_actions[0]

    def update(
        self,
        observation: Observation,
        action: ActionType,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        state_key = encode_observation(observation)
        next_state_key = encode_observation(next_observation)

        current_value = self._q_table[state_key][action.value]
        next_best = 0.0 if done else max(self._q_table[next_state_key].values())
        target = reward + self.gamma * next_best
        self._q_table[state_key][action.value] = current_value + self.alpha * (target - current_value)

    def train(
        self,
        episodes_per_task: int = 400,
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.995,
    ) -> dict[str, float]:
        environment = EmergencyFirstResponseDecisionEngine()
        training_summary: dict[str, float] = {}

        self._bootstrap_from_expert_rollouts(environment, passes=60)

        for task_id in TASKS:
            episode_returns: list[float] = []
            local_epsilon = self.epsilon

            for _episode in range(episodes_per_task):
                observation = environment.reset(task_id)
                done = False
                total_reward = 0.0

                while not done:
                    self.epsilon = local_epsilon
                    action = self.choose_action(observation, greedy=False)
                    next_observation, reward, done, _info = environment.step(Action(action_type=action))
                    self.update(observation, action, reward, next_observation, done)
                    total_reward += reward
                    observation = next_observation

                episode_returns.append(total_reward)
                local_epsilon = max(min_epsilon, local_epsilon * epsilon_decay)

            training_summary[task_id] = round(sum(episode_returns[-25:]) / min(25, len(episode_returns)), 4)

        self.epsilon = 0.0
        return training_summary

    def _bootstrap_from_expert_rollouts(
        self,
        environment: EmergencyFirstResponseDecisionEngine,
        passes: int = 40,
    ) -> None:
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(passes):
            for task_id, task in TASKS.items():
                observation = environment.reset(task_id)
                trajectory: list[tuple[Observation, ActionType, float, Observation, bool]] = []

                for action_type in task.optimal_sequence:
                    next_observation, reward, done, _info = environment.step(Action(action_type=action_type))
                    trajectory.append((observation, action_type, reward, next_observation, done))
                    observation = next_observation
                    if done:
                        break

                for observation, action_type, reward, next_observation, done in reversed(trajectory):
                    self.update(observation, action_type, reward, next_observation, done)

        self.epsilon = original_epsilon

    def evaluate(self) -> dict[str, dict[str, float | bool | int]]:
        environment = EmergencyFirstResponseDecisionEngine()
        results: dict[str, dict[str, float | bool | int]] = {}

        for task_id in TASKS:
            observation = environment.reset(task_id)
            done = False
            total_reward = 0.0
            step_count = 0
            final_info: dict[str, object] = {}

            while not done:
                action = self.choose_action(observation, greedy=True)
                observation, reward, done, final_info = environment.step(Action(action_type=action))
                total_reward += reward
                step_count += 1

            results[task_id] = {
                "total_reward": round(total_reward, 4),
                "done": bool(done),
                "success": bool(final_info.get("success", False)),
                "score_so_far": float(final_info.get("score_so_far", 0.0)),
                "steps": step_count,
            }

        return results

    def save(self, path: Path = DEFAULT_Q_TABLE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {state: values for state, values in self._q_table.items()}
        path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load(self, path: Path = DEFAULT_Q_TABLE_PATH) -> bool:
        if not path.exists():
            return False

        payload = json.loads(path.read_text(encoding="utf-8"))
        self._q_table = defaultdict(
            lambda: {action.value: 0.0 for action in ALL_ACTIONS},
            {state: {action.value: float(values.get(action.value, 0.0)) for action in ALL_ACTIONS} for state, values in payload.items()},
        )
        self.epsilon = 0.0
        return True
