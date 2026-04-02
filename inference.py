from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional local fallback
    OpenAI = None  # type: ignore[assignment]

from environment.env import EmergencyFirstResponseDecisionEngine
from environment.models import Action, ActionType, Observation
from environment.tasks import TASKS
from rl_agent import DEFAULT_Q_TABLE_PATH, QLearningEmergencyAgent


class BaselineEmergencyAgent:
    def __init__(self, client: Any, model_name: str, use_model: bool, rl_agent: QLearningEmergencyAgent | None = None) -> None:
        self._client = client
        self._model_name = model_name
        self._use_model = use_model
        self._rl_agent = rl_agent

    def choose_action(self, observation: Observation) -> ActionType:
        if self._rl_agent is not None:
            return self._rl_agent.choose_action(observation, greedy=True)

        if not self._use_model or self._client is None:
            return self._fallback_policy(observation)

        prompt = self._build_prompt(observation)
        try:
            completion = self._client.chat.completions.create(
                model=self._model_name,
                temperature=0,
                max_tokens=16,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a first-response decision policy. Return exactly one action token from "
                            "the allowed action list and nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            candidate = (completion.choices[0].message.content or "").strip().split()[0]
            return ActionType(candidate)
        except Exception:
            return self._fallback_policy(observation)

    def _build_prompt(self, observation: Observation) -> str:
        return (
            f"task_id={observation.task_id}\n"
            f"time_elapsed={observation.time_elapsed}\n"
            f"risk_level={observation.risk_level}\n"
            f"conscious_status={observation.patient_condition.conscious_status.value}\n"
            f"breathing_status={observation.patient_condition.breathing_status.value}\n"
            f"bleeding_severity={observation.patient_condition.bleeding_severity.value}\n"
            f"pulse_status={observation.patient_condition.pulse_status.value}\n"
            f"airway_status={observation.patient_condition.airway_status.value}\n"
            f"actions_taken={[action.value for action in observation.actions_taken]}\n"
            f"available_actions={[action.value for action in observation.available_actions]}\n"
            "Respond with one action token."
        )

    def _fallback_policy(self, observation: Observation) -> ActionType:
        planned_actions = TASKS[observation.task_id].optimal_sequence
        next_index = len(observation.actions_taken)
        if next_index < len(planned_actions):
            return planned_actions[next_index]
        return ActionType.MONITOR_PATIENT


def build_client() -> tuple[Any, bool]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    base_url = os.getenv("API_BASE_URL")
    if OpenAI is None or not api_key:
        return None, False
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url.rstrip("/")), True
    return OpenAI(api_key=api_key), True


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    client, use_model = build_client()
    learned_agent = QLearningEmergencyAgent()
    has_learned_policy = learned_agent.load(DEFAULT_Q_TABLE_PATH)
    agent = BaselineEmergencyAgent(
        client=client,
        model_name=model_name,
        use_model=use_model,
        rl_agent=learned_agent if has_learned_policy else None,
    )
    environment = EmergencyFirstResponseDecisionEngine()

    for task_id in TASKS:
        print(f"[START] task={task_id}")
        observation = environment.reset(task_id)
        done = False
        step_index = 0

        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, _info = environment.step(Action(action_type=action))
            print(f"[STEP] step={step_index} reward={reward}")
            step_index += 1

        print(f"[END] task={task_id}")


if __name__ == "__main__":
    main()
