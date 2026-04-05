from __future__ import annotations

from environment.env import EmergencyFirstResponseDecisionEngine
from environment.grader import EmergencyTaskGrader
from environment.models import Action, ActionType
from environment.tasks import TASKS
from rl_agent import QLearningEmergencyAgent


def run() -> None:
    grader = EmergencyTaskGrader()
    environment = EmergencyFirstResponseDecisionEngine()
    random_sequence = [
        ActionType.WAIT,
        ActionType.WAIT,
        ActionType.WAIT,
        ActionType.MONITOR_PATIENT,
    ]

    for task_id, task in TASKS.items():
        first_observation = environment.reset(task_id)
        assert first_observation.task_id == task_id

        score_optimal = grader.grade_task(task_id, task.optimal_sequence)
        score_random = grader.grade_task(task_id, random_sequence)
        assert 0.0 <= score_optimal <= 1.0
        assert score_optimal >= 0.95, f"Optimal sequence for {task_id} scored too low: {score_optimal}"
        assert score_random < 0.5, (
            f"Grader for {task_id} doesn't discriminate - random got {score_random}"
        )
        assert score_optimal > score_random, (
            f"Optimal sequence for {task_id} should outscore random: optimal={score_optimal}, random={score_random}"
        )

        rewards_first: list[float] = []
        rewards_second: list[float] = []

        environment.reset(task_id)
        for action_type in task.optimal_sequence:
            _, reward, _, _ = environment.step(Action(action_type=action_type))
            rewards_first.append(reward)

        environment.reset(task_id)
        for action_type in task.optimal_sequence:
            _, reward, _, _ = environment.step(Action(action_type=action_type))
            rewards_second.append(reward)

        assert rewards_first == rewards_second

    agent = QLearningEmergencyAgent()
    agent.train(episodes_per_task=500)
    evaluation = agent.evaluate()
    assert all(result["success"] for result in evaluation.values())
    assert all(float(result["score_so_far"]) >= 0.95 for result in evaluation.values())

    print("validation_passed")


if __name__ == "__main__":
    run()
