from __future__ import annotations

from environment.env import EmergencyFirstResponseDecisionEngine
from environment.grader import EmergencyTaskGrader
from environment.models import Action
from environment.tasks import TASKS
from rl_agent import QLearningEmergencyAgent


def run() -> None:
    grader = EmergencyTaskGrader()
    environment = EmergencyFirstResponseDecisionEngine()

    for task_id, task in TASKS.items():
        first_observation = environment.reset(task_id)
        assert first_observation.task_id == task_id
        score = grader.grade_task(task_id, task.optimal_sequence)
        assert 0.0 <= score <= 1.0

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
