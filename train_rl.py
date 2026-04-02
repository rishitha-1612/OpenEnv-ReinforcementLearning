from __future__ import annotations

from rl_agent import DEFAULT_Q_TABLE_PATH, QLearningEmergencyAgent


def main() -> None:
    agent = QLearningEmergencyAgent()
    summary = agent.train()
    path = agent.save(DEFAULT_Q_TABLE_PATH)
    evaluation = agent.evaluate()

    print("training_summary", summary)
    print("policy_path", path)
    print("evaluation", evaluation)


if __name__ == "__main__":
    main()
