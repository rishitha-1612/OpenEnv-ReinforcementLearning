import { useEffect, useState } from "react";

const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (window.location.port === "5173" ? "http://127.0.0.1:8000" : window.location.origin);

const FALLBACK_TASKS = [
  {
    task_id: "cardiac_arrest_easy",
    difficulty: "easy",
    description: "Clear cardiac arrest case with nearby AED access.",
    scenario_summary:
      "An adult has collapsed in a staffed airport terminal and is not breathing normally.",
    max_steps: 6
  },
  {
    task_id: "severe_bleeding_medium",
    difficulty: "medium",
    description: "Severe external bleeding with partial physiological visibility.",
    scenario_summary:
      "A kitchen worker has a deep laceration with rapid blood loss and limited support.",
    max_steps: 7
  },
  {
    task_id: "road_accident_hard",
    difficulty: "hard",
    description: "Dynamic roadside trauma with hazards, bleeding, and airway compromise.",
    scenario_summary:
      "A motorcyclist is injured by the roadside with evolving airway and bleeding risks.",
    max_steps: 9
  }
];

const ACTIONS = [
  "CALL_EMERGENCY",
  "CHECK_SCENE_SAFETY",
  "CHECK_RESPONSIVENESS",
  "CHECK_BREATHING",
  "CHECK_PULSE",
  "START_CPR",
  "USE_AED",
  "APPLY_PRESSURE",
  "CONTROL_AIRWAY",
  "PLACE_RECOVERY_POSITION",
  "MONITOR_PATIENT",
  "WAIT"
];

function App() {
  const [apiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [tasks, setTasks] = useState(FALLBACK_TASKS);
  const [selectedTaskId, setSelectedTaskId] = useState(FALLBACK_TASKS[0].task_id);
  const [selectedAction, setSelectedAction] = useState("CALL_EMERGENCY");
  const [stepOutput, setStepOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasStarted, setHasStarted] = useState(false);

  useEffect(() => {
    async function loadTasks() {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(`${apiBaseUrl}/tasks`);
        if (!response.ok) {
          throw new Error(`Unable to load tasks (${response.status})`);
        }

        const payload = await response.json();
        if (Array.isArray(payload) && payload.length > 0) {
          setTasks(payload);
          setSelectedTaskId(payload[0].task_id);
        }
      } catch (err) {
        setTasks(FALLBACK_TASKS);
        setError(`${err.message || "Unable to load tasks."} Using built-in task list instead.`);
      } finally {
        setLoading(false);
      }
    }

    loadTasks();
  }, [apiBaseUrl]);

  async function handleReset(taskId = selectedTaskId) {
    if (!taskId) {
      return;
    }

    setLoading(true);
    setError("");

      try {
      const response = await fetch(`${apiBaseUrl}/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: taskId })
      });

      if (!response.ok) {
        throw new Error(`Reset failed (${response.status})`);
      }

      const payload = await response.json();
      setStepOutput(payload);
      setSelectedTaskId(taskId);
      setHasStarted(true);
      const firstAction = payload.observation.available_actions?.[0] || "CALL_EMERGENCY";
      setSelectedAction(firstAction);
    } catch (err) {
      setError(err.message || "Reset failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleStep() {
    if (!selectedAction) {
      return;
    }

    setLoading(true);
    setError("");

      try {
      const response = await fetch(`${apiBaseUrl}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action_type: selectedAction })
      });

      if (!response.ok) {
        throw new Error(`Step failed (${response.status})`);
      }

      const payload = await response.json();
      setStepOutput(payload);
      setHasStarted(true);
    } catch (err) {
      setError(err.message || "Step failed.");
    } finally {
      setLoading(false);
    }
  }

  const selectedTask = tasks.find((task) => task.task_id === selectedTaskId);
  const observation = stepOutput?.observation;
  const info = stepOutput?.info;

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Decision Support UI</p>
          <h1>Emergency First-Response Decision Engine</h1>
          <p className="hero-copy">
            A simple control panel for resetting emergency scenarios, sending structured actions,
            and reviewing how the environment state evolves step by step.
          </p>
        </div>
        <div className="api-pill">Run everything from this page</div>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>Scenario Control</h2>
          <p className="muted">
            Pick a scenario card below, then start or reset it from this page.
          </p>

          <div className="task-list">
            {tasks.map((task) => (
              <button
                key={task.task_id}
                type="button"
                className={`task-card selectable ${selectedTaskId === task.task_id ? "selected" : ""}`}
                onClick={() => {
                  setSelectedTaskId(task.task_id);
                  setStepOutput(null);
                  setHasStarted(false);
                  setError("");
                }}
              >
                <div className="tag-row">
                  <span className={`tag tag-${task.difficulty}`}>{task.difficulty}</span>
                  <span className="tag">max steps {task.max_steps}</span>
                </div>
                <strong>{task.task_id}</strong>
                <p>{task.description}</p>
                <p className="muted">{task.scenario_summary}</p>
              </button>
            ))}
          </div>

          <div className="button-row">
            <button className="primary-button" onClick={() => handleReset(selectedTaskId)} disabled={loading || !selectedTaskId}>
              {loading ? "Working..." : hasStarted ? "Reset Current Scenario" : "Start Selected Scenario"}
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>Action Panel</h2>
          <div className="action-grid">
            {(observation?.available_actions || ACTIONS).map((action) => (
              <button
                key={action}
                type="button"
                className={`action-button ${selectedAction === action ? "active" : ""}`}
                onClick={() => setSelectedAction(action)}
              >
                {action}
              </button>
            ))}
          </div>

          <button className="primary-button" onClick={handleStep} disabled={loading || !stepOutput}>
            Submit Step
          </button>

          <div className="status-strip">
            <div>
              <span className="status-label">Reward</span>
              <strong>{stepOutput ? stepOutput.reward : "-"}</strong>
            </div>
            <div>
              <span className="status-label">Done</span>
              <strong>{stepOutput ? String(stepOutput.done) : "-"}</strong>
            </div>
            <div>
              <span className="status-label">Score</span>
              <strong>{info?.score_so_far ?? "-"}</strong>
            </div>
          </div>

          {error ? <p className="error-banner">{error}</p> : null}
          {!hasStarted ? (
            <p className="muted">Start a scenario first, then choose an action and submit a step.</p>
          ) : null}
        </section>

        <section className="panel">
          <h2>Observation</h2>
          {observation ? (
            <div className="details-grid">
              <Detail label="Risk Level" value={observation.risk_level} />
              <Detail label="Time Elapsed" value={observation.time_elapsed} />
              <Detail label="Consciousness" value={observation.patient_condition.conscious_status} />
              <Detail label="Breathing" value={observation.patient_condition.breathing_status} />
              <Detail label="Bleeding" value={observation.patient_condition.bleeding_severity} />
              <Detail label="Pulse" value={observation.patient_condition.pulse_status} />
              <Detail label="Airway" value={observation.patient_condition.airway_status} />
              <Detail label="Location" value={observation.environment_context.location_type} />
              <Detail label="Help" value={observation.environment_context.help_availability} />
            </div>
          ) : (
            <p className="muted">Reset a task to load the first observation.</p>
          )}
        </section>

        <section className="panel">
          <h2>Response Details</h2>
          <p>{observation?.last_action_effect || "No step has been taken yet."}</p>

          <h3>Actions Taken</h3>
          <div className="chip-wrap">
            {(observation?.actions_taken || []).map((action, index) => (
              <span className="chip" key={`${action}-${index}`}>
                {index + 1}. {action}
              </span>
            ))}
          </div>

          <h3>Termination</h3>
          <p className="muted">{info?.termination_reason || "Episode is active."}</p>
        </section>

        <section className="panel">
          <h2>Structured JSON</h2>
          <pre>{JSON.stringify(stepOutput, null, 2) || "{}"}</pre>
        </section>
      </main>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
