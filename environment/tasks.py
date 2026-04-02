from __future__ import annotations

from environment.models import (
    ActionType,
    AirwayStatus,
    BleedingSeverity,
    BreathingStatus,
    ConsciousStatus,
    DifficultyLevel,
    EnvironmentContext,
    PatientCondition,
    PulseStatus,
    TaskDefinition,
)


CARDIAC_ARREST_EASY = TaskDefinition(
    task_id="cardiac_arrest_easy",
    difficulty=DifficultyLevel.EASY,
    description="Clear cardiac arrest scenario in a public location with strong cues and nearby equipment.",
    scenario_summary=(
        "An adult has collapsed in a staffed airport terminal. A bystander reports the patient is "
        "unresponsive and not breathing normally. An AED cabinet is visible nearby."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNCONSCIOUS,
        breathing_status=BreathingStatus.ABSENT,
        bleeding_severity=BleedingSeverity.NONE,
        pulse_status=PulseStatus.ABSENT,
        airway_status=AirwayStatus.CLEAR,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNCONSCIOUS,
        breathing_status=BreathingStatus.ABSENT,
        bleeding_severity=BleedingSeverity.NONE,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="airport_terminal",
        help_availability="trained staff and bystanders nearby",
        hazards=[],
        equipment_available=["AED", "first_aid_kit", "phone"],
    ),
    max_steps=6,
    optimal_sequence=[
        ActionType.CALL_EMERGENCY,
        ActionType.CHECK_BREATHING,
        ActionType.START_CPR,
        ActionType.USE_AED,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Emergency services contacted, CPR started promptly, and AED applied.",
    failure_criteria="Repeated delays in CPR or AED use cause irreversible deterioration.",
    hidden_notes={"patient_type": "adult"},
)


SEVERE_BLEEDING_MEDIUM = TaskDefinition(
    task_id="severe_bleeding_medium",
    difficulty=DifficultyLevel.MEDIUM,
    description="Visible severe bleeding with partial physiological information and required prioritization.",
    scenario_summary=(
        "A kitchen worker has a deep laceration to the forearm from broken glass. Blood is rapidly soaking "
        "through towels. The patient is awake but frightened, and professional help is not yet on scene."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.ALERT,
        breathing_status=BreathingStatus.NORMAL,
        bleeding_severity=BleedingSeverity.SEVERE,
        pulse_status=PulseStatus.WEAK,
        airway_status=AirwayStatus.CLEAR,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.ALERT,
        breathing_status=BreathingStatus.NORMAL,
        bleeding_severity=BleedingSeverity.SEVERE,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.CLEAR,
    ),
    environment_context=EnvironmentContext(
        location_type="restaurant_kitchen",
        help_availability="one coworker available to assist",
        hazards=["broken_glass", "slippery_floor"],
        equipment_available=["gloves", "clean_cloth", "phone"],
    ),
    max_steps=7,
    optimal_sequence=[
        ActionType.CHECK_SCENE_SAFETY,
        ActionType.CALL_EMERGENCY,
        ActionType.APPLY_PRESSURE,
        ActionType.CHECK_PULSE,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Scene safety addressed, bleeding controlled, pulse reassessed, and patient monitored.",
    failure_criteria="Bleeding remains uncontrolled long enough to cause shock and collapse.",
    hidden_notes={"injury_location": "forearm"},
)


ROAD_ACCIDENT_HARD = TaskDefinition(
    task_id="road_accident_hard",
    difficulty=DifficultyLevel.HARD,
    description="Dynamic roadside trauma requiring hazard awareness, hemorrhage control, airway management, and reassessment.",
    scenario_summary=(
        "A motorcyclist is lying beside a road after a collision. Traffic is still moving nearby. The patient "
        "is confused, breathing shallowly, and has heavy bleeding from the thigh. There is no immediate clinical support."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.CONFUSED,
        breathing_status=BreathingStatus.SHALLOW,
        bleeding_severity=BleedingSeverity.SEVERE,
        pulse_status=PulseStatus.WEAK,
        airway_status=AirwayStatus.COMPROMISED,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.CONFUSED,
        breathing_status=BreathingStatus.UNKNOWN,
        bleeding_severity=BleedingSeverity.SEVERE,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="roadside",
        help_availability="one untrained driver stopped to help",
        hazards=["moving_traffic", "fuel_spill_risk"],
        equipment_available=["phone", "clean_cloth", "high_visibility_jacket"],
    ),
    max_steps=9,
    optimal_sequence=[
        ActionType.CHECK_SCENE_SAFETY,
        ActionType.CALL_EMERGENCY,
        ActionType.APPLY_PRESSURE,
        ActionType.CHECK_BREATHING,
        ActionType.CONTROL_AIRWAY,
        ActionType.CHECK_PULSE,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Hazards managed, emergency help activated, bleeding controlled, airway supported, and patient kept under observation.",
    failure_criteria="Uncontrolled bleeding or worsening airway compromise leads to critical deterioration.",
    hidden_notes={"injury_location": "thigh", "airway_risk": "vomiting"},
)


TASKS: dict[str, TaskDefinition] = {
    CARDIAC_ARREST_EASY.task_id: CARDIAC_ARREST_EASY,
    SEVERE_BLEEDING_MEDIUM.task_id: SEVERE_BLEEDING_MEDIUM,
    ROAD_ACCIDENT_HARD.task_id: ROAD_ACCIDENT_HARD,
}


DEFAULT_TASK_ID = CARDIAC_ARREST_EASY.task_id
