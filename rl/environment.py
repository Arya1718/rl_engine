"""
Healthcare Environment for Fairness-Constrained RL
----------------------------------------------------
A rich, stochastic healthcare MDP with:

State representation (8-dimensional discretised):
  Disease severity (5 bins) | Cumulative financial burden (5 bins)
  Recent adherence (binary) | Comorbidity index (3 levels)
  Treatment resistance level (3 levels) | Time-in-treatment period (3 bins)
  Insurance coverage tier (3 levels) | Age-risk bracket (3 levels)

Action space (5 treatments):
  Conservative | Moderate | Standard | Intensive | Aggressive

Transition dynamics:
  Affordability-driven adherence (RFB-modulated)
  Comorbidity-modulated efficacy and risk
  Treatment resistance accumulation with decay
  Economic shocks (unemployment, insurance loss)
  Seasonal disease burden fluctuation
  Drug interaction effects (synergy penalties for rapid switching)
  Age-dependent vulnerability progression
  Non-stationary cost inflation and efficacy drift
  Social determinants multiplier
  Disease-specific progression models (diabetes/cardiac/respiratory)
  Pandemic wave dynamics (sinusoidal severity surges)
  Medication fatigue (prolonged same-treatment efficacy decay)
  Geographic access barriers affecting cost and adherence
  Genetic risk factor modulation of disease trajectory
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class PatientProfile:
    economic_capacity: float
    insurance_support: float
    vulnerability: float
    age: float
    risk_tier: int
    comorbidity_count: int = 0
    social_determinant_score: float = 0.5
    education_level: int = 1
    geographic_access: float = 0.8
    genetic_risk_factor: float = 0.0
    chronic_condition_type: int = 0  # 0=general, 1=diabetes, 2=cardiac, 3=respiratory


@dataclass
class PatientState:
    severity: float
    cumulative_cost: float
    last_adherence: float
    comorbidity_index: float = 0.0
    treatment_resistance: float = 0.0
    time_step: int = 0
    previous_action: int = -1
    consecutive_same_action: int = 0
    cumulative_adherence: float = 0.0
    total_steps: int = 0
    economic_shock_active: bool = False
    shock_remaining_steps: int = 0
    severity_trend: float = 0.0
    medication_fatigue: float = 0.0  # accumulated fatigue from prolonged treatment
    peak_severity: float = 0.0       # highest severity seen (for recovery tracking)


# Treatment action definitions
_TREATMENT_ACTIONS: List[Dict] = [
    {
        "name": "Conservative",
        "cost": 95.0,
        "efficacy": 0.12,
        "risk": 0.05,
        "resistance_rate": 0.005,
        "interaction_sensitivity": 0.02,
        "specialisation": "general",
        "fatigue_rate": 0.002,
    },
    {
        "name": "Moderate",
        "cost": 180.0,
        "efficacy": 0.20,
        "risk": 0.09,
        "resistance_rate": 0.012,
        "interaction_sensitivity": 0.05,
        "specialisation": "general",
        "fatigue_rate": 0.005,
    },
    {
        "name": "Standard",
        "cost": 280.0,
        "efficacy": 0.29,
        "risk": 0.14,
        "resistance_rate": 0.020,
        "interaction_sensitivity": 0.08,
        "specialisation": "targeted",
        "fatigue_rate": 0.010,
    },
    {
        "name": "Intensive",
        "cost": 420.0,
        "efficacy": 0.37,
        "risk": 0.19,
        "resistance_rate": 0.032,
        "interaction_sensitivity": 0.12,
        "specialisation": "targeted",
        "fatigue_rate": 0.016,
    },
    {
        "name": "Aggressive",
        "cost": 600.0,
        "efficacy": 0.44,
        "risk": 0.26,
        "resistance_rate": 0.048,
        "interaction_sensitivity": 0.18,
        "specialisation": "experimental",
        "fatigue_rate": 0.024,
    },
]


# Disease-specific progression models
_DISEASE_MODELS: Dict[int, Dict] = {
    0: {  # general
        "name": "General",
        "severity_drift": 0.0,
        "treatment_response_mod": 1.0,
        "relapse_prob": 0.02,
        "complications_threshold": 0.85,
    },
    1: {  # diabetes
        "name": "Diabetes",
        "severity_drift": 0.008,        # slow progressive worsening
        "treatment_response_mod": 0.92,  # slightly harder to treat
        "relapse_prob": 0.04,            # higher relapse
        "complications_threshold": 0.70, # complications at lower severity
    },
    2: {  # cardiac
        "name": "Cardiac",
        "severity_drift": 0.005,
        "treatment_response_mod": 0.85,  # harder to treat
        "relapse_prob": 0.06,            # significant relapse risk
        "complications_threshold": 0.65, # critical threshold lower
    },
    3: {  # respiratory
        "name": "Respiratory",
        "severity_drift": 0.006,
        "treatment_response_mod": 0.95,
        "relapse_prob": 0.03,
        "complications_threshold": 0.75,
    },
}


class HealthcareEnvironment:
    """
    Rich healthcare MDP environment with socioeconomic fairness modelling.

    Scenarios: baseline | low_income_skew | high_chronic_load |
               policy_shock | pandemic_surge | rural_disparity
    """

    def __init__(
        self,
        scenario: str = "baseline",
        drift_cost_inflation: float = 0.0,
        drift_adherence_drop: float = 0.0,
        drift_efficacy_drop: float = 0.0,
    ) -> None:
        self.actions: List[Dict] = [dict(a) for a in _TREATMENT_ACTIONS]
        self.scenario = scenario
        self.drift_cost_inflation = max(-0.4, min(1.5, drift_cost_inflation))
        self.drift_adherence_drop = max(0.0, min(0.6, drift_adherence_drop))
        self.drift_efficacy_drop = max(0.0, min(0.7, drift_efficacy_drop))

        self._economic_shock_prob = 0.03
        self._seasonal_amplitude = 0.06
        self._resistance_decay = 0.008
        self._interaction_penalty_window = 2
        self._pandemic_wave_amplitude = 0.0
        self._pandemic_wave_period = 8.0

        if scenario == "pandemic_surge":
            self._economic_shock_prob = 0.08
            self._seasonal_amplitude = 0.12
            self._pandemic_wave_amplitude = 0.15
            self._pandemic_wave_period = 6.0
        elif scenario == "policy_shock":
            self._economic_shock_prob = 0.06
        elif scenario == "rural_disparity":
            self._seasonal_amplitude = 0.09

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def _disease_model(self, chronic_type: int) -> Dict:
        return _DISEASE_MODELS.get(chronic_type, _DISEASE_MODELS[0])

    # Patient Profile Sampling
    def sample_profile(self, rng) -> PatientProfile:
        segment_probs = [0.38, 0.36, 0.26]
        vulnerability_boost = 0.0
        comorbidity_boost = 0
        age_range = (21, 86)
        geo_base = (0.5, 1.0)

        if self.scenario == "low_income_skew":
            segment_probs = [0.64, 0.26, 0.10]
            vulnerability_boost = 0.08
        elif self.scenario == "high_chronic_load":
            segment_probs = [0.42, 0.38, 0.20]
            vulnerability_boost = 0.14
            comorbidity_boost = 1
            age_range = (45, 89)
        elif self.scenario == "policy_shock":
            segment_probs = [0.48, 0.36, 0.16]
            vulnerability_boost = 0.10
        elif self.scenario == "pandemic_surge":
            segment_probs = [0.52, 0.32, 0.16]
            vulnerability_boost = 0.16
            comorbidity_boost = 1
        elif self.scenario == "rural_disparity":
            segment_probs = [0.50, 0.34, 0.16]
            vulnerability_boost = 0.06
            geo_base = (0.15, 0.55)

        segment = int(rng.choice([0, 1, 2], p=segment_probs))
        age = float(rng.uniform(*age_range))

        age_factor = max(0.0, (age - 40.0) / 50.0)
        lam = 0.4 + 1.2 * age_factor + 0.5 * comorbidity_boost
        comorbidity_count = min(3, int(rng.poisson(lam)))

        chronic_type = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.25, 0.20, 0.15]))
        education = int(rng.choice([0, 1, 2], p=[0.25, 0.50, 0.25]))
        social_score = float(rng.uniform(0.2 + 0.15 * education, 0.6 + 0.15 * education))
        geographic = float(rng.uniform(*geo_base))
        genetic_risk = float(rng.beta(2, 8))

        if segment == 0:
            return PatientProfile(
                economic_capacity=float(rng.uniform(700, 1500)),
                insurance_support=float(rng.uniform(0.05, 0.22)),
                vulnerability=float(min(1.0, rng.uniform(0.55, 0.95) + vulnerability_boost)),
                age=age, risk_tier=2, comorbidity_count=comorbidity_count,
                social_determinant_score=float(min(1.0, social_score * 0.7)),
                education_level=min(education, 1), geographic_access=geographic,
                genetic_risk_factor=genetic_risk, chronic_condition_type=chronic_type,
            )
        if segment == 1:
            return PatientProfile(
                economic_capacity=float(rng.uniform(1500, 3200)),
                insurance_support=float(rng.uniform(0.20, 0.48)),
                vulnerability=float(min(1.0, rng.uniform(0.30, 0.68) + vulnerability_boost)),
                age=age, risk_tier=1, comorbidity_count=comorbidity_count,
                social_determinant_score=float(min(1.0, social_score)),
                education_level=education, geographic_access=geographic,
                genetic_risk_factor=genetic_risk, chronic_condition_type=chronic_type,
            )
        return PatientProfile(
            economic_capacity=float(rng.uniform(3200, 6000)),
            insurance_support=float(rng.uniform(0.42, 0.70)),
            vulnerability=float(min(1.0, rng.uniform(0.08, 0.38) + vulnerability_boost)),
            age=age, risk_tier=0, comorbidity_count=min(comorbidity_count, 2),
            social_determinant_score=float(min(1.0, social_score * 1.15)),
            education_level=max(education, 1),
            geographic_access=min(1.0, geographic * 1.1),
            genetic_risk_factor=genetic_risk, chronic_condition_type=chronic_type,
        )

    # Initial State
    def initial_state(self, rng) -> PatientState:
        init_severity = float(rng.uniform(0.40, 0.92))
        return PatientState(
            severity=init_severity,
            cumulative_cost=0.0,
            last_adherence=1.0,
            comorbidity_index=float(rng.uniform(0.0, 0.35)),
            treatment_resistance=0.0,
            time_step=0, previous_action=-1,
            consecutive_same_action=0,
            cumulative_adherence=0.0, total_steps=0,
            economic_shock_active=False, shock_remaining_steps=0,
            severity_trend=0.0,
            medication_fatigue=0.0,
            peak_severity=init_severity,
        )

    # State Discretisation (8-D)
    def discretize_state(
        self, profile: PatientProfile, state: PatientState
    ) -> Tuple[int, ...]:
        severity_bin = min(4, int(state.severity * 5.0))
        burden = state.cumulative_cost / max(profile.economic_capacity, 1.0)
        burden_bin = min(4, int(min(burden, 0.999) * 5.0))
        adherence_bin = 1 if state.last_adherence >= 0.5 else 0
        comorbidity_bin = min(2, int(state.comorbidity_index * 3.0))
        resistance_bin = min(2, int(state.treatment_resistance * 3.0))
        time_bin = min(2, state.time_step // 6) if state.time_step >= 0 else 0
        ins_bin = 0 if profile.insurance_support < 0.25 else (1 if profile.insurance_support < 0.50 else 2)
        age_bin = 0 if profile.age < 35 else (1 if profile.age < 60 else 2)
        return (severity_bin, burden_bin, adherence_bin,
                comorbidity_bin, resistance_bin, time_bin,
                ins_bin, age_bin)

    # Environment Step
    def step(self, profile: PatientProfile, state: PatientState, action_id: int, rng) -> dict:
        action = self.actions[action_id]
        disease_model = self._disease_model(profile.chronic_condition_type)

        # 1. Economic shock modelling
        shock_active = state.economic_shock_active
        shock_remaining = state.shock_remaining_steps
        if shock_active and shock_remaining > 0:
            shock_remaining -= 1
            if shock_remaining <= 0:
                shock_active = False
        elif not shock_active and rng.random() < self._economic_shock_prob:
            shock_active = True
            shock_remaining = int(rng.integers(2, 5))

        insurance_effective = profile.insurance_support
        capacity_effective = profile.economic_capacity
        if shock_active:
            insurance_effective *= 0.55
            capacity_effective *= 0.70

        # 2. Cost computation with inflation, geographic premium, pandemic surcharge
        base_cost = action["cost"]
        geo_premium = 1.0 + 0.15 * (1.0 - profile.geographic_access)
        inflation_factor = 1.0 + self.drift_cost_inflation
        episode_inflation = 1.0 + 0.003 * state.time_step

        # Pandemic wave: costs surge during pandemic peaks
        pandemic_cost_surge = 0.0
        if self._pandemic_wave_amplitude > 0:
            wave = self._pandemic_wave_amplitude * max(0.0, math.sin(
                2.0 * math.pi * state.time_step / self._pandemic_wave_period
            ))
            pandemic_cost_surge = wave * 0.25  # up to 25% cost increase at wave peak

        effective_cost = base_cost * (1.0 - insurance_effective) * inflation_factor * geo_premium * episode_inflation * (1.0 + pandemic_cost_surge)

        # 3. Adherence probability (multi-factor)
        projected_cost = state.cumulative_cost + effective_cost
        projected_rfb = projected_cost / max(capacity_effective, 1.0)

        adherence_prob = (
            0.90
            - 0.52 * projected_rfb
            - 0.09 * state.severity
            - 0.10 * profile.vulnerability
            + 0.08 * insurance_effective
            + 0.06 * profile.social_determinant_score
            + 0.04 * profile.education_level * 0.5
            + 0.03 * profile.geographic_access
            - 0.05 * state.comorbidity_index
            - 0.04 * state.treatment_resistance
            - 0.02 * state.medication_fatigue  # fatigue reduces willingness
        )

        seasonal_effect = self._seasonal_amplitude * math.sin(
            2.0 * math.pi * state.time_step / 12.0
        )
        adherence_prob -= abs(seasonal_effect)
        adherence_prob -= self.drift_adherence_drop

        if shock_active:
            adherence_prob -= 0.18

        # Pandemic wave reduces adherence
        if self._pandemic_wave_amplitude > 0:
            wave = self._pandemic_wave_amplitude * max(0.0, math.sin(
                2.0 * math.pi * state.time_step / self._pandemic_wave_period
            ))
            adherence_prob -= wave * 0.12

        adherence_prob = float(min(0.97, max(0.03, adherence_prob)))
        adhered = 1.0 if rng.random() < adherence_prob else 0.0

        # 4. Treatment efficacy with disease-specific modifiers
        base_efficacy = action["efficacy"] * (1.0 - self.drift_efficacy_drop)
        base_efficacy *= disease_model["treatment_response_mod"]

        resistance_penalty = state.treatment_resistance * 0.40
        fatigue_penalty = state.medication_fatigue * 0.25  # medication fatigue reduces efficacy
        effective_efficacy = base_efficacy * (1.0 - resistance_penalty) * (1.0 - fatigue_penalty)

        comorbidity_efficacy_mod = 1.0 - 0.18 * state.comorbidity_index
        comorbidity_risk_mod = 1.0 + 0.25 * state.comorbidity_index

        condition_bonus = 0.0
        if profile.chronic_condition_type == 1 and action["specialisation"] == "targeted":
            condition_bonus = 0.04
        elif profile.chronic_condition_type == 2 and action["specialisation"] == "experimental":
            condition_bonus = 0.06
        elif profile.chronic_condition_type == 3 and action["specialisation"] == "general":
            condition_bonus = 0.03

        interaction_penalty = 0.0
        if (state.previous_action >= 0 and state.previous_action != action_id
                and state.consecutive_same_action < self._interaction_penalty_window):
            interaction_penalty = action["interaction_sensitivity"] * 0.5

        age_vulnerability_mod = 1.0 - 0.20 * max(0.0, (profile.age - 55.0) / 35.0)

        # Genetic risk factor: increases severity progression
        genetic_severity_mod = 1.0 + 0.15 * profile.genetic_risk_factor

        noise = float(rng.normal(0.0, 0.025))

        if adhered > 0.5:
            improvement = (
                effective_efficacy
                * comorbidity_efficacy_mod
                * age_vulnerability_mod
                * (1.0 - 0.22 * profile.vulnerability)
                + condition_bonus
                - interaction_penalty
                + noise
            )
            improvement = max(0.0, improvement)
            next_severity = max(0.0, state.severity - improvement)
            cost_paid = effective_cost
        else:
            base_risk = action["risk"] * comorbidity_risk_mod
            deterioration = (
                0.08
                + 0.32 * base_risk
                + 0.18 * profile.vulnerability
                + 0.10 * state.comorbidity_index
                + 0.06 * profile.genetic_risk_factor
                + abs(noise)
            ) * genetic_severity_mod
            next_severity = min(1.0, state.severity + deterioration)
            cost_paid = 0.18 * effective_cost

        # 5. Disease-specific progression: natural severity drift
        next_severity += disease_model["severity_drift"]
        next_severity = min(1.0, next_severity)

        # Disease relapse: random severity spike
        if rng.random() < disease_model["relapse_prob"]:
            relapse_amount = float(rng.uniform(0.05, 0.15))
            next_severity = min(1.0, next_severity + relapse_amount)

        # 6. Treatment resistance accumulation
        new_resistance = state.treatment_resistance
        if adhered > 0.5:
            new_resistance += action["resistance_rate"]
            if state.previous_action == action_id:
                new_resistance += action["resistance_rate"] * 0.3 * min(state.consecutive_same_action, 4)
        new_resistance = max(0.0, min(1.0, new_resistance - self._resistance_decay))

        # 7. Medication fatigue
        new_fatigue = state.medication_fatigue
        if adhered > 0.5:
            fatigue_rate = action.get("fatigue_rate", 0.005)
            if state.previous_action == action_id:
                new_fatigue += fatigue_rate * (1.0 + 0.2 * state.consecutive_same_action)
            else:
                # Switching treatment reduces fatigue
                new_fatigue = max(0.0, new_fatigue - 0.02)
                new_fatigue += fatigue_rate * 0.3
        # Natural fatigue recovery
        new_fatigue = max(0.0, min(1.0, new_fatigue - 0.005))

        # 8. Comorbidity index evolution
        new_comorbidity = state.comorbidity_index
        if next_severity > 0.7:
            new_comorbidity += 0.008 * next_severity
        if next_severity < state.severity:
            new_comorbidity -= 0.003
        new_comorbidity += 0.001 * max(0, profile.age - 50) / 40.0
        # Complications cascade for disease-specific thresholds
        if next_severity > disease_model["complications_threshold"]:
            new_comorbidity += 0.006
        new_comorbidity = max(0.0, min(1.0, new_comorbidity + 0.002 * profile.comorbidity_count))

        # 9. Severity trend
        severity_delta = next_severity - state.severity
        alpha_trend = 0.3
        new_severity_trend = alpha_trend * severity_delta + (1 - alpha_trend) * state.severity_trend

        # 10. Peak severity tracking
        new_peak = max(state.peak_severity, next_severity)

        # 11. Action tracking
        if action_id == state.previous_action:
            new_consecutive = state.consecutive_same_action + 1
        else:
            new_consecutive = 1

        # 12. Construct next state
        next_state = PatientState(
            severity=float(next_severity),
            cumulative_cost=float(state.cumulative_cost + cost_paid),
            last_adherence=float(adhered),
            comorbidity_index=float(new_comorbidity),
            treatment_resistance=float(new_resistance),
            time_step=state.time_step + 1,
            previous_action=action_id,
            consecutive_same_action=new_consecutive,
            cumulative_adherence=state.cumulative_adherence + adhered,
            total_steps=state.total_steps + 1,
            economic_shock_active=shock_active,
            shock_remaining_steps=shock_remaining,
            severity_trend=float(new_severity_trend),
            medication_fatigue=float(new_fatigue),
            peak_severity=float(new_peak),
        )

        prev_rfb = state.cumulative_cost / max(capacity_effective, 1.0)
        next_rfb = next_state.cumulative_cost / max(capacity_effective, 1.0)

        return {
            "next_state": next_state,
            "adhered": adhered,
            "adherence_prob": adherence_prob,
            "risk": action["risk"] * comorbidity_risk_mod,
            "effective_cost": effective_cost,
            "prev_rfb": prev_rfb,
            "rfb": next_rfb,
            "clinical_improvement": state.severity - next_state.severity,
            "resistance_delta": new_resistance - state.treatment_resistance,
            "comorbidity_delta": new_comorbidity - state.comorbidity_index,
            "interaction_penalty": interaction_penalty,
            "economic_shock": shock_active,
            "severity_trend": new_severity_trend,
            "medication_fatigue": new_fatigue,
            "disease_model": disease_model["name"],
            "profile": {
                "economic_capacity": profile.economic_capacity,
                "age": profile.age,
                "risk_tier": profile.risk_tier,
                "vulnerability": profile.vulnerability,
                "comorbidity_count": profile.comorbidity_count,
                "chronic_condition_type": profile.chronic_condition_type,
                "social_determinant_score": profile.social_determinant_score,
                "geographic_access": profile.geographic_access,
                "genetic_risk_factor": profile.genetic_risk_factor,
            },
        }
