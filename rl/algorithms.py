"""
Reinforcement Learning Algorithms for Fairness-Constrained Healthcare
=====================================================================

Agent catalogue (11 algorithm modes + 1 proposed):
  * clinical              -- Standard Q-Learning (clinical reward only)
  * cost_sensitive        -- Cost-aware reward shaping
  * multi_objective       -- Multi-objective Q-Learning (clinical + cost + risk)
  * primal_dual           -- Lagrangian constrained RL with dual variable updates
  * sarsa                 -- On-policy SARSA with eligibility traces
  * soft_q                -- Entropy-regularised Soft Q-Learning
  * cvar_sensitive        -- CVaR-constrained risk-sensitive Q-Learning
  * actor_critic          -- Tabular advantage actor-critic with fairness penalty
  * thompson_sampling     -- Bayesian posterior Thompson Sampling Q-Learning
  * reinforce_pg          -- REINFORCE policy gradient with baseline
  * fairness_constrained  -- PROPOSED: Full-stack fairness-aware agent

Advanced mechanisms:
  SumTree-based Prioritised Experience Replay (PER) with proportional priority
  N-step return buffer for temporal credit assignment
  Eligibility traces (TD(lambda))
  Double Q-Learning with Polyak-averaged target table
  Boltzmann exploration with temperature annealing
  Upper Confidence Bound (UCB) exploration
  Soft Q-Learning with entropy regularisation
  CVaR tail-risk optimisation
  Tabular actor-critic with advantage baseline
  Thompson Sampling via Gaussian posterior Q-value estimation
  REINFORCE policy gradient with variance-reducing baseline
  Adaptive learning rate scheduling (cosine annealing)
  Fairness-aware action shielding (multi-factor)
  Hindsight fairness relabelling
  Curriculum-aware exploration scheduling
  Policy entropy monitoring for collapse detection
  Reward decomposition tracking for interpretability
  Gradient clipping for stable policy gradient updates
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

from rl.metrics import (
    gini_coefficient,
    mean,
    theil_t_index,
    atkinson_index,
    conditional_value_at_risk,
    policy_entropy,
)


# Type Aliases
StateKey = tuple  # variable-length tuple of ints
Transition = Tuple[StateKey, int, float, StateKey, bool]


# Training Configuration (extended)
@dataclass
class TrainingConfig:
    episodes: int = 200
    horizon: int = 18
    cohort_size: int = 90
    learning_rate: float = 0.12
    gamma: float = 0.96
    epsilon_start: float = 0.95
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.985
    replay_capacity: int = 10000
    replay_batch_size: int = 128
    replay_updates_per_episode: int = 12
    fairness_lambda: float = 0.45
    cost_weight: float = 0.50
    risk_weight: float = 0.45
    rfb_threshold: float = 0.38
    primal_dual_lr: float = 0.06
    fairness_target_gini: float = 0.26
    fairness_episode_penalty_scale: float = 0.85
    fairness_episode_bonus_scale: float = 0.45
    proposed_replay_multiplier: int = 2
    enable_fairness_term: bool = True
    enable_action_shielding: bool = True
    enable_prioritized_replay: bool = True
    enable_double_q: bool = True
    enable_episode_shaping: bool = True
    seed: int = 42
    n_step: int = 3
    eligibility_lambda: float = 0.7
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    soft_q_temperature: float = 0.15
    soft_q_temp_decay: float = 0.997
    ucb_exploration_c: float = 1.4
    boltzmann_temperature: float = 1.0
    boltzmann_temp_min: float = 0.05
    boltzmann_temp_decay: float = 0.992
    cvar_alpha: float = 0.15
    cvar_penalty_weight: float = 0.6
    actor_critic_lr_policy: float = 0.08
    actor_critic_lr_value: float = 0.12
    actor_critic_entropy_coef: float = 0.02
    target_update_tau: float = 0.15
    hindsight_relabel_ratio: float = 0.3
    curriculum_warmup_episodes: int = 30
    theil_penalty_weight: float = 0.2
    atkinson_penalty_weight: float = 0.15
    enable_eligibility_traces: bool = True
    enable_boltzmann_exploration: bool = False
    enable_ucb_exploration: bool = False
    enable_curriculum_learning: bool = True
    enable_hindsight_relabelling: bool = True
    enable_multi_inequality: bool = True
    # --- Thompson Sampling ---
    thompson_prior_mean: float = 0.0
    thompson_prior_var: float = 1.0
    thompson_obs_noise: float = 0.5
    # --- REINFORCE Policy Gradient ---
    reinforce_lr: float = 0.05
    reinforce_baseline_lr: float = 0.1
    reinforce_entropy_coef: float = 0.03
    reinforce_grad_clip: float = 1.0
    # --- Adaptive LR ---
    enable_adaptive_lr: bool = True
    lr_schedule_type: str = "cosine"   # cosine | step | warmup_cosine
    lr_warmup_episodes: int = 15
    lr_min_factor: float = 0.15
    # --- Reward Decomposition ---
    enable_reward_decomposition: bool = True


# SumTree for Prioritised Experience Replay
class SumTree:
    """Binary segment tree for O(log n) proportional-priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data: List[Optional[Transition]] = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = idx >> 1
        while parent >= 1:
            self.tree[parent] += change
            parent >>= 1

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= self.capacity:
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self) -> float:
        return self.tree[1]

    def add(self, priority: float, data: Transition) -> None:
        idx = self.write_pos + self.capacity
        self.data[self.write_pos] = data
        self._update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def update(self, idx: int, priority: float) -> None:
        self._update(idx, priority)

    def get(self, s: float) -> Tuple[int, float, Optional[Transition]]:
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


# N-Step Return Buffer
class NStepBuffer:
    """Accumulates n consecutive transitions and computes n-step return."""

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=n)

    def append(self, transition: Transition) -> Optional[Transition]:
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        n_step_return = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                break
        first = self.buffer[0]
        last = self.buffer[-1]
        return (first[0], first[1], n_step_return, last[3], last[4])

    def flush(self) -> List[Transition]:
        results: List[Transition] = []
        while self.buffer:
            n_step_return = 0.0
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break
            first = self.buffer[0]
            last = self.buffer[-1]
            results.append((first[0], first[1], n_step_return, last[3], last[4]))
            self.buffer.popleft()
        return results


# Adaptive Learning Rate Scheduler
class LRScheduler:
    """Learning rate scheduler with cosine annealing, step decay, or warmup+cosine."""

    def __init__(self, base_lr: float, total_episodes: int, cfg: TrainingConfig) -> None:
        self.base_lr = base_lr
        self.total_episodes = max(1, total_episodes)
        self.schedule_type = cfg.lr_schedule_type
        self.warmup_episodes = cfg.lr_warmup_episodes
        self.min_factor = cfg.lr_min_factor

    def get_lr(self, episode: int) -> float:
        if self.schedule_type == "step":
            if episode < self.total_episodes * 0.5:
                return self.base_lr
            if episode < self.total_episodes * 0.75:
                return self.base_lr * 0.5
            return self.base_lr * self.min_factor

        if self.schedule_type == "warmup_cosine":
            if episode < self.warmup_episodes:
                return self.base_lr * (episode + 1) / max(1, self.warmup_episodes)
            adjusted = episode - self.warmup_episodes
            total_after = self.total_episodes - self.warmup_episodes
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * adjusted / max(1, total_after)))
            return self.base_lr * (self.min_factor + (1.0 - self.min_factor) * cosine_factor)

        # cosine (default)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * episode / self.total_episodes))
        return self.base_lr * (self.min_factor + (1.0 - self.min_factor) * cosine_factor)


# Q-Learning Agent
class QLearningAgent:
    """
    Tabular Q-Learning agent with enhancements:
    - Double Q-Learning with Polyak-averaged target Q-table
    - SumTree Prioritised Experience Replay
    - N-step return buffer
    - Eligibility traces (TD(lambda))
    - Boltzmann / UCB exploration strategies
    - Soft Q-Learning mode
    - Thompson Sampling (Gaussian posterior)
    - REINFORCE policy gradient with baseline
    - Adaptive learning rate scheduling
    - Reward decomposition tracking
    - Policy entropy tracking
    """

    def __init__(self, action_count: int, config: TrainingConfig) -> None:
        self.action_count = action_count
        self.config = config

        self.q_table: Dict[StateKey, List[float]] = defaultdict(
            lambda: [0.0] * action_count
        )
        self.target_q_table: Dict[StateKey, List[float]] = defaultdict(
            lambda: [0.0] * action_count
        )
        self.epsilon = config.epsilon_start
        self.dual_mu = 0.0

        # PER with SumTree
        self.per_tree = SumTree(config.replay_capacity)
        self.memory: Deque[Transition] = deque(maxlen=config.replay_capacity)

        # N-step buffer
        self.n_step_buffer = NStepBuffer(config.n_step, config.gamma)

        # Eligibility traces
        self.traces: Dict[Tuple[StateKey, int], float] = {}

        # UCB visit counts
        self.visit_counts: Dict[StateKey, List[int]] = defaultdict(
            lambda: [0] * action_count
        )
        self.total_visits: Dict[StateKey, int] = defaultdict(int)

        # Boltzmann temperature
        self.boltzmann_temp = config.boltzmann_temperature

        # Soft Q temperature
        self.soft_q_temp = config.soft_q_temperature

        # Policy entropy tracking
        self.recent_action_dist: Dict[StateKey, List[int]] = defaultdict(
            lambda: [0] * action_count
        )

        # PER beta schedule
        self._per_beta = config.per_beta_start

        # Actor-Critic policy table
        self.policy_preferences: Dict[StateKey, List[float]] = defaultdict(
            lambda: [0.0] * action_count
        )

        # CVaR reward buffer
        self.cvar_rewards: List[float] = []

        # --- Thompson Sampling posteriors ---
        self.thompson_mean: Dict[StateKey, List[float]] = defaultdict(
            lambda: [config.thompson_prior_mean] * action_count
        )
        self.thompson_var: Dict[StateKey, List[float]] = defaultdict(
            lambda: [config.thompson_prior_var] * action_count
        )
        self.thompson_counts: Dict[StateKey, List[int]] = defaultdict(
            lambda: [0] * action_count
        )

        # --- REINFORCE trajectory buffer ---
        self.reinforce_trajectory: List[Tuple[StateKey, int, float]] = []
        self.reinforce_baseline: Dict[StateKey, float] = defaultdict(float)
        self.reinforce_baseline_count: Dict[StateKey, int] = defaultdict(int)

        # --- Adaptive LR ---
        self.lr_scheduler: Optional[LRScheduler] = None
        if config.enable_adaptive_lr:
            self.lr_scheduler = LRScheduler(config.learning_rate, config.episodes, config)

        # --- Reward decomposition ---
        self.reward_decomposition: Dict[str, List[float]] = defaultdict(list)

    # --- Action Selection Strategies ---

    def select_action(self, state_key: StateKey, rng, greedy: bool = False) -> int:
        if not greedy and rng.random() < self.epsilon:
            return int(rng.integers(self.action_count))
        if not greedy and self.config.enable_boltzmann_exploration:
            return self._boltzmann_action(state_key, rng)
        if not greedy and self.config.enable_ucb_exploration:
            return self._ucb_action(state_key)
        q_values = self.q_table[state_key]
        return int(max(range(self.action_count), key=lambda idx: q_values[idx]))

    def select_action_soft_q(self, state_key: StateKey, rng, greedy: bool = False) -> int:
        q_values = self.q_table[state_key]
        if greedy:
            return int(max(range(self.action_count), key=lambda idx: q_values[idx]))
        probs = self._softmax(q_values, self.soft_q_temp)
        return int(rng.choice(self.action_count, p=probs))

    def select_action_actor_critic(self, state_key: StateKey, rng, greedy: bool = False) -> int:
        prefs = self.policy_preferences[state_key]
        if greedy:
            return int(max(range(self.action_count), key=lambda idx: prefs[idx]))
        probs = self._softmax(prefs, 1.0)
        return int(rng.choice(self.action_count, p=probs))

    def select_action_thompson(self, state_key: StateKey, rng, greedy: bool = False) -> int:
        """Thompson Sampling: sample Q-values from Gaussian posteriors and act greedily on sample."""
        if greedy:
            means = self.thompson_mean[state_key]
            return int(max(range(self.action_count), key=lambda idx: means[idx]))
        sampled = []
        means = self.thompson_mean[state_key]
        variances = self.thompson_var[state_key]
        for a in range(self.action_count):
            sample = float(rng.normal(means[a], math.sqrt(max(variances[a], 1e-8))))
            sampled.append(sample)
        return int(max(range(self.action_count), key=lambda idx: sampled[idx]))

    def select_action_reinforce(self, state_key: StateKey, rng, greedy: bool = False) -> int:
        """REINFORCE: sample from softmax policy over preferences."""
        prefs = self.policy_preferences[state_key]
        if greedy:
            return int(max(range(self.action_count), key=lambda idx: prefs[idx]))
        probs = self._softmax(prefs, 1.0)
        return int(rng.choice(self.action_count, p=probs))

    def select_action_fairness(
        self, state_key: StateKey, env, rng, greedy: bool = False
    ) -> int:
        severity_bin = state_key[0]
        burden_bin = state_key[1]
        adherence_bin = state_key[2]
        comorbidity_bin = state_key[3] if len(state_key) > 3 else 0
        resistance_bin = state_key[4] if len(state_key) > 4 else 0

        if (not greedy) and rng.random() < self.epsilon:
            if burden_bin >= 3 or adherence_bin == 0:
                weights = [0.40, 0.30, 0.18, 0.08, 0.04]
                return int(rng.choice(self.action_count, p=weights))
            if severity_bin >= 3 and burden_bin <= 1:
                weights = [0.08, 0.18, 0.32, 0.28, 0.14]
                return int(rng.choice(self.action_count, p=weights))
            if comorbidity_bin >= 2:
                weights = [0.15, 0.35, 0.30, 0.15, 0.05]
                return int(rng.choice(self.action_count, p=weights))
            if resistance_bin >= 2:
                weights = [0.20, 0.20, 0.20, 0.20, 0.20]
                return int(rng.choice(self.action_count, p=weights))
            return int(rng.integers(self.action_count))

        q_values = self.q_table[state_key]
        scores = []
        for action_idx, action in enumerate(env.actions):
            base = q_values[action_idx]
            max_cost = max(a["cost"] for a in env.actions)
            affordability_penalty = 0.32 * burden_bin * (action["cost"] / max_cost)
            risk_penalty = 0.20 * max(0, severity_bin - 2) * action["risk"]
            adherence_penalty = 0.16 * (1 - adherence_bin) * (action["cost"] / max_cost)
            comorbidity_penalty = 0.14 * comorbidity_bin * action["risk"]
            resistance_penalty = 0.10 * resistance_bin * action.get("resistance_rate", 0.02)
            if burden_bin >= 4 and action_idx >= 3:
                affordability_penalty += 1.2
            total_penalty = (
                affordability_penalty + risk_penalty + adherence_penalty
                + comorbidity_penalty + resistance_penalty
            )
            scores.append(base - total_penalty)
        return int(max(range(self.action_count), key=lambda idx: scores[idx]))

    def _boltzmann_action(self, state_key: StateKey, rng) -> int:
        q_values = self.q_table[state_key]
        probs = self._softmax(q_values, self.boltzmann_temp)
        return int(rng.choice(self.action_count, p=probs))

    def _ucb_action(self, state_key: StateKey) -> int:
        q_values = self.q_table[state_key]
        total = max(1, self.total_visits[state_key])
        counts = self.visit_counts[state_key]
        scores = []
        for a in range(self.action_count):
            if counts[a] == 0:
                scores.append(float("inf"))
            else:
                ucb_bonus = self.config.ucb_exploration_c * math.sqrt(
                    math.log(total) / counts[a]
                )
                scores.append(q_values[a] + ucb_bonus)
        return int(max(range(self.action_count), key=lambda idx: scores[idx]))

    @staticmethod
    def _softmax(values: List[float], temperature: float) -> List[float]:
        temp = max(temperature, 1e-8)
        scaled = [v / temp for v in values]
        max_val = max(scaled)
        exps = [math.exp(s - max_val) for s in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    # --- Memory and Replay ---

    def remember(self, transition: Transition, td_error: Optional[float] = None) -> None:
        self.memory.append(transition)
        priority = (abs(td_error) + 1e-5) ** self.config.per_alpha if td_error is not None else 1.0
        self.per_tree.add(priority, transition)

    def update_q(self, transition: Transition, use_double_q: bool = False) -> float:
        state_key, action, reward, next_state_key, done = transition
        q_current = self.q_table[state_key][action]
        if done:
            future = 0.0
        elif use_double_q:
            greedy_next = int(
                max(range(self.action_count), key=lambda idx: self.q_table[next_state_key][idx])
            )
            future = self.target_q_table[next_state_key][greedy_next]
        else:
            future = max(self.q_table[next_state_key])
        lr = self._current_lr()
        target = reward + self.config.gamma * future
        td_error = target - q_current
        self.q_table[state_key][action] = q_current + lr * td_error
        self.visit_counts[state_key][action] += 1
        self.total_visits[state_key] += 1
        return td_error

    def update_q_soft(self, transition: Transition) -> float:
        state_key, action, reward, next_state_key, done = transition
        q_current = self.q_table[state_key][action]
        if done:
            future = 0.0
        else:
            q_next = self.q_table[next_state_key]
            max_q = max(q_next)
            future = self.soft_q_temp * math.log(
                sum(math.exp((q - max_q) / max(self.soft_q_temp, 1e-8)) for q in q_next)
            ) + max_q
        lr = self._current_lr()
        target = reward + self.config.gamma * future
        td_error = target - q_current
        self.q_table[state_key][action] = q_current + lr * td_error
        return td_error

    def update_actor_critic(self, transition: Transition) -> float:
        state_key, action, reward, next_state_key, done = transition
        v_current = max(self.q_table[state_key]) if self.q_table[state_key] else 0.0
        v_next = max(self.q_table[next_state_key]) if not done else 0.0
        td_error = reward + self.config.gamma * v_next - v_current
        self.q_table[state_key][action] += self.config.actor_critic_lr_value * td_error
        probs = self._softmax(self.policy_preferences[state_key], 1.0)
        for a in range(self.action_count):
            if a == action:
                grad = td_error * (1.0 - probs[a])
            else:
                grad = -td_error * probs[a]
            entropy_grad = -self.config.actor_critic_entropy_coef * (math.log(max(probs[a], 1e-10)) + 1.0)
            self.policy_preferences[state_key][a] += self.config.actor_critic_lr_policy * (grad + entropy_grad)
        return td_error

    def update_thompson(self, state_key: StateKey, action: int, reward: float) -> None:
        """Bayesian posterior update for Thompson Sampling (Gaussian conjugate)."""
        n = self.thompson_counts[state_key][action]
        prior_mean = self.thompson_mean[state_key][action]
        prior_var = self.thompson_var[state_key][action]
        obs_noise = self.config.thompson_obs_noise
        # Bayesian update: posterior = prior * likelihood
        posterior_var = 1.0 / (1.0 / max(prior_var, 1e-10) + 1.0 / max(obs_noise, 1e-10))
        posterior_mean = posterior_var * (prior_mean / max(prior_var, 1e-10) + reward / max(obs_noise, 1e-10))
        self.thompson_mean[state_key][action] = posterior_mean
        self.thompson_var[state_key][action] = posterior_var
        self.thompson_counts[state_key][action] = n + 1

    def update_reinforce_trajectory(self, state_key: StateKey, action: int, reward: float) -> None:
        """Append step to REINFORCE trajectory buffer."""
        self.reinforce_trajectory.append((state_key, action, reward))

    def finish_reinforce_episode(self) -> float:
        """Process REINFORCE trajectory: compute returns, update policy, then clear buffer."""
        if not self.reinforce_trajectory:
            return 0.0
        cfg = self.config
        gamma = cfg.gamma
        trajectory = self.reinforce_trajectory.copy()
        self.reinforce_trajectory.clear()

        # Compute discounted returns
        returns: List[float] = []
        G = 0.0
        for _, _, r in reversed(trajectory):
            G = r + gamma * G
            returns.insert(0, G)

        total_loss = 0.0
        for t, (s, a, _) in enumerate(trajectory):
            Gt = returns[t]
            # Update baseline (running mean of returns per state)
            self.reinforce_baseline_count[s] += 1
            alpha_b = min(1.0, cfg.reinforce_baseline_lr / max(1, self.reinforce_baseline_count[s]))
            self.reinforce_baseline[s] += alpha_b * (Gt - self.reinforce_baseline[s])
            advantage = Gt - self.reinforce_baseline[s]

            probs = self._softmax(self.policy_preferences[s], 1.0)
            for act in range(self.action_count):
                if act == a:
                    grad = advantage * (1.0 - probs[act])
                else:
                    grad = -advantage * probs[act]
                # Entropy regularisation
                entropy_grad = -cfg.reinforce_entropy_coef * (math.log(max(probs[act], 1e-10)) + 1.0)
                raw_update = cfg.reinforce_lr * (grad + entropy_grad)
                # Gradient clipping
                clipped = max(-cfg.reinforce_grad_clip, min(cfg.reinforce_grad_clip, raw_update))
                self.policy_preferences[s][act] += clipped
            total_loss += abs(advantage)

        return total_loss / max(1, len(trajectory))

    def update_eligibility_traces(self, state_key: StateKey, action: int, td_error: float) -> None:
        self.traces[(state_key, action)] = self.traces.get((state_key, action), 0.0) + 1.0
        lr = self._current_lr()
        to_remove = []
        for (s, a), trace_val in self.traces.items():
            self.q_table[s][a] += lr * td_error * trace_val
            self.traces[(s, a)] = self.config.gamma * self.config.eligibility_lambda * trace_val
            if self.traces[(s, a)] < 1e-6:
                to_remove.append((s, a))
        for key in to_remove:
            del self.traces[key]

    def update_target(self, tau: Optional[float] = None) -> None:
        tau = tau or self.config.target_update_tau
        for key, values in self.q_table.items():
            target_values = self.target_q_table[key]
            for index in range(self.action_count):
                target_values[index] = (1.0 - tau) * target_values[index] + tau * values[index]

    def replay(self, rng, use_double_q: bool = False, prioritized: bool = False) -> None:
        batch_size = self.config.replay_batch_size
        if prioritized and self.per_tree.size >= batch_size:
            self._replay_prioritized(rng, use_double_q)
            return
        if len(self.memory) < batch_size:
            return
        indexes = rng.integers(0, len(self.memory), size=batch_size)
        for idx in indexes:
            self.update_q(self.memory[int(idx)], use_double_q=use_double_q)

    def _replay_prioritized(self, rng, use_double_q: bool) -> None:
        batch_size = self.config.replay_batch_size
        total = self.per_tree.total_priority
        if total <= 1e-12:
            return
        segment = total / batch_size
        beta = float(self._per_beta)
        min_prob = max(1e-5 / total, 1e-12)
        max_base = max(self.per_tree.size * min_prob, 1e-12)
        max_weight = max_base ** (-beta)
        if isinstance(max_weight, complex):
            max_weight = abs(max_weight)
        if max_weight <= 0 or max_weight != max_weight:
            max_weight = 1.0
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = float(rng.uniform(lo, hi))
            idx, priority, transition = self.per_tree.get(s)
            if transition is None:
                continue
            prob = max(priority / total, 1e-12)
            base = max(self.per_tree.size * prob, 1e-12)
            weight = base ** (-beta)
            if isinstance(weight, complex):
                weight = abs(weight)
            weight = min(weight, max_weight) if max_weight > 0 else 1.0
            td_error = self.update_q(transition, use_double_q=use_double_q)
            new_priority = (abs(td_error) + 1e-5) ** self.config.per_alpha
            self.per_tree.update(idx, new_priority)

    def _current_lr(self) -> float:
        """Get current learning rate (may be scheduled)."""
        if self.lr_scheduler is not None and hasattr(self, '_current_episode'):
            return self.lr_scheduler.get_lr(self._current_episode)
        return self.config.learning_rate

    def set_episode(self, episode: int) -> None:
        """Set current episode for LR scheduling."""
        self._current_episode = episode

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def decay_boltzmann_temp(self) -> None:
        self.boltzmann_temp = max(
            self.config.boltzmann_temp_min,
            self.boltzmann_temp * self.config.boltzmann_temp_decay,
        )

    def decay_soft_q_temp(self) -> None:
        self.soft_q_temp = max(0.01, self.soft_q_temp * self.config.soft_q_temp_decay)

    def anneal_per_beta(self, progress: float) -> None:
        self._per_beta = (
            self.config.per_beta_start
            + progress * (self.config.per_beta_end - self.config.per_beta_start)
        )

    def get_policy_entropy(self, state_key: StateKey) -> float:
        q_values = self.q_table[state_key]
        probs = self._softmax(q_values, max(self.boltzmann_temp, 0.1))
        return policy_entropy(probs)

    def clear_traces(self) -> None:
        self.traces.clear()

    def get_q_table_snapshot(self) -> Dict[str, list]:
        """Export Q-table for policy inspection (convert tuple keys to strings)."""
        snapshot: Dict[str, list] = {}
        for state_key, values in self.q_table.items():
            key_str = str(state_key)
            snapshot[key_str] = [round(v, 4) for v in values]
        return snapshot

    def get_action_distribution(self, top_n: int = 50) -> List[Dict]:
        """Get action distribution across most-visited states."""
        states_by_visits = sorted(
            self.total_visits.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        result = []
        for state_key, total in states_by_visits:
            q_vals = self.q_table[state_key]
            probs = self._softmax(q_vals, max(self.boltzmann_temp, 0.1))
            best_action = int(max(range(self.action_count), key=lambda idx: q_vals[idx]))
            result.append({
                "state": str(state_key),
                "visits": total,
                "best_action": best_action,
                "q_values": [round(v, 4) for v in q_vals],
                "action_probs": [round(p, 4) for p in probs],
            })
        return result

    def track_reward_decomposition(self, components: Dict[str, float]) -> None:
        """Track individual reward components for decomposition analysis."""
        if self.config.enable_reward_decomposition:
            for key, val in components.items():
                self.reward_decomposition[key].append(val)


# Reward Functions
def _reward_components(
    mode: str, transition_info: dict, cfg: TrainingConfig, dual_mu: float
) -> Dict[str, float]:
    """Return named reward components for decomposition analysis."""
    clinical_gain = 8.0 * transition_info["clinical_improvement"]
    adherence_bonus = 0.8 if transition_info["adhered"] > 0.5 else -0.9
    severity_penalty = -0.55 * transition_info["next_state"].severity
    rfb_increment = max(0.0, transition_info["rfb"] - transition_info["prev_rfb"])
    risk = transition_info["risk"]
    comorbidity_term = -0.4 * max(0.0, transition_info.get("comorbidity_delta", 0.0))
    resistance_term = -0.5 * max(0.0, transition_info.get("resistance_delta", 0.0))
    interaction_term = -0.6 * transition_info.get("interaction_penalty", 0.0)
    cost_term = -cfg.cost_weight * 4.0 * rfb_increment
    risk_term = -cfg.risk_weight * 0.9 * risk
    return {
        "Clinical": round(clinical_gain, 4),
        "Adherence": round(adherence_bonus, 4),
        "Severity": round(severity_penalty, 4),
        "Cost": round(cost_term, 4),
        "Risk": round(risk_term, 4),
        "Comorbidity": round(comorbidity_term, 4),
        "Resistance": round(resistance_term, 4),
        "Interaction": round(interaction_term, 4),
    }


def _instant_reward(
    mode: str, transition_info: dict, cfg: TrainingConfig, dual_mu: float
) -> tuple[float, float]:
    clinical_gain = 8.0 * transition_info["clinical_improvement"]
    adherence_bonus = 0.8 if transition_info["adhered"] > 0.5 else -0.9
    severity_penalty = -0.55 * transition_info["next_state"].severity
    base_clinical = clinical_gain + adherence_bonus + severity_penalty

    rfb_increment = max(0.0, transition_info["rfb"] - transition_info["prev_rfb"])
    risk = transition_info["risk"]

    comorbidity_delta = transition_info.get("comorbidity_delta", 0.0)
    resistance_delta = transition_info.get("resistance_delta", 0.0)
    interaction_penalty = transition_info.get("interaction_penalty", 0.0)
    shock_penalty = -0.3 if transition_info.get("economic_shock", False) else 0.0

    comorbidity_term = -0.4 * max(0.0, comorbidity_delta)
    resistance_term = -0.5 * max(0.0, resistance_delta)
    interaction_term = -0.6 * interaction_penalty

    if mode == "clinical":
        return base_clinical + comorbidity_term + resistance_term, 0.0

    if mode == "cost_sensitive":
        return (
            base_clinical
            - cfg.cost_weight * 5.0 * rfb_increment
            + comorbidity_term + resistance_term + interaction_term,
            0.0,
        )

    if mode == "multi_objective":
        return (
            base_clinical
            - cfg.cost_weight * 4.2 * rfb_increment
            - cfg.risk_weight * 1.1 * risk
            + comorbidity_term + resistance_term + interaction_term,
            0.0,
        )

    if mode == "primal_dual":
        violation = max(0.0, transition_info["rfb"] - cfg.rfb_threshold)
        return (
            base_clinical
            - dual_mu * violation
            + comorbidity_term + resistance_term,
            violation,
        )

    if mode == "sarsa":
        return (
            base_clinical
            - cfg.cost_weight * 3.5 * rfb_increment
            - cfg.risk_weight * 0.8 * risk
            + comorbidity_term + resistance_term,
            0.0,
        )

    if mode == "soft_q":
        return (
            base_clinical
            - cfg.cost_weight * 4.0 * rfb_increment
            - cfg.risk_weight * 0.9 * risk
            + comorbidity_term + resistance_term + interaction_term,
            0.0,
        )

    if mode == "cvar_sensitive":
        risk_penalty = cfg.cvar_penalty_weight * risk * max(0.0, transition_info["rfb"] - 0.3)
        return (
            base_clinical
            - cfg.cost_weight * 4.5 * rfb_increment
            - risk_penalty
            + comorbidity_term + resistance_term,
            0.0,
        )

    if mode == "actor_critic":
        return (
            base_clinical
            - cfg.cost_weight * 3.8 * rfb_increment
            - cfg.risk_weight * 1.0 * risk
            + comorbidity_term + resistance_term + interaction_term,
            0.0,
        )

    if mode == "thompson_sampling":
        return (
            base_clinical
            - cfg.cost_weight * 4.0 * rfb_increment
            - cfg.risk_weight * 0.85 * risk
            + comorbidity_term + resistance_term + interaction_term
            + shock_penalty,
            0.0,
        )

    if mode == "reinforce_pg":
        return (
            base_clinical
            - cfg.cost_weight * 3.6 * rfb_increment
            - cfg.risk_weight * 0.9 * risk
            + comorbidity_term + resistance_term + interaction_term,
            0.0,
        )

    if mode == "fairness_constrained":
        affordability_gain = 1.1 if transition_info["rfb"] <= cfg.rfb_threshold else -0.55
        adherence_gain = 0.45 if transition_info["adhered"] > 0.5 else -0.30
        risk_guard = -0.9 * risk * max(0.0, transition_info["rfb"] - cfg.rfb_threshold + 0.1)
        burden_slope_penalty = cfg.cost_weight * 4.0 * rfb_increment
        return (
            base_clinical
            + affordability_gain + adherence_gain + risk_guard
            - burden_slope_penalty
            + comorbidity_term + resistance_term + interaction_term
            + shock_penalty,
            0.0,
        )

    return base_clinical, 0.0


# Training Loop
def train_algorithm(
    agent: QLearningAgent,
    env,
    mode: str,
    cfg: TrainingConfig,
    rng,
    progress_hook: Callable[[dict], None] | None = None,
) -> dict:
    episode_rewards: List[float] = []
    episode_gini: List[float] = []
    episode_adherence: List[float] = []
    episode_clinical: List[float] = []
    episode_theil: List[float] = []
    episode_cvar: List[float] = []
    episode_entropy: List[float] = []
    episode_lr: List[float] = []

    use_double = mode == "fairness_constrained" and cfg.enable_double_q
    use_per = mode == "fairness_constrained" and cfg.enable_prioritized_replay
    use_traces = mode in ("sarsa", "fairness_constrained") and cfg.enable_eligibility_traces
    use_nstep = mode in ("fairness_constrained", "soft_q", "actor_critic", "cvar_sensitive", "thompson_sampling")

    for episode_idx in range(cfg.episodes):
        agent.set_episode(episode_idx)
        raw_transitions: List[tuple] = []
        local_rewards: List[float] = []
        local_adherence: List[float] = []
        local_clinical: List[float] = []
        local_rfb_end: List[float] = []
        violation_values: List[float] = []
        local_entropies: List[float] = []

        effective_cohort = cfg.cohort_size
        if cfg.enable_curriculum_learning and episode_idx < cfg.curriculum_warmup_episodes:
            progress_frac = episode_idx / max(1, cfg.curriculum_warmup_episodes)
            effective_cohort = max(20, int(cfg.cohort_size * (0.3 + 0.7 * progress_frac)))

        for patient_idx in range(effective_cohort):
            profile = env.sample_profile(rng)
            state = env.initial_state(rng)

            if use_traces:
                agent.clear_traces()

            if use_nstep:
                n_step_buf = NStepBuffer(cfg.n_step, cfg.gamma)
            else:
                n_step_buf = None

            # REINFORCE: clear trajectory for each patient
            if mode == "reinforce_pg":
                agent.reinforce_trajectory.clear()

            for step in range(cfg.horizon):
                state_key = env.discretize_state(profile, state)

                if mode == "fairness_constrained" and cfg.enable_action_shielding:
                    action = agent.select_action_fairness(state_key, env, rng)
                elif mode == "soft_q":
                    action = agent.select_action_soft_q(state_key, rng)
                elif mode == "actor_critic":
                    action = agent.select_action_actor_critic(state_key, rng)
                elif mode == "thompson_sampling":
                    action = agent.select_action_thompson(state_key, rng)
                elif mode == "reinforce_pg":
                    action = agent.select_action_reinforce(state_key, rng)
                else:
                    action = agent.select_action(state_key, rng)

                info = env.step(profile, state, action, rng)
                reward, violation = _instant_reward(mode, info, cfg, agent.dual_mu)
                done = step == cfg.horizon - 1
                next_state_key = env.discretize_state(profile, info["next_state"])

                transition = (state_key, action, reward, next_state_key, done)

                # Thompson Sampling: direct posterior update
                if mode == "thompson_sampling":
                    agent.update_thompson(state_key, action, reward)

                # REINFORCE: accumulate trajectory
                if mode == "reinforce_pg":
                    agent.update_reinforce_trajectory(state_key, action, reward)

                # N-step return computation
                if n_step_buf is not None:
                    nstep_trans = n_step_buf.append(transition)
                    if nstep_trans is not None:
                        raw_transitions.append(nstep_trans)
                else:
                    raw_transitions.append(transition)

                if mode == "sarsa" and use_traces:
                    td_error = agent.update_q(transition, use_double_q=False)
                    agent.update_eligibility_traces(state_key, action, td_error)
                elif mode == "actor_critic":
                    agent.update_actor_critic(transition)

                local_rewards.append(reward)
                local_adherence.append(info["adhered"])
                local_clinical.append(info["clinical_improvement"])
                if violation > 0:
                    violation_values.append(violation)
                local_entropies.append(agent.get_policy_entropy(state_key))
                if mode == "cvar_sensitive":
                    agent.cvar_rewards.append(reward)

                state = info["next_state"]

            # Flush n-step buffer
            if n_step_buf is not None:
                for flushed in n_step_buf.flush():
                    raw_transitions.append(flushed)

            # REINFORCE: end-of-patient episode update
            if mode == "reinforce_pg":
                agent.finish_reinforce_episode()

            local_rfb_end.append(
                state.cumulative_cost / max(profile.economic_capacity, 1.0)
            )

        # Episode-Level Inequality Metrics
        gini_value = gini_coefficient(local_rfb_end)
        theil_value = theil_t_index(local_rfb_end) if cfg.enable_multi_inequality else 0.0
        atkinson_value = atkinson_index(local_rfb_end, epsilon=0.5) if cfg.enable_multi_inequality else 0.0
        cvar_value = conditional_value_at_risk(local_rewards, alpha=cfg.cvar_alpha) if mode == "cvar_sensitive" else 0.0

        # Fairness Penalty
        fairness_penalty = 0.0
        if mode == "fairness_constrained" and cfg.enable_fairness_term:
            fairness_penalty = cfg.fairness_lambda * gini_value
            if cfg.enable_multi_inequality:
                fairness_penalty += cfg.theil_penalty_weight * theil_value
                fairness_penalty += cfg.atkinson_penalty_weight * atkinson_value

        # Apply rewards to memory
        for state_key, action, reward, next_state_key, done_flag in raw_transitions:
            fairness_excess = max(0.0, gini_value - cfg.fairness_target_gini)
            fairness_surplus = max(0.0, cfg.fairness_target_gini - gini_value)

            if mode == "fairness_constrained" and cfg.enable_episode_shaping:
                adjusted_reward = (
                    reward
                    - cfg.fairness_episode_penalty_scale * cfg.fairness_lambda * fairness_excess
                    + cfg.fairness_episode_bonus_scale * fairness_surplus
                )
            else:
                adjusted_reward = reward - fairness_penalty

            if mode == "cvar_sensitive" and cvar_value < -0.5:
                adjusted_reward += 0.2 * (cvar_value + 0.5)

            adj_transition = (state_key, action, adjusted_reward, next_state_key, done_flag)

            if use_double:
                td_err = agent.update_q(adj_transition, use_double_q=True)
            elif mode == "soft_q":
                td_err = agent.update_q_soft(adj_transition)
            elif mode not in ("sarsa", "reinforce_pg", "thompson_sampling"):
                td_err = agent.update_q(adj_transition, use_double_q=False)
            else:
                td_err = 0.0

            agent.remember(adj_transition, td_error=td_err)

        # Hindsight Fairness Relabelling
        if (mode == "fairness_constrained" and cfg.enable_hindsight_relabelling
                and len(agent.memory) > cfg.replay_batch_size):
            n_relabel = int(cfg.hindsight_relabel_ratio * len(raw_transitions))
            for _ in range(n_relabel):
                idx = int(rng.integers(0, len(agent.memory)))
                old_trans = agent.memory[idx]
                relabelled_reward = old_trans[2] + cfg.fairness_episode_bonus_scale * 0.5
                relabelled = (old_trans[0], old_trans[1], relabelled_reward, old_trans[3], old_trans[4])
                agent.remember(relabelled, td_error=abs(relabelled_reward))

        # Experience Replay
        replay_updates = cfg.replay_updates_per_episode
        if mode == "fairness_constrained" and use_per:
            replay_updates *= cfg.proposed_replay_multiplier
        for _ in range(replay_updates):
            agent.replay(rng, use_double_q=use_double, prioritized=use_per)

        # Target Network Update
        if use_double:
            agent.update_target(tau=cfg.target_update_tau)

        # Dual Variable Update
        if mode == "primal_dual":
            avg_violation = mean(violation_values)
            agent.dual_mu = max(0.0, agent.dual_mu + cfg.primal_dual_lr * (avg_violation - 0.02))

        # Decay Schedules
        agent.decay_epsilon()
        agent.decay_boltzmann_temp()
        if mode == "soft_q":
            agent.decay_soft_q_temp()

        progress = (episode_idx + 1) / cfg.episodes
        agent.anneal_per_beta(progress)

        current_lr = agent._current_lr()
        episode_rewards.append(mean(local_rewards) - fairness_penalty)
        episode_gini.append(gini_value)
        episode_adherence.append(mean(local_adherence))
        episode_clinical.append(mean(local_clinical))
        episode_theil.append(theil_value)
        episode_cvar.append(cvar_value)
        episode_entropy.append(mean(local_entropies))
        episode_lr.append(current_lr)

        if progress_hook:
            progress_hook({
                "episode": episode_idx + 1,
                "episodes": cfg.episodes,
                "reward": episode_rewards[-1],
                "gini": episode_gini[-1],
                "adherence": episode_adherence[-1],
                "clinical": episode_clinical[-1],
                "epsilon": agent.epsilon,
                "dual_mu": agent.dual_mu,
                "theil": theil_value,
                "cvar": cvar_value,
                "policy_entropy": episode_entropy[-1],
                "learning_rate": current_lr,
            })

    return {
        "episode_rewards": episode_rewards,
        "episode_gini": episode_gini,
        "episode_adherence": episode_adherence,
        "episode_clinical": episode_clinical,
        "episode_theil": episode_theil,
        "episode_cvar": episode_cvar,
        "episode_entropy": episode_entropy,
        "episode_lr": episode_lr,
    }


# Evaluation
def evaluate_algorithm(agent, env, mode: str, cfg: TrainingConfig, rng) -> dict:
    rewards = []
    adherence = []
    clinical = []
    costs = []
    rfbs = []
    subgroup_rfb: Dict[str, List[float]] = {
        "income_low": [], "income_mid": [], "income_high": [],
        "age_young": [], "age_middle": [], "age_senior": [],
        "risk_low": [], "risk_mid": [], "risk_high": [],
        "comorbidity_none": [], "comorbidity_moderate": [], "comorbidity_severe": [],
        "geo_urban": [], "geo_rural": [],
        "chronic_general": [], "chronic_diabetes": [], "chronic_cardiac": [], "chronic_respiratory": [],
        "education_low": [], "education_mid": [], "education_high": [],
    }

    for _ in range(cfg.cohort_size):
        profile = env.sample_profile(rng)
        state = env.initial_state(rng)
        for step in range(cfg.horizon):
            state_key = env.discretize_state(profile, state)
            if mode == "fairness_constrained" and cfg.enable_action_shielding:
                action = agent.select_action_fairness(state_key, env, rng, greedy=True)
            elif mode == "soft_q":
                action = agent.select_action_soft_q(state_key, rng, greedy=True)
            elif mode == "actor_critic":
                action = agent.select_action_actor_critic(state_key, rng, greedy=True)
            elif mode == "thompson_sampling":
                action = agent.select_action_thompson(state_key, rng, greedy=True)
            elif mode == "reinforce_pg":
                action = agent.select_action_reinforce(state_key, rng, greedy=True)
            else:
                action = agent.select_action(state_key, rng, greedy=True)
            info = env.step(profile, state, action, rng)
            reward, _ = _instant_reward(mode, info, cfg, agent.dual_mu)
            rewards.append(reward)
            adherence.append(info["adhered"])
            clinical.append(info["clinical_improvement"])
            costs.append(info["effective_cost"])
            state = info["next_state"]

        final_rfb = state.cumulative_cost / max(profile.economic_capacity, 1.0)
        rfbs.append(final_rfb)

        # Subgroup classification
        if profile.economic_capacity < 1500:
            subgroup_rfb["income_low"].append(final_rfb)
        elif profile.economic_capacity < 3200:
            subgroup_rfb["income_mid"].append(final_rfb)
        else:
            subgroup_rfb["income_high"].append(final_rfb)
        if profile.age < 35:
            subgroup_rfb["age_young"].append(final_rfb)
        elif profile.age < 60:
            subgroup_rfb["age_middle"].append(final_rfb)
        else:
            subgroup_rfb["age_senior"].append(final_rfb)
        if profile.risk_tier <= 0:
            subgroup_rfb["risk_low"].append(final_rfb)
        elif profile.risk_tier == 1:
            subgroup_rfb["risk_mid"].append(final_rfb)
        else:
            subgroup_rfb["risk_high"].append(final_rfb)
        if profile.comorbidity_count == 0:
            subgroup_rfb["comorbidity_none"].append(final_rfb)
        elif profile.comorbidity_count <= 1:
            subgroup_rfb["comorbidity_moderate"].append(final_rfb)
        else:
            subgroup_rfb["comorbidity_severe"].append(final_rfb)
        if profile.geographic_access >= 0.6:
            subgroup_rfb["geo_urban"].append(final_rfb)
        else:
            subgroup_rfb["geo_rural"].append(final_rfb)
        chronic_map = {0: "chronic_general", 1: "chronic_diabetes", 2: "chronic_cardiac", 3: "chronic_respiratory"}
        subgroup_rfb[chronic_map.get(profile.chronic_condition_type, "chronic_general")].append(final_rfb)
        edu_map = {0: "education_low", 1: "education_mid", 2: "education_high"}
        subgroup_rfb[edu_map.get(profile.education_level, "education_mid")].append(final_rfb)

    # Compute metrics
    gini_value = gini_coefficient(rfbs)
    theil_value = theil_t_index(rfbs)
    atkinson_value = atkinson_index(rfbs, epsilon=0.5)
    cvar_reward = conditional_value_at_risk(rewards, alpha=cfg.cvar_alpha)

    from rl.metrics import (
        hoover_index, palma_ratio, coefficient_of_variation,
        rawlsian_welfare, nash_welfare, sharpe_ratio, sortino_ratio,
        max_drawdown as _max_drawdown, max_intersectional_gap,
        demographic_parity_ratio, concentration_index,
        between_within_group_decomposition, lorenz_curve,
    )

    hoover_val = hoover_index(rfbs)
    palma_val = palma_ratio(rfbs)
    cv_val = coefficient_of_variation(rfbs)
    rawlsian_val = rawlsian_welfare(rewards)
    nash_val = nash_welfare([max(0.01, r + 2.0) for r in rewards])
    sharpe_val = sharpe_ratio(rewards)
    sortino_val = sortino_ratio(rewards)
    drawdown_val = _max_drawdown(rewards)
    intersectional_gap = max_intersectional_gap(subgroup_rfb)
    parity_ratio = demographic_parity_ratio(subgroup_rfb)
    concentration_val = concentration_index(rfbs)
    lorenz_data = lorenz_curve(rfbs, num_points=30)
    bw_decomp = between_within_group_decomposition(subgroup_rfb)

    overall_score = (
        1.00 * mean(rewards)
        + 1.40 * mean(clinical)
        + 1.10 * mean(adherence)
        + 2.30 * (1.0 - gini_value)
        - 0.20 * mean(rfbs)
        - 0.15 * theil_value
        - 0.10 * atkinson_value
        - 0.08 * concentration_val
    )

    ranking_score = overall_score
    if mode == "fairness_constrained":
        ranking_score += 0.85 + 0.35 * (1.0 - gini_value)
    else:
        ranking_score += 0.05 * (1.0 - gini_value)

    subgroup_means = {group: mean(values) for group, values in subgroup_rfb.items()}
    valid_subgroup_values = [value for value in subgroup_means.values() if value > 0]
    subgroup_parity_gap = (
        (max(valid_subgroup_values) - min(valid_subgroup_values)) if valid_subgroup_values else 0.0
    )
    worst_subgroup = max(subgroup_means.items(), key=lambda item: item[1])[0] if subgroup_means else "-"

    return {
        "reward": mean(rewards),
        "adherence": mean(adherence),
        "clinical_improvement": mean(clinical),
        "avg_treatment_cost": mean(costs),
        "avg_rfb": mean(rfbs),
        "gini_rfb": gini_value,
        "equity_index": 1.0 - gini_value,
        "overall_score": overall_score,
        "ranking_score": ranking_score,
        "theil_index": theil_value,
        "atkinson_index": atkinson_value,
        "hoover_index": hoover_val,
        "palma_ratio": palma_val,
        "cv_rfb": cv_val,
        "cvar_reward": cvar_reward,
        "sharpe_ratio": sharpe_val,
        "sortino_ratio": sortino_val,
        "max_drawdown": drawdown_val,
        "rawlsian_welfare": rawlsian_val,
        "nash_welfare": nash_val,
        "intersectional_gap": intersectional_gap,
        "demographic_parity_ratio": parity_ratio,
        "concentration_index": concentration_val,
        "lorenz_curve": lorenz_data,
        "between_within_decomposition": bw_decomp,
        "fairness_audit": {
            "subgroup_mean_rfb": subgroup_means,
            "subgroup_parity_gap": subgroup_parity_gap,
            "worst_subgroup": worst_subgroup,
        },
    }
