"""
Metrics Module for Fairness-Constrained RL
-------------------------------------------
Comprehensive metrics suite for evaluating health-equity outcomes.

Categories:
 1. Inequality indices: Gini, Theil-L, Theil-T, Atkinson, Hoover,
    Palma, Generalized Entropy, Concentration Index
 2. Statistical tests: Mann-Whitney U, Wilcoxon Signed-Rank,
    KS 2-sample, Bootstrap BCa CI
 3. Welfare functions: Utilitarian, Rawlsian, Nash, Sen, Prioritarian
 4. Risk metrics: CVaR, VaR, Sharpe, Sortino, Max Drawdown
 5. Divergence measures: KL, JS, Wasserstein-1, Total Variation,
    Hellinger, Chi-Square
 6. Convergence diagnostics: Autocorrelation, EWMA Residual Variance,
    ESS, Geweke Z, R-hat (split chain)
 7. Fairness metrics: Equalized Opportunity Difference,
    Between-Within Group Decomposition, Demographic Parity Gap,
    Intersectional Disparity Ratio
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


_EPS = 1e-12


# ── 0. Basic Utilities ──────────────────────────────────────────


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def coefficient_of_variation(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    if abs(mu) < _EPS:
        return 0.0
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return float(math.sqrt(var) / abs(mu))


def conditional_value_at_risk(values: List[float], alpha: float = 0.05) -> float:
    """Alias for CVaR function (used by algorithms module)."""
    return cvar(values, alpha)


def policy_entropy(action_data, n_actions: int = 5) -> float:
    """Shannon entropy of the action distribution. Higher = more exploratory.
    Accepts either a dict of {action: count} or a list of probabilities."""
    if isinstance(action_data, dict):
        total = sum(action_data.values())
        if total == 0 or n_actions < 2:
            return 0.0
        probs = [action_data.get(a, 0) / total for a in range(n_actions)]
    elif isinstance(action_data, (list, tuple)):
        probs = list(action_data)
        n_actions = len(probs) if len(probs) > 0 else n_actions
    else:
        return 0.0
    if n_actions < 2:
        return 0.0
    ent = 0.0
    for p in probs:
        if p > _EPS:
            ent -= p * math.log(p)
    max_ent = math.log(n_actions)
    return float(ent / max_ent) if max_ent > _EPS else 0.0


def max_intersectional_gap(subgroup_outcomes: Dict[str, List[float]]) -> float:
    """Maximum gap between best and worst subgroup mean outcomes."""
    means = []
    for vals in subgroup_outcomes.values():
        if vals:
            means.append(sum(vals) / len(vals))
    if len(means) < 2:
        return 0.0
    return float(max(means) - min(means))


def demographic_parity_ratio(subgroup_outcomes: Dict[str, List[float]]) -> float:
    """Ratio of worst subgroup mean to best subgroup mean. 1.0 = perfect parity."""
    means = []
    for vals in subgroup_outcomes.values():
        if vals:
            means.append(sum(vals) / len(vals))
    if len(means) < 2:
        return 1.0
    best = max(means)
    worst = min(means)
    if best < _EPS:
        return 1.0
    return float(worst / best)


#  1. Inequality Indices 


def gini_coefficient(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    s = sorted(values)
    total = sum(s)
    if total <= _EPS:
        return 0.0
    cum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(s, 1):
        cum += v
        weighted_sum += i * v
    return float((2.0 * weighted_sum) / (n * total) - (n + 1.0) / n)


def theil_l_index(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    pos = [v for v in values if v > _EPS]
    if len(pos) < 2:
        return 0.0
    mu = sum(pos) / len(pos)
    return float(sum(math.log(mu / v) for v in pos) / len(pos))


def theil_t_index(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    pos = [v for v in values if v > _EPS]
    if len(pos) < 2:
        return 0.0
    mu = sum(pos) / len(pos)
    return float(sum((v / mu) * math.log(v / mu) for v in pos) / len(pos))


def atkinson_index(values: List[float], epsilon: float = 0.5) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    pos = [v for v in values if v > _EPS]
    if len(pos) < 2:
        return 0.0
    mu = sum(pos) / len(pos)
    if abs(epsilon - 1.0) < 1e-8:
        log_mean = sum(math.log(v) for v in pos) / len(pos)
        return float(1.0 - math.exp(log_mean) / mu)
    p = 1.0 - epsilon
    mean_p = sum((v / mu) ** p for v in pos) / len(pos)
    return float(1.0 - mean_p ** (1.0 / p))


def hoover_index(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total <= _EPS:
        return 0.0
    mu = total / n
    return float(sum(abs(v - mu) for v in values) / (2.0 * total))


def palma_ratio(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    s = sorted(values)
    k40 = max(1, int(0.4 * n))
    k90 = max(k40 + 1, int(0.9 * n))
    bottom_40 = sum(s[:k40])
    top_10 = sum(s[k90:])
    if bottom_40 <= _EPS:
        return float("inf")
    return float(top_10 / bottom_40)


def generalized_entropy_index(values: List[float], alpha: float = 2.0) -> float:
    """
    Generalized Entropy Index GE(alpha).
    alpha=0 -> Theil-L (mean log deviation)
    alpha=1 -> Theil-T
    alpha=2 -> Half squared coefficient of variation
    """
    n = len(values)
    if n < 2:
        return 0.0
    pos = [v for v in values if v > _EPS]
    if len(pos) < 2:
        return 0.0
    mu = sum(pos) / len(pos)
    if abs(alpha) < 1e-8:
        return theil_l_index(pos)
    if abs(alpha - 1.0) < 1e-8:
        return theil_t_index(pos)
    factor = 1.0 / (alpha * (alpha - 1.0))
    inner = sum(((v / mu) ** alpha - 1.0) for v in pos) / len(pos)
    return float(factor * inner)


def concentration_index(health_values: List[float], rank_values: Optional[List[float]] = None) -> float:
    """
    Concentration Index for health inequality.
    If rank_values is None, uses natural ordering (index-based ranking).
    Returns value in [-1, 1]. Negative = concentrated among disadvantaged.
    """
    n = len(health_values)
    if n < 2:
        return 0.0
    if rank_values is None:
        rank_values = list(range(n))
    if len(rank_values) != n:
        return 0.0
    pairs = sorted(zip(rank_values, health_values), key=lambda x: x[0])
    h = [p[1] for p in pairs]
    mu_h = sum(h) / n
    if abs(mu_h) < _EPS:
        return 0.0
    weighted = sum((2.0 * (i + 1) / n - 1.0 - 1.0 / n) * h[i] for i in range(n))
    return float(weighted / (n * mu_h))


def lorenz_curve(values: List[float], num_points: int = 20) -> Dict[str, List[float]]:
    """
    Compute Lorenz curve points.
    Returns dict with 'x' (cumulative population share) and 'y' (cumulative value share).
    """
    if not values:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    s = sorted(values)
    n = len(s)
    total = sum(s)
    if total <= _EPS:
        xs = [i / num_points for i in range(num_points + 1)]
        return {"x": xs, "y": xs}
    xs = [0.0]
    ys = [0.0]
    cum = 0.0
    step = max(1, n // num_points)
    for i in range(0, n, step):
        chunk = s[i:i + step]
        cum += sum(chunk)
        xs.append((i + len(chunk)) / n)
        ys.append(cum / total)
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(1.0)
    return {"x": xs, "y": ys}


#  2. Statistical Tests 


def mann_whitney_u(a: List[float], b: List[float]) -> Dict[str, float]:
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"u_stat": 0.0, "z": 0.0, "p_approx": 1.0}
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ranks = _assign_ranks(combined)
    r1 = sum(r for r, g in zip(ranks, combined) if g[1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu_u = n1 * n2 / 2.0
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0) + _EPS
    z = (u1 - mu_u) / sigma_u
    p_approx = 2.0 * _norm_cdf(-abs(z))
    return {"u_stat": float(u1), "z": float(z), "p_approx": float(p_approx)}


def wilcoxon_signed_rank(diffs: List[float]) -> Dict[str, float]:
    non_zero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > _EPS]
    if len(non_zero) < 2:
        return {"w_stat": 0.0, "z": 0.0, "p_approx": 1.0}
    non_zero.sort(key=lambda x: x[0])
    n = len(non_zero)
    w_plus = 0.0
    for i, (_, sign) in enumerate(non_zero, 1):
        if sign > 0:
            w_plus += i
    mu_w = n * (n + 1) / 4.0
    sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0) + _EPS
    z = (w_plus - mu_w) / sigma_w
    p_approx = 2.0 * _norm_cdf(-abs(z))
    return {"w_stat": float(w_plus), "z": float(z), "p_approx": float(p_approx)}


def ks_2sample(a: List[float], b: List[float]) -> Dict[str, float]:
    if not a or not b:
        return {"d_stat": 0.0, "p_approx": 1.0}
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    merged = sorted(set(sa + sb))
    d_max = 0.0
    for x in merged:
        fa = _ecdf_val(sa, x)
        fb = _ecdf_val(sb, x)
        d_max = max(d_max, abs(fa - fb))
    ne = (na * nb) / (na + nb + _EPS)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne + _EPS)) * d_max
    p_approx = max(0.0, min(1.0, 2.0 * math.exp(-2.0 * lam * lam)))
    return {"d_stat": float(d_max), "p_approx": float(p_approx)}


def bootstrap_bca_ci(
    data: List[float], stat_fn=None, n_boot: int = 2000,
    alpha: float = 0.05, rng=None,
) -> Dict[str, float]:
    import numpy as _np
    if rng is None:
        rng = _np.random.default_rng(42)
    if stat_fn is None:
        stat_fn = lambda x: sum(x) / max(len(x), 1)
    n = len(data)
    if n < 2:
        val = stat_fn(data)
        return {"low": val, "high": val, "point": val}
    arr = _np.array(data, dtype=_np.float64)
    theta_hat = float(stat_fn(data))
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True).tolist()
        boots.append(stat_fn(sample))
    boots.sort()
    z0 = _ppf(_norm_cdf, sum(1 for b in boots if b < theta_hat) / max(len(boots), 1))
    jack = []
    for i in range(n):
        leave_out = data[:i] + data[i + 1:]
        jack.append(stat_fn(leave_out))
    jack_mean = sum(jack) / n
    num = sum((jack_mean - j) ** 3 for j in jack)
    den = sum((jack_mean - j) ** 2 for j in jack)
    a_hat = num / (6.0 * (den ** 1.5 + _EPS))
    z_alpha = _ppf(_norm_cdf, alpha / 2.0)
    z_1alpha = _ppf(_norm_cdf, 1.0 - alpha / 2.0)
    a1 = _norm_cdf(z0 + (z0 + z_alpha) / (1.0 - a_hat * (z0 + z_alpha) + _EPS))
    a2 = _norm_cdf(z0 + (z0 + z_1alpha) / (1.0 - a_hat * (z0 + z_1alpha) + _EPS))
    lo_idx = max(0, min(len(boots) - 1, int(a1 * len(boots))))
    hi_idx = max(0, min(len(boots) - 1, int(a2 * len(boots))))
    return {"low": float(boots[lo_idx]), "high": float(boots[hi_idx]), "point": float(theta_hat)}


#  3. Welfare Functions 


def utilitarian_welfare(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def rawlsian_welfare(values: List[float]) -> float:
    return float(min(values)) if values else 0.0


def nash_welfare(values: List[float]) -> float:
    pos = [v for v in values if v > 0]
    if not pos:
        return 0.0
    log_sum = sum(math.log(v) for v in pos)
    return float(math.exp(log_sum / len(pos)))


def sen_welfare(values: List[float]) -> float:
    """Sen's welfare = mean * (1 - Gini)."""
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    g = gini_coefficient(values)
    return float(mu * (1.0 - g))


def prioritarian_welfare(values: List[float], exponent: float = 0.5) -> float:
    """Prioritarian welfare: sum of v^exponent / n (concave transformation)."""
    if not values:
        return 0.0
    pos = [max(v, _EPS) for v in values]
    return float(sum(v ** exponent for v in pos) / len(pos))


#  4. Risk Metrics 


def cvar(values: List[float], alpha: float = 0.05) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(1, int(len(s) * alpha))
    return float(sum(s[:k]) / k)


def value_at_risk(values: List[float], alpha: float = 0.05) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * alpha)))
    return float(s[idx])


def sharpe_ratio(values: List[float], risk_free: float = 0.0) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    std = math.sqrt(var) + _EPS
    return float((mu - risk_free) / std)


def sortino_ratio(values: List[float], risk_free: float = 0.0) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    downside = [v for v in values if v < risk_free]
    if not downside:
        return float(mu - risk_free) / _EPS
    ds_var = sum((v - risk_free) ** 2 for v in downside) / len(downside)
    ds_std = math.sqrt(ds_var) + _EPS
    return float((mu - risk_free) / ds_std)


def max_drawdown(cumulative: List[float]) -> float:
    if len(cumulative) < 2:
        return 0.0
    peak = cumulative[0]
    mdd = 0.0
    for v in cumulative:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > mdd:
            mdd = dd
    return float(mdd)


#  5. Divergence Measures 


def kl_divergence(p: List[float], q: List[float]) -> float:
    if len(p) != len(q) or not p:
        return 0.0
    total_p = sum(p) + _EPS
    total_q = sum(q) + _EPS
    kl = 0.0
    for pi, qi in zip(p, q):
        pi_n = pi / total_p + _EPS
        qi_n = qi / total_q + _EPS
        kl += pi_n * math.log(pi_n / qi_n)
    return float(max(0.0, kl))


def js_divergence(p: List[float], q: List[float]) -> float:
    if len(p) != len(q) or not p:
        return 0.0
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def wasserstein_1(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    pts = sorted(set(sa + sb))
    w = 0.0
    prev = pts[0] if pts else 0.0
    for x in pts[1:]:
        fa = _ecdf_val(sa, prev)
        fb = _ecdf_val(sb, prev)
        w += abs(fa - fb) * (x - prev)
        prev = x
    return float(w)


def total_variation(p: List[float], q: List[float]) -> float:
    if len(p) != len(q) or not p:
        return 0.0
    tp = sum(p) + _EPS
    tq = sum(q) + _EPS
    return float(0.5 * sum(abs(pi / tp - qi / tq) for pi, qi in zip(p, q)))


def hellinger_distance(p: List[float], q: List[float]) -> float:
    if len(p) != len(q) or not p:
        return 0.0
    tp = sum(p) + _EPS
    tq = sum(q) + _EPS
    s = sum((math.sqrt(pi / tp) - math.sqrt(qi / tq)) ** 2 for pi, qi in zip(p, q))
    return float(math.sqrt(s / 2.0))


def chi_square_divergence(p: List[float], q: List[float]) -> float:
    if len(p) != len(q) or not p:
        return 0.0
    tp = sum(p) + _EPS
    tq = sum(q) + _EPS
    return float(sum((pi / tp - qi / tq) ** 2 / (qi / tq + _EPS)
                      for pi, qi in zip(p, q)))


#  6. Convergence Diagnostics 


def autocorrelation(series: List[float], lag: int = 1) -> float:
    n = len(series)
    if n < lag + 2:
        return 0.0
    mu = sum(series) / n
    var = sum((x - mu) ** 2 for x in series) / n
    if var < _EPS:
        return 0.0
    cov = sum((series[i] - mu) * (series[i + lag] - mu) for i in range(n - lag))
    return float(cov / (n * var))


def ewma_residual_variance(series: List[float], span: int = 20) -> float:
    if len(series) < 3:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    ewma = series[0]
    residuals = []
    for v in series[1:]:
        ewma = alpha * v + (1.0 - alpha) * ewma
        residuals.append(v - ewma)
    if not residuals:
        return 0.0
    return float(sum(r * r for r in residuals) / len(residuals))


def effective_sample_size(series: List[float]) -> float:
    n = len(series)
    if n < 4:
        return float(n)
    rho1 = autocorrelation(series, 1)
    rho2 = autocorrelation(series, 2)
    tau = 1.0 + 2.0 * (abs(rho1) + abs(rho2))
    return float(max(1.0, n / tau))


def geweke_z(series: List[float], frac_a: float = 0.1, frac_b: float = 0.5) -> float:
    """
    Geweke convergence diagnostic: compare means of first frac_a and last frac_b
    of the chain. Large |z| suggests non-convergence.
    """
    n = len(series)
    if n < 10:
        return 0.0
    na = max(2, int(n * frac_a))
    nb = max(2, int(n * frac_b))
    a = series[:na]
    b = series[-nb:]
    mu_a = sum(a) / na
    mu_b = sum(b) / nb
    var_a = sum((x - mu_a) ** 2 for x in a) / max(na - 1, 1)
    var_b = sum((x - mu_b) ** 2 for x in b) / max(nb - 1, 1)
    se = math.sqrt(var_a / na + var_b / nb + _EPS)
    return float((mu_a - mu_b) / se)


def split_rhat(series: List[float]) -> float:
    """
    Split R-hat diagnostic: split chain into two halves, compare
    within-chain and between-chain variance. Values near 1.0 indicate convergence.
    """
    n = len(series)
    if n < 8:
        return 1.0
    mid = n // 2
    chain1 = series[:mid]
    chain2 = series[mid:]
    m1, m2 = sum(chain1) / len(chain1), sum(chain2) / len(chain2)
    grand = (m1 + m2) / 2.0
    B = len(chain1) * ((m1 - grand) ** 2 + (m2 - grand) ** 2)
    W1 = sum((x - m1) ** 2 for x in chain1) / max(len(chain1) - 1, 1)
    W2 = sum((x - m2) ** 2 for x in chain2) / max(len(chain2) - 1, 1)
    W = (W1 + W2) / 2.0
    n_half = len(chain1)
    var_hat = (1.0 - 1.0 / n_half) * W + B / n_half
    return float(math.sqrt(max(1.0, var_hat / (W + _EPS))))


def ewma(series: List[float], span: int = 10) -> List[float]:
    """Exponentially weighted moving average."""
    if not series:
        return []
    alpha = 2.0 / (span + 1.0)
    result = [series[0]]
    for v in series[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


def linear_slope(series: List[float]) -> float:
    """Least-squares linear slope of the series."""
    n = len(series)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(series) / n
    num = sum((i - x_mean) * (series[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return float(num / (den + _EPS))


def convergence_rate(series: List[float], window: int = 20) -> float:
    """
    Ratio of variance in the last `window` steps to total variance.
    Lower values indicate better convergence.
    """
    if len(series) < window + 2:
        return 1.0
    total_mu = sum(series) / len(series)
    total_var = sum((v - total_mu) ** 2 for v in series) / len(series)
    tail = series[-window:]
    tail_mu = sum(tail) / len(tail)
    tail_var = sum((v - tail_mu) ** 2 for v in tail) / len(tail)
    return float(tail_var / (total_var + _EPS))


#  7. Fairness Metrics 


def equalized_opportunity_difference(
    outcomes_a: List[float], outcomes_b: List[float],
    threshold: float = 0.5,
) -> float:
    """
    Equalized Opportunity Difference: difference in true positive rates
    between two groups. Uses threshold to binarize outcomes.
    Returns value in [-1, 1]. 0 = perfect equalized opportunity.
    """
    if not outcomes_a or not outcomes_b:
        return 0.0
    tpr_a = sum(1.0 for v in outcomes_a if v >= threshold) / max(len(outcomes_a), 1)
    tpr_b = sum(1.0 for v in outcomes_b if v >= threshold) / max(len(outcomes_b), 1)
    return float(tpr_a - tpr_b)


def demographic_parity_gap(
    outcomes_a: List[float], outcomes_b: List[float],
) -> float:
    """
    Demographic Parity Gap: difference in mean outcomes between groups.
    """
    if not outcomes_a or not outcomes_b:
        return 0.0
    mu_a = sum(outcomes_a) / len(outcomes_a)
    mu_b = sum(outcomes_b) / len(outcomes_b)
    return float(mu_a - mu_b)


def between_within_group_decomposition(
    groups: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Decompose total variance into between-group and within-group components.
    Returns dict with 'total', 'between', 'within', 'between_share'.
    """
    all_vals = []
    for v_list in groups.values():
        all_vals.extend(v_list)
    if len(all_vals) < 2:
        return {"total": 0.0, "between": 0.0, "within": 0.0, "between_share": 0.0}
    grand_mean = sum(all_vals) / len(all_vals)
    total_var = sum((v - grand_mean) ** 2 for v in all_vals) / len(all_vals)
    between_var = 0.0
    within_var = 0.0
    for name, v_list in groups.items():
        if not v_list:
            continue
        g_mean = sum(v_list) / len(v_list)
        n_g = len(v_list)
        between_var += n_g * (g_mean - grand_mean) ** 2
        within_var += sum((v - g_mean) ** 2 for v in v_list)
    n_total = len(all_vals)
    between_var /= n_total
    within_var /= n_total
    between_share = between_var / (total_var + _EPS) if total_var > _EPS else 0.0
    return {
        "total": float(total_var),
        "between": float(between_var),
        "within": float(within_var),
        "between_share": float(min(1.0, between_share)),
    }


def intersectional_disparity_ratio(
    group_outcomes: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Compute disparity ratios for all groups relative to the best-off group.
    Returns dict mapping group_name -> ratio (0 to 1, where 1 = parity).
    """
    if not group_outcomes:
        return {}
    means = {}
    for name, vals in group_outcomes.items():
        if vals:
            means[name] = sum(vals) / len(vals)
    if not means:
        return {}
    best = max(means.values())
    if best < _EPS:
        return {name: 1.0 for name in means}
    return {name: float(mu / best) for name, mu in means.items()}


#  Helper Functions 


def _assign_ranks(combined: list) -> List[float]:
    n = len(combined)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    return ranks


def _ecdf_val(sorted_data: list, x: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_data[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / n


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ppf(cdf_fn, p: float) -> float:
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cdf_fn(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


#  Aggregate Summary 


def compute_full_summary(
    outcomes: List[float],
    costs: List[float],
    group_a_outcomes: Optional[List[float]] = None,
    group_b_outcomes: Optional[List[float]] = None,
    reward_series: Optional[List[float]] = None,
    group_outcomes: Optional[Dict[str, List[float]]] = None,
    rank_values: Optional[List[float]] = None,
) -> Dict:
    """Light-weight entry point that bundles commonly needed metrics."""
    summary: Dict = {}
    summary["gini"] = gini_coefficient(outcomes)
    summary["theil_l"] = theil_l_index(outcomes)
    summary["theil_t"] = theil_t_index(outcomes)
    summary["atkinson"] = atkinson_index(outcomes)
    summary["hoover"] = hoover_index(outcomes)
    summary["palma"] = palma_ratio(outcomes)
    summary["ge_alpha2"] = generalized_entropy_index(outcomes, alpha=2.0)
    summary["utilitarian"] = utilitarian_welfare(outcomes)
    summary["rawlsian"] = rawlsian_welfare(outcomes)
    summary["nash"] = nash_welfare(outcomes)
    summary["sen"] = sen_welfare(outcomes)
    summary["prioritarian"] = prioritarian_welfare(outcomes)
    summary["cvar_5"] = cvar(outcomes, 0.05)
    summary["var_5"] = value_at_risk(outcomes, 0.05)
    summary["sharpe"] = sharpe_ratio(outcomes)
    summary["sortino"] = sortino_ratio(outcomes)
    if costs:
        summary["cost_gini"] = gini_coefficient(costs)
    if rank_values and len(rank_values) == len(outcomes):
        summary["concentration_index"] = concentration_index(outcomes, rank_values)
    if group_a_outcomes and group_b_outcomes:
        summary["mann_whitney"] = mann_whitney_u(group_a_outcomes, group_b_outcomes)
        summary["ks_test"] = ks_2sample(group_a_outcomes, group_b_outcomes)
        summary["eq_opp_diff"] = equalized_opportunity_difference(
            group_a_outcomes, group_b_outcomes
        )
        summary["demographic_parity_gap"] = demographic_parity_gap(
            group_a_outcomes, group_b_outcomes
        )
    if group_outcomes:
        summary["between_within"] = between_within_group_decomposition(group_outcomes)
        summary["intersectional_disparity"] = intersectional_disparity_ratio(group_outcomes)
    if reward_series and len(reward_series) > 4:
        summary["autocorr_1"] = autocorrelation(reward_series, 1)
        summary["ewma_var"] = ewma_residual_variance(reward_series)
        summary["ess"] = effective_sample_size(reward_series)
        summary["geweke_z"] = geweke_z(reward_series)
        summary["split_rhat"] = split_rhat(reward_series)
        summary["max_drawdown"] = max_drawdown(reward_series)
    summary["lorenz"] = lorenz_curve(outcomes)
    return summary
