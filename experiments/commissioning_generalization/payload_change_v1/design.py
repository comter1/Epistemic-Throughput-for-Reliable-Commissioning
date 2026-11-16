"""Frozen public designs and evidence-dependent payload test selection.

All hypothetical updates are 17-dimensional linear Gaussian calculations.
There are no simulator rollouts, true-load queries or future sensor accesses.
"""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np

from .estimator import (SCALE, payload_to_pi, project_physical, default_parameters,
                        default_regressor, whiten_features, pi_to_payload)


def _sqrt_psd(matrix):
    eig, vec = np.linalg.eigh((matrix + matrix.T) / 2)
    return vec * np.sqrt(np.maximum(eig, 0))


def _project_sample_matrix(samples, geometry=None):
    """Vectorized version of the same moment feasibility projection as the fit."""
    a = np.asarray(samples, dtype=float)
    geometry = geometry or {}
    center = np.asarray(geometry.get("box_center", [0., 0., .07]))
    half = np.asarray(geometry.get("box_size", [.16, .16, .16])) / 2
    m = np.clip(a[:, 0], .2, 1.5)
    c = np.clip(a[:, 1:4] / np.maximum(a[:, 0:1], .02), center - half * .999, center + half * .999)
    norm = np.linalg.norm(c, axis=1)
    c *= np.minimum(1., .12 / np.maximum(norm, 1e-20))[:, None]
    c = np.clip(c, center - half * .999, center + half * .999)
    io = np.zeros((len(a), 3, 3))
    io[:, 0, 0], io[:, 1, 1], io[:, 2, 2] = a[:, 4], a[:, 5], a[:, 6]
    io[:, 0, 1] = io[:, 1, 0] = a[:, 7]
    io[:, 0, 2] = io[:, 2, 0] = a[:, 8]
    io[:, 1, 2] = io[:, 2, 1] = a[:, 9]
    second = .5 * np.trace(io, axis1=1, axis2=2)[:, None, None] * np.eye(3) - io
    sc = second - np.einsum("ni,nj->nij", a[:, 1:4], a[:, 1:4]) / np.maximum(a[:, 0:1, None], .02)
    eig, vec = np.linalg.eigh(sc)
    sc = (vec * np.maximum(eig, m[:, None] * 1e-8)[:, None, :]) @ vec.transpose(0, 2, 1)
    upper = m[:, None] * np.maximum(half ** 2 - (c - center) ** 2, 1e-8)
    shrink = np.minimum(1., np.min(upper / np.maximum(np.diagonal(sc, axis1=1, axis2=2), 1e-20), axis=1))
    sc *= shrink[:, None, None]
    ic = np.trace(sc, axis1=1, axis2=2)[:, None, None] * np.eye(3) - sc
    io = ic + m[:, None, None] * (np.sum(c*c, axis=1)[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", c, c))
    return np.column_stack([m, m[:, None] * c, io[:, 0, 0], io[:, 1, 1], io[:, 2, 2], io[:, 0, 1], io[:, 0, 2], io[:, 1, 2]])


def build_design_bank(actions, work_actions, parameters=None, regressor=None):
    """Compile expected information from reference grids, not physical rollouts.

    Work actions must be frozen public representatives of all three declared
    workload categories. Predictor rows retain both original and half speed.
    """
    from .trajectories import reference_grid
    parameters = parameters or default_parameters()
    regressor = regressor or default_regressor()
    before_calls = regressor.inverse_calls
    designs = {}
    for action in actions:
        grid = reference_grid(action, hz=20)
        if isinstance(grid, dict):
            ts, q, dq, ddq = (grid[k] for k in ("time", "q", "dq", "ddq"))
        else:
            ts, q, dq, ddq = grid
        known, y = regressor.batch(q, dq, ddq)
        features = {"Y": y, "y": np.zeros_like(known), "time": np.asarray(ts)}
        h, _ = whiten_features(features, parameters)
        designs[action["id"]] = {"information17": (h.T @ h).tolist(),
                                  "duration": float(action["duration"]), "kind": action["kind"],
                                  "static_singular_values": np.linalg.svd(y.reshape(-1, 10)[:, :4], compute_uv=False).tolist(),
                                  "reference_rows": len(ts)}
    full_y, half_y, full_known, half_known = [], [], [], []
    for action in work_actions:
        grid = reference_grid(action, hz=4)
        if isinstance(grid, dict):
            ts, q, dq, ddq = (grid[k] for k in ("time", "q", "dq", "ddq"))
        else:
            ts, q, dq, ddq = grid
        kf, yf = regressor.batch(q, dq, ddq)
        kh, yh = regressor.batch(q, np.asarray(dq) * .5, np.asarray(ddq) * .25)
        full_y.append(yf); half_y.append(yh)
        full_known.append(kf); half_known.append(kh)
    limits = np.asarray(regressor.known.limits)
    bank = {"actions": designs, "Y_full": np.concatenate(full_y).tolist(),
            "Y_half": np.concatenate(half_y).tolist(),
            "known_full": np.concatenate(full_known).tolist(),
            "known_half": np.concatenate(half_known).tolist(), "torque_limits": limits.tolist(),
            "parameter_samples": 128, "prospective_outcomes": 16, "sampling_seed": 194633,
            "impact_threshold": .05, "near_limit_ratio": .95,
            "entropy_known_threshold_bits": .01,
            "uncertainty_interval": "maximum full torque width over approximate joint 10D 99% parameter ellipsoid",
            "joint_interval_radius": float(parameters["task_joint_interval_radius"]),
            "reference_note": "Public reference q/dq/ddq only; no dynamics integration.",
            "inverse_dynamics_calls": regressor.inverse_calls - before_calls}
    bank["sha256"] = hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return bank


def fixed_schedules(bank, parameters=None):
    """Greedy source-prior D-optimal static design, then dynamic diversification."""
    p = parameters or default_parameters()
    prior = np.zeros((17, 17))
    prior[:10, :10] = p["prior_covariance"]
    prior[10:, 10:] = np.diag(np.asarray(p["bias_sd"]) ** 2)
    info = np.linalg.inv(prior / np.outer(SCALE, SCALE))
    static = [k for k, a in bank["actions"].items() if a["kind"] == "static"]
    dynamic = [k for k, a in bank["actions"].items() if a["kind"] == "dynamic"]
    chosen = []
    evidence = []
    for pool, count, interest in ((static, 4, slice(0, 4)), (dynamic, 2, slice(4, 10))):
        for _ in range(min(count, len(pool))):
            scores = {}
            for action in pool:
                f = np.asarray(bank["actions"][action]["information17"]) * np.outer(SCALE, SCALE)
                cov = np.linalg.inv(info + f)
                # Marginal covariance includes the seven unknown persistent biases.
                sign, logdet = np.linalg.slogdet(cov[interest, interest])
                scores[action] = -float(logdet) if sign > 0 else -float("inf")
            best = max(pool, key=lambda k: (scores[k], k))
            chosen.append(best)
            evidence.append({"selected": best, "candidate_scores": scores})
            info += np.asarray(bank["actions"][best]["information17"]) * np.outer(SCALE, SCALE)
            pool.remove(best)
    return {"fixed_four_pose": chosen[:4], "fixed_mixed_suite": chosen,
            "selection_rule": "source-prior marginal log determinant, mass/first moments then inertia",
            "selection_evidence": evidence, "design_bank_hash": bank.get("sha256")}


def _arrays(bank):
    limit = np.asarray(bank["torque_limits"])
    yf = np.asarray(bank["Y_full"]) / limit[None, :, None]
    yh = np.asarray(bank["Y_half"]) / limit[None, :, None]
    kf = np.asarray(bank["known_full"]) / limit[None, :]
    kh = np.asarray(bank["known_half"]) / limit[None, :]
    return yf.reshape(-1, 10), yh.reshape(-1, 10), kf.ravel(), kh.ravel()


def _states(samples, configured_pi, bank, arrays=None):
    yf, yh, kf, kh = arrays or _arrays(bank)
    impact = np.max(np.abs((samples - configured_pi) @ yf.T), axis=1) > bank.get("impact_threshold", .05)
    full = np.max(np.abs(samples @ yf.T + kf), axis=1) >= bank.get("near_limit_ratio", .95)
    half = np.max(np.abs(samples @ yh.T + kh), axis=1) < bank.get("near_limit_ratio", .95)
    return impact.astype(int) + 2 * full.astype(int) + 4 * (full & half).astype(int)


def _entropy(states):
    counts = np.bincount(states, minlength=8)
    p = counts[counts > 0] / len(states)
    return float(-np.sum(p * np.log2(p)))


def assess(estimator, bank, configured_payload, threshold):
    start = time.process_time()
    mean, cov, _ = estimator.posterior()
    yf, yh, kf, kh = _arrays(bank)
    sd_full = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", yf, cov[:10, :10], yf), 0))
    sd_half = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", yh, cov[:10, :10], yh), 0))
    z = float(estimator.parameters["task_joint_interval_radius"])
    ratio = float(np.max(2 * z * sd_full))
    configured_pi = payload_to_pi(configured_payload)
    impact = float(np.max(np.abs(yf @ (mean[:10] - configured_pi))))
    full_demand = float(np.max(np.abs(yf @ mean[:10] + kf) + z * sd_full))
    half_demand = float(np.max(np.abs(yh @ mean[:10] + kh) + z * sd_half))
    estimate = estimator.estimate()
    sufficient = bool(estimate["residual_consistent"] and ratio <= threshold and estimate["processed_measurement_vectors"] > 0)
    speed = .5 if full_demand >= bank.get("near_limit_ratio", .95) and half_demand < bank.get("near_limit_ratio", .95) else 1.
    update = bool(impact > bank.get("impact_threshold", .05))
    reasons = []
    if ratio > threshold:
        reasons.append("task_torque_interval_above_common_threshold")
    if not estimate["residual_consistent"]:
        reasons.append("new_measurement_residual_inconsistent")
    if estimate["regressor_rank"] < 10:
        reasons.append("some_inertial_directions_unobserved_but_task_effect_checked")
    return {"evidence_sufficient": sufficient, "stop": sufficient,
            "torque_interval_ratio": ratio, "common_threshold": float(threshold),
            "torque_interval_definition": "full width over joint 10D 99% approximate parameter ellipsoid",
            "joint_interval_radius": z,
            "old_config_impact_ratio": impact,
            "predicted_full_torque_peak_ratio": full_demand,
            "predicted_half_torque_peak_ratio": half_demand,
            "setting_recommendation": "update_config" if update else "keep_config",
            "update_config": update, "final_speed": speed,
            "candidate_payload_config": estimate["payload_estimate"] if update else configured_payload,
            "unresolved_reasons": reasons, "assessment_cpu_seconds": time.process_time() - start}


def choose_action(estimator, candidates, bank, configured_payload, seed=None):
    """Expected entropy reduction of three work decisions per physical second.

    For a linear Gaussian observation, posterior covariance is outcome-independent
    and posterior means have covariance C-Cpost. Sampling these means exactly
    integrates the noisy joint payload/bias predictive distribution without
    constructing thousands of sensor rows or independently resetting the bias.
    """
    start = time.process_time()
    candidates = [c["id"] if isinstance(c, dict) else c for c in candidates]
    if not candidates:
        return None, {"reason": "no_feasible_action", "selection_cpu_seconds": 0.}
    mean, covariance, _ = estimator.posterior()
    scaled_cov = covariance / np.outer(SCALE, SCALE)
    prior_info = np.linalg.inv(scaled_cov + np.eye(17) * 1e-12)
    rng = np.random.default_rng(bank.get("sampling_seed", 194633) if seed is None else seed)
    nsample = int(bank.get("parameter_samples", 128))
    noutcome = int(bank.get("prospective_outcomes", 16))
    sample_z = rng.standard_normal((nsample, 10))
    outcome_z = rng.standard_normal((noutcome, 17))
    geometry = estimator.geometry
    samples = _project_sample_matrix(mean[:10] + sample_z @ _sqrt_psd(covariance[:10, :10]).T, geometry)
    configured_pi = payload_to_pi(configured_payload)
    arrays = _arrays(bank)
    prior_entropy = _entropy(_states(samples, configured_pi, bank, arrays))
    yf = arrays[0]
    prior_variance = float(np.mean(np.einsum("ni,ij,nj->n", yf, covariance[:10, :10], yf)))
    rows = []
    for action in candidates:
        design = bank["actions"][action]
        # Keep source-calibrated uncertainty inflation in prospective updates:
        # c*(C^-1+F)^-1, rather than spuriously treating c*C as uninflated prior.
        information = (np.asarray(design["information17"]) * np.outer(SCALE, SCALE)
                       / float(estimator.parameters["posterior_covariance_inflation"]))
        post_cov = np.linalg.inv(prior_info + information) * np.outer(SCALE, SCALE)
        mean_cov = (covariance - post_cov)
        prospective_means = mean + outcome_z @ _sqrt_psd(mean_cov).T
        offsets = sample_z @ _sqrt_psd(post_cov[:10, :10]).T
        future_samples = _project_sample_matrix((prospective_means[:, None, :10] + offsets[None, :, :]).reshape(-1, 10), geometry)
        future_states = _states(future_samples, configured_pi, bank, arrays).reshape(noutcome, nsample)
        after_entropy = float(np.mean([_entropy(z) for z in future_states]))
        entropy_gain = prior_entropy - after_entropy
        variance = float(np.mean(np.einsum("ni,ij,nj->n", yf, post_cov[:10, :10], yf)))
        duration = float(design["duration"])
        rows.append({"action_id": action, "expected_entropy_after_bits": after_entropy,
                     "expected_entropy_gain_bits": entropy_gain,
                     "entropy_gain_per_second": entropy_gain / duration,
                     "task_variance_reduction_per_second": (prior_variance - variance) / duration,
                     "duration_seconds": duration})
    fallback = prior_entropy <= bank.get("entropy_known_threshold_bits", .01)
    score = "task_variance_reduction_per_second" if fallback else "entropy_gain_per_second"
    best = max(rows, key=lambda row: (row[score], row["action_id"]))
    return best["action_id"], {"reason": "task_torque_variance_fallback" if fallback else "work_decision_entropy",
                              "prior_decision_entropy_bits": prior_entropy,
                              "prior_task_torque_variance": prior_variance,
                              "scores": rows, "selected_action": best["action_id"],
                              "parameter_samples": nsample, "prospective_outcomes": noutcome,
                              "persistent_bias_joint_update": True,
                              "physical_hypothesis_rollouts": 0,
                              "selection_cpu_seconds": time.process_time() - start}
