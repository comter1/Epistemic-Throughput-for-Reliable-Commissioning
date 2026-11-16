"""Payload Evidence Estimator: public measurements, separate known dynamics.

Inertial coordinates are [m,hx,hy,hz,Ixx,Iyy,Izz,Ixy,Ixz,Iyz], with
inertia about the hand origin and all vectors expressed in the hand frame.
Seven persistent torque biases are estimated jointly. This is an approximate
Gaussian engineering estimator, not a calibrated Bayesian safety certificate.
"""
from __future__ import annotations

from functools import lru_cache
import copy
import time
import numpy as np
from scipy.signal import savgol_filter

VERSION = "payload-evidence-estimator-v1"
PROCESSING_VERSION = "measured_interval_acceleration_sg11_v2"
PARAMETER_NAMES = ["mass", "hx", "hy", "hz", "Ixx_O", "Iyy_O", "Izz_O", "Ixy_O", "Ixz_O", "Iyz_O"]
SCALE = np.array([1., .08, .08, .08, .01, .01, .01, .005, .005, .005] + [.05] * 7)


def inertia_matrix(pi):
    p = np.asarray(pi)
    return np.array([[p[4], p[7], p[8]], [p[7], p[5], p[9]], [p[8], p[9], p[6]]])


def payload_to_pi(payload):
    m = float(payload["mass"])
    c = np.asarray(payload["com"], dtype=float)
    ic = np.asarray(payload["inertia_com"], dtype=float)
    io = ic + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    return np.array([m, *(m * c), io[0, 0], io[1, 1], io[2, 2], io[0, 1], io[0, 2], io[1, 2]])


def pi_to_payload(pi, geometry=None):
    p = np.asarray(pi, dtype=float)
    m = float(p[0])
    if m <= 0:
        raise ValueError("Nonpositive payload mass")
    c = p[1:4] / m
    ic = inertia_matrix(p) - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    geometry = geometry or {}
    return {"mass": m, "com": c.tolist(), "inertia_com": ic.tolist(),
            "box_size": list(geometry.get("box_size", [.16, .16, .16])),
            "box_center": list(geometry.get("box_center", [0., 0., .07]))}


def pseudo_inertia(pi):
    """Wensing et al. (2017): J = [[.5 tr(I)1-I, h], [h^T,m]]."""
    p = np.asarray(pi, dtype=float)
    io = inertia_matrix(p)
    j = np.zeros((4, 4))
    j[:3, :3] = .5 * np.trace(io) * np.eye(3) - io
    j[:3, 3] = j[3, :3] = p[1:4]
    j[3, 3] = p[0]
    return j


def project_physical(pi, geometry=None):
    """Moment projection with explicit displacement, never silent uncertainty loss.

    Positive mass, bounded COM and positive mass-distribution covariance imply
    PSD pseudo-inertia and the principal-moment triangle inequalities. A bound
    on second moments respects the known box dimensions. This is a sequential
    feasible projection, not the globally optimal SDP of the cited paper.
    """
    p = np.asarray(pi, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite inertial estimate")
    geometry = geometry or {}
    center = np.asarray(geometry.get("box_center", [0., 0., .07]), dtype=float)
    half = np.asarray(geometry.get("box_size", [.16, .16, .16]), dtype=float) / 2
    m = float(np.clip(p[0], .2, 1.5))
    c_raw = p[1:4] / max(p[0], .02)
    c = np.clip(c_raw, center - half * .999, center + half * .999)
    norm = np.linalg.norm(c)
    if norm > .12:
        c *= .12 / norm
        c = np.clip(c, center - half * .999, center + half * .999)
    # Central second moment S_C is the covariance of physical mass, not I_C.
    io = inertia_matrix(p)
    second = .5 * np.trace(io) * np.eye(3) - io
    sc = (second - np.outer(p[1:4], p[1:4]) / max(p[0], .02))
    eigen, vec = np.linalg.eigh((sc + sc.T) / 2)
    sc = (vec * np.maximum(eigen, m * 1e-8)) @ vec.T
    # E[(x-center)^2] <= half^2 is necessary for the known bounding box.
    upper = m * np.maximum(half ** 2 - (c - center) ** 2, 1e-8)
    shrink = min(1., float(np.min(upper / np.maximum(np.diag(sc), 1e-20))))
    sc *= shrink
    ic = np.trace(sc) * np.eye(3) - sc
    out = payload_to_pi({"mass": m, "com": c, "inertia_com": ic})
    change = out - p
    return out, {"method": "positive_mass_bounded_COM_central_second_moment_projection",
                 "parameter_displacement": change.tolist(),
                 "scaled_displacement_norm": float(np.linalg.norm(change / SCALE[:10])),
                 "projected": bool(np.max(np.abs(change)) > 1e-12),
                 "pseudo_inertia_min_eigenvalue": float(np.linalg.eigvalsh(pseudo_inertia(out))[0]),
                 "central_second_moment_shrink": shrink,
                 "uncertainty_rule": "retain Gaussian covariance and add displacement outer product"}


class PayloadRegressor:
    """Exact linear mapping from eleven physically valid immutable models.

    Finite differences here recover an exactly affine inertial parameter map;
    the basis is inside the physically feasible region, never a negative-mass
    or unit-negative-inertia MuJoCo model. No integration or plant query occurs.
    """
    def __init__(self, dynamics_factory=None, step_multiplier=1.):
        if dynamics_factory is None:
            from .physics import Dynamics
            dynamics_factory = Dynamics
        self.known = dynamics_factory(None)
        base = {"mass": .8, "com": [0., 0., .07],
                "inertia_com": (np.eye(3) * .002).tolist()}
        self.base_pi = payload_to_pi(base)
        self.steps = np.array([.08, .003, .003, .003, .0003, .0003, .0003,
                              .00015, .00015, .00015]) * step_multiplier
        self.base = dynamics_factory(pi_to_payload(self.base_pi))
        self.basis = []
        for j, delta in enumerate(self.steps):
            p = self.base_pi.copy()
            p[j] += delta
            if np.linalg.eigvalsh(pseudo_inertia(p))[0] <= 0:
                raise ValueError("Regressor basis is not physically feasible")
            self.basis.append(dynamics_factory(pi_to_payload(p)))
        self.inverse_calls = 0

    def at(self, q, dq, ddq):
        nominal = self.base.inverse(q, dq, ddq)
        y = np.column_stack([(model.inverse(q, dq, ddq) - nominal) / step
                             for model, step in zip(self.basis, self.steps)])
        # Use affine intercept rather than a separate no-payload ID call.
        known = nominal - y @ self.base_pi
        self.inverse_calls += 11
        return known, y

    def batch(self, q, dq, ddq):
        pairs = [self.at(a, b, c) for a, b, c in zip(q, dq, ddq)]
        if not pairs:
            return np.empty((0, 7)), np.empty((0, 7, 10))
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


@lru_cache(maxsize=1)
def default_regressor():
    return PayloadRegressor()


def default_parameters():
    base = {"mass": .85, "com": [0., 0., .075],
            "inertia_com": np.diag([.001, .001, .001]).tolist()}
    sd = np.array([.4, .03, .03, .035, .006, .006, .004, .003, .003, .003])
    return {"version": VERSION, "processing_version": PROCESSING_VERSION, "fit_split": "unfitted_engineering_defaults",
            "prior_mean": payload_to_pi(base).tolist(), "prior_covariance": np.diag(sd ** 2).tolist(),
            "bias_sd": [.02] * 7, "torque_residual_sd": [.08] * 7,
            "processed_residual_rho": [0.] * 7, "posterior_covariance_inflation": 1.,
            "residual_consistency_max": 4., "savgol_window": 11, "savgol_order": 3,
            "sample_stride": 5, "settled_velocity_max": .01,
            "settled_acceleration_max": .1, "settled_position_span_max": .001,
            "torque_interval_z": 2.5758293035489004,
            "task_joint_parameter_dimension": 10,
            "task_joint_coverage": .99,
            "task_joint_interval_radius": float(np.sqrt(23.209251158954356)),
            "task_interval_definition": "maximum torque range over approximate joint 10-dimensional 99% Gaussian ellipsoid; mass/COM marginal intervals use normal z separately",
            "uncertainty_kind": "approximate Gaussian with conservative physical-projection displacement",
            "source_reference": "https://arxiv.org/abs/1701.04395"}


_ALIASES = {"q": "joint_position_measured", "dq": "joint_velocity_measured",
            "torque": "joint_torque_measured", "time": "timestamp",
            "event_id": "measurement_event_id"}


def normalize_log(log):
    if "public" in log:
        log = log["public"]
    data = {}
    for short, long in _ALIASES.items():
        value = log.get(short, log.get(long))
        if value is None:
            raise ValueError("Missing measured field: " + short)
        data[short] = np.asarray(value)
    n = len(data["time"])
    if any(data[k].shape != (n, 7) for k in ("q", "dq", "torque")):
        raise ValueError("Sensor arrays must have shape [samples,7]")
    ex = log.get("execution_id", "execution")
    data["execution_id"] = np.repeat(ex, n) if np.ndim(ex) == 0 else np.asarray(ex)
    contacts = np.asarray(log.get("unexpected_contact_flags", np.zeros(n, dtype=bool)))
    data["contact"] = contacts.any(axis=1) if contacts.ndim > 1 else contacts
    if len(data["event_id"]) != n or len(data["execution_id"]) != n:
        raise ValueError("Measurement identity count mismatch")
    return data


def measurement_features(log, parameters=None, regressor=None):
    """Filter acquired measurements; actual acceleration is never an input."""
    p = parameters or default_parameters()
    regressor = regressor or default_regressor()
    data = normalize_log(log)
    n = len(data["time"])
    window = int(p["savgol_window"])
    if n <= window:
        return {"Y": np.empty((0, 7, 10)), "y": np.empty((0, 7)),
                "time": np.empty(0), "static": np.empty(0, bool), "event_id": np.empty(0)}
    dt = float(np.median(np.diff(data["time"])))
    if not .009 < dt < .011:
        raise ValueError(f"Expected contiguous 100Hz acquired measurements, dt={dt}")
    # The backend measures the torque held over the preceding control interval.
    # Match it to a measured velocity difference over THAT SAME interval. A
    # centered derivative at t[k] paired with torque[k] has a half-step phase
    # error, especially on the deliberately sharp original L0 transitions.
    # This interval construction uses acquired sensors only and the published
    # zero-order-hold convention, never requested torque or hidden qacc.
    q = (data["q"][1:] + data["q"][:-1]) * .5
    dq = (data["dq"][1:] + data["dq"][:-1]) * .5
    ddq = np.diff(data["dq"], axis=0) / np.diff(data["time"])[:, None]
    torque = data["torque"][1:]
    order = int(p["savgol_order"])
    q, dq, ddq, torque = [savgol_filter(v, window, order, axis=0) for v in (q,dq,ddq,torque)]
    n -= 1
    edge = window // 2
    selected = np.arange(edge, n - edge, int(p["sample_stride"]))
    selected = np.array([i for i in selected if not np.any(data["contact"][i-edge:i+edge+1])], dtype=int)
    settled = np.array([bool(np.max(np.abs(data["dq"][i-edge:i+edge+1])) < p["settled_velocity_max"]
                            and np.max(np.abs(ddq[i])) < p["settled_acceleration_max"]
                            and np.max(np.ptp(data["q"][i-edge:i+edge+1], axis=0)) < p["settled_position_span_max"])
                        for i in selected])
    q, dq, ddq = q[selected], dq[selected], ddq[selected]
    dq[settled] = 0.
    ddq[settled] = 0.
    known, ymat = regressor.batch(q, dq, ddq)
    return {"Y": ymat, "y": torque[selected] - known, "time": (data["time"][selected]+data["time"][selected+1])*.5,
            "static": settled, "event_id": data["event_id"][selected+1],
            "q": q, "dq": dq, "ddq_estimated": ddq,
            "torque_measured_filtered": torque[selected], "known_torque": known}


def whiten_features(features, parameters):
    ymat = features["Y"]
    y = features["y"]
    n = len(y)
    if not n:
        return np.empty((0, 17)), np.empty(0)
    h = np.concatenate([ymat, np.broadcast_to(np.eye(7), (n, 7, 7))], axis=2)
    rho = np.asarray(parameters["processed_residual_rho"])
    sigma = np.asarray(parameters["torque_residual_sd"])
    out_h, out_y = h.copy(), y.copy()
    if n > 1:
        gap = np.diff(features["time"])
        expected_dt = .01 * int(parameters["sample_stride"])
        adjacent = np.abs(gap - expected_dt) < expected_dt * .1
        indexes = np.flatnonzero(adjacent) + 1
        out_h[indexes] -= rho[None, :, None] * h[indexes - 1]
        out_y[indexes] -= rho[None, :] * y[indexes - 1]
        innovation_sd = sigma * np.sqrt(1 - rho ** 2)
        out_h[indexes] /= innovation_sd[None, :, None]
        out_y[indexes] /= innovation_sd[None, :]
        reset = np.r_[True, ~adjacent]
    else:
        reset = np.ones(n, bool)
    out_h[reset] /= sigma[None, :, None]
    out_y[reset] /= sigma[None, :]
    return out_h.reshape(-1, 17), out_y.reshape(-1)


class PayloadEvidenceEstimator:
    def __init__(self, parameters=None, geometry=None, regressor=None):
        self.parameters = copy.deepcopy(parameters or default_parameters())
        self.geometry = copy.deepcopy(geometry or {})
        self.regressor = regressor or default_regressor()
        self._records = {}
        self._features = {}
        self._dirty = set()
        self._last_execution = None
        self.fitting_cpu_seconds = 0.
        self.duplicate_events = 0
        self._inverse_calls = 0
        self.mean = np.r_[self.parameters["prior_mean"], np.zeros(7)]
        self.covariance = np.zeros((17, 17))
        self.covariance[:10, :10] = self.parameters["prior_covariance"]
        self.covariance[10:, 10:] = np.diag(np.asarray(self.parameters["bias_sd"]) ** 2)
        self._prior_mean = self.mean.copy()
        self._prior_cov = self.covariance.copy()
        self._result = None

    def ingest(self, log):
        start = time.process_time()
        data = normalize_log(log)
        acquired = 0
        for i, event in enumerate(data["event_id"]):
            key = str(event)
            if key in self._records:
                self.duplicate_events += 1
                # A duplicate identity with changed content is corruption.
                old = self._records[key]
                if any(not np.array_equal(old[k], data[k][i]) for k in ("q", "dq", "torque", "time")):
                    raise ValueError("Conflicting values for acquired event identity")
                continue
            row = {k: v[i].copy() if hasattr(v[i], "copy") else v[i] for k, v in data.items()}
            self._records[key] = row
            self._dirty.add(str(row["execution_id"]))
            self._last_execution = str(row["execution_id"])
            acquired += 1
        if acquired:
            self._refit()
        self.fitting_cpu_seconds += time.process_time() - start
        return self.estimate()

    def _refit(self):
        for execution in self._dirty:
            rows = sorted((v for v in self._records.values() if str(v["execution_id"]) == execution), key=lambda r: float(r["time"]))
            log = {k: np.asarray([r[k] for r in rows]) for k in ("q", "dq", "torque", "time", "event_id", "execution_id")}
            log["unexpected_contact_flags"] = np.asarray([r["contact"] for r in rows])
            before_calls = self.regressor.inverse_calls
            self._features[execution] = measurement_features(log, self.parameters, self.regressor)
            self._inverse_calls += self.regressor.inverse_calls - before_calls
        self._dirty.clear()
        prior_scaled_cov = self._prior_cov / np.outer(SCALE, SCALE)
        info = np.linalg.inv(prior_scaled_cov)
        rhs = info @ (self._prior_mean / SCALE)
        hs, ys = [], []
        for features in self._features.values():
            h, y = whiten_features(features, self.parameters)
            scaled = h * SCALE
            info += scaled.T @ scaled
            rhs += scaled.T @ y
            hs.append(h)
            ys.append(y)
        cov = np.linalg.inv(info)
        self.mean = (cov @ rhs) * SCALE
        self.covariance = cov * np.outer(SCALE, SCALE)
        h = np.concatenate(hs) if hs else np.empty((0, 17))
        y = np.concatenate(ys) if ys else np.empty(0)
        feasible_mean = self.mean.copy()
        feasible_mean[:10] = project_physical(self.mean[:10], self.geometry)[0]
        # Check the actual physically constrained estimate that can be deployed,
        # not an infeasible unconstrained fit that only fits the acquired data.
        residual = y - h @ feasible_mean
        self._residual_score = float(np.mean(residual ** 2)) if len(y) else float("inf")
        last = self._features.get(self._last_execution)
        lh, ly = whiten_features(last, self.parameters) if last is not None else (np.empty((0, 17)), np.empty(0))
        self._last_residual_score = float(np.mean((ly - lh @ feasible_mean) ** 2)) if len(ly) else float("inf")
        # Raw regressor singular values in declared physical coordinates.
        self._singular = np.linalg.svd(h[:, :10], compute_uv=False) if len(h) else np.zeros(10)
        self._sample_count = len(y) // 7
        self._result = None

    def posterior(self):
        pi, projection = project_physical(self.mean[:10], self.geometry)
        m = self.mean.copy()
        m[:10] = pi
        covariance = self.covariance * float(self.parameters["posterior_covariance_inflation"])
        displacement = m - self.mean
        covariance += np.outer(displacement, displacement)
        return m, covariance, projection

    def estimate(self):
        m, cov, projection = self.posterior()
        c = m[1:4] / m[0]
        jac = np.zeros((3, 10))
        jac[:, 0] = -c / m[0]
        jac[:, 1:4] = np.eye(3) / m[0]
        com_cov = jac @ cov[:10, :10] @ jac.T
        singular = getattr(self, "_singular", np.zeros(10))
        rank = int(np.sum(singular > max(float(singular[0]) * 1e-8, 1e-10)))
        residual = getattr(self, "_residual_score", float("inf"))
        last_residual = getattr(self, "_last_residual_score", float("inf"))
        result = {"payload_estimate": pi_to_payload(m[:10], self.geometry),
                  "parameter_mean10": m[:10].tolist(), "covariance10": cov[:10, :10].tolist(),
                  "joint_mean17": m.tolist(), "joint_covariance17": cov.tolist(),
                  "unconstrained_parameter_mean10": self.mean[:10].tolist(),
                  "mass_sd": float(np.sqrt(cov[0, 0])), "com_covariance": com_cov.tolist(),
                  "com_sd": np.sqrt(np.maximum(np.diag(com_cov), 0)).tolist(),
                  "persistent_bias_estimate": m[10:].tolist(), "projection": projection,
                  "regressor_singular_values": singular.tolist(), "regressor_rank": rank,
                  "residual_score": residual if np.isfinite(residual) else None,
                  "last_block_residual_score": last_residual if np.isfinite(last_residual) else None,
                  "residual_consistent": bool(max(residual, last_residual) <= self.parameters["residual_consistency_max"]),
                  "unique_measurement_events": len(self._records),
                  "processed_measurement_vectors": getattr(self, "_sample_count", 0),
                  "static_vectors": sum(int(f["static"].sum()) for f in self._features.values()),
                  "virtual_inverse_dynamics_calls": self._inverse_calls,
                  "fitting_cpu_seconds": self.fitting_cpu_seconds,
                  "uncertainty_kind": self.parameters["uncertainty_kind"]}
        return result


def calibration_record(logs, true_payload, parameters=None, regressor=None):
    """SOURCE ONLY. Labels are explicit; no noise realization or plant is used."""
    parameters = parameters or default_parameters()
    regressor = regressor or default_regressor()
    before_calls = regressor.inverse_calls
    pi = payload_to_pi(true_payload)
    blocks = [measurement_features(log, parameters, regressor) for log in logs]
    residuals = [f["y"] - np.einsum("njp,p->nj", f["Y"], pi) for f in blocks if len(f["y"])]
    if not residuals:
        return {"pi": pi.tolist(), "count": 0, "processing_version": PROCESSING_VERSION,
                "virtual_inverse_dynamics_calls": regressor.inverse_calls - before_calls}
    residual = np.concatenate(residuals)
    center = np.mean(residual, axis=0)
    centered = residual - center
    lag_xy = np.zeros(7)
    lag_xx = np.zeros(7)
    for r in residuals:
        r = r - center
        lag_xy += np.sum(r[1:] * r[:-1], axis=0)
        lag_xx += np.sum(r[:-1] ** 2, axis=0)
    static_residuals = [r[f["static"]] for f, r in zip([b for b in blocks if len(b["y"])], residuals)]
    static_rows = np.concatenate(static_residuals)
    return {"pi": pi.tolist(), "count": len(residual), "residual_mean": center.tolist(),
            "processing_version": PROCESSING_VERSION,
            "virtual_inverse_dynamics_calls": regressor.inverse_calls - before_calls,
            "static_residual_mean": np.mean(static_rows, axis=0).tolist() if len(static_rows) >= 10 else None,
            "centered_sum_squares": np.sum(centered ** 2, axis=0).tolist(),
            "lag_xy": lag_xy.tolist(), "lag_xx": lag_xx.tolist(),
            "absolute_residual_q99": np.quantile(np.abs(centered), .99, axis=0).tolist(),
            "static_vectors": sum(int(f["static"].sum()) for f in blocks)}


def fit_parameters(records):
    """Fit from SOURCE calibration summaries, before validation or test access."""
    records = list(records)
    if any(r.get("processing_version") != PROCESSING_VERSION for r in records):
        raise ValueError("Source calibration uses stale sensor processing; regenerate summaries from saved raw public records")
    pars = default_parameters()
    pi = np.asarray([r["pi"] for r in records])
    mean = pi.mean(axis=0)
    covariance = np.cov(pi, rowvar=False) * 1.5
    # No artificial identification of unobserved inertia via prior correlations.
    covariance[:4, 4:] = covariance[4:, :4] = 0.
    floor = np.array([.03, .002, .002, .002, .00015, .00015, .00015, .0001, .0001, .0001])
    covariance += np.diag(floor ** 2)
    good = [r for r in records if r["count"]]
    if not good:
        raise ValueError("No valid source measurements for calibration")
    count = sum(r["count"] for r in good)
    sigma = np.sqrt(np.sum([r["centered_sum_squares"] for r in good], axis=0) / count)
    sigma = np.maximum(sigma, .03)
    rho = np.sum([r["lag_xy"] for r in good], axis=0) / np.maximum(np.sum([r["lag_xx"] for r in good], axis=0), 1e-20)
    rho = np.clip(rho, 0., .95)
    bias_means = np.asarray([r.get("static_residual_mean") or r["residual_mean"] for r in good])
    bias_sd = np.maximum(np.sqrt(np.mean(bias_means ** 2, axis=0)), .02)
    # Conservative tail coverage from measured residuals, independent of test.
    tail = np.quantile(np.asarray([r["absolute_residual_q99"] for r in good]), .95, axis=0)
    tail_inflation = max(1., float(np.max(tail / (2.5758293035489004 * sigma))))
    pars.update({"fit_split": "source", "source_case_count": len(records),
                 "source_processed_vectors": count, "prior_mean": mean.tolist(),
                 "prior_covariance": covariance.tolist(), "torque_residual_sd": (sigma * tail_inflation).tolist(),
                 "processed_residual_rho": rho.tolist(), "bias_sd": bias_sd.tolist(),
                 "source_residual_tail_inflation": tail_inflation,
                 "source_bias_means_rms": np.sqrt(np.mean(bias_means ** 2, axis=0)).tolist(),
                 "calibration_note": "Source true inertial labels minus public measured-signal model; no injected bias/noise draws."})
    return pars


def recalibrate_source_case(run_root, case_id):
    """Recompute small calibration summaries from retained source raw records."""
    case, logs, _ = _load_source_logs(run_root, case_id)
    return calibration_record(logs, case["true_payload"])


def source_coverage_record(logs, true_payload, parameters):
    """Source-only interval calibration, independent of private noise realizations."""
    estimator = PayloadEvidenceEstimator(parameters, geometry=true_payload)
    true_pi = payload_to_pi(true_payload)
    result = []
    previous_calls = 0
    for log in logs:
        estimate = estimator.ingest(log)
        mean, covariance, _ = estimator.posterior()
        delta = true_pi - mean[:10]
        scaled_cov = covariance[:10, :10] / np.outer(SCALE[:10], SCALE[:10])
        distance = float((delta / SCALE[:10]) @ np.linalg.solve(scaled_cov + np.eye(10)*1e-12, delta / SCALE[:10]))
        mass_z = abs(delta[0]) / max(np.sqrt(covariance[0, 0]), 1e-12)
        com_delta = np.asarray(true_payload["com"]) - np.asarray(estimate["payload_estimate"]["com"])
        com_sd = np.asarray(estimate["com_sd"])
        result.append({"mass_abs_z": float(mass_z), "com_max_abs_z": float(np.max(np.abs(com_delta)/np.maximum(com_sd, 1e-12))),
                       "parameter_mahalanobis_squared": distance,
                       "residual_score": estimate["residual_score"],
                       "last_block_residual_score": estimate["last_block_residual_score"],
                       "mass_error_kg": abs(float(delta[0])), "com_error_m": float(np.linalg.norm(com_delta)),
                       "virtual_inverse_dynamics_calls": estimate["virtual_inverse_dynamics_calls"]-previous_calls})
        previous_calls = estimate["virtual_inverse_dynamics_calls"]
    return result


def _load_source_logs(run_root, case_id):
    import json
    from pathlib import Path
    root = Path(run_root)
    cases = json.loads((root / "private/source.json").read_text())
    case = next(row for row in cases if row["case_id"] == case_id)
    paths = sorted((root / "raw/source" / case_id).glob("block_*/public.npz"))
    if not paths:
        raise FileNotFoundError("No acquired source records: " + case_id)
    logs, metadata = [], []
    for path in paths:
        with np.load(path) as arrays:
            log = {k: arrays[k] for k in ("q", "dq", "torque", "time", "event_id", "unexpected_contact_flags")}
        meta = json.loads(path.with_name("metadata.json").read_text())
        log["execution_id"] = meta["execution_id"]
        logs.append(log); metadata.append(meta)
    return case, logs, metadata


def source_coverage_case(run_root, case_id, parameters):
    case, logs, metadata = _load_source_logs(run_root, case_id)
    rows = source_coverage_record(logs, case["true_payload"], parameters)
    for row, meta in zip(rows, metadata):
        row["action_id"] = meta["action"]["id"]
    return rows


def fit_uncertainty_inflation(coverage_records, parameters):
    """Conservative predeclared source 99% error-quantile calibration.

    All retained prefixes receive equal weight; these are empirical source
    calibration figures, not out-of-sample coverage guarantees. No threshold is
    selected against test or validation labels here.
    """
    from scipy.stats import chi2
    rows = []
    for record in coverage_records:
        rows.extend(record if isinstance(record, list) else [record])
    if not rows:
        raise ValueError("No source coverage records")
    mass = np.asarray([r["mass_abs_z"] for r in rows])
    com = np.asarray([r["com_max_abs_z"] for r in rows])
    maha = np.asarray([r["parameter_mahalanobis_squared"] for r in rows])
    z = float(parameters["torque_interval_z"])
    cut = float(chi2.ppf(.99, 10))
    ratios = {"mass": float((np.quantile(mass, .99)/z)**2),
              "com_max_component": float((np.quantile(com, .99)/z)**2),
              "ten_parameter_ellipsoid": float(np.quantile(maha, .99)/cut)}
    inflation = max(1., *ratios.values())
    result = copy.deepcopy(parameters)
    result["posterior_covariance_inflation"] *= inflation
    residuals = [max(r["residual_score"] or 0.,r["last_block_residual_score"] or 0.) for r in rows]
    result["residual_consistency_max"] = max(4., float(np.quantile(residuals,.99)*1.2))
    summary = {"split": "source", "prefix_count":len(rows), "quantile":.99,
               "multiplicative_covariance_inflation":inflation,
               "total_covariance_inflation":result["posterior_covariance_inflation"],
               "candidate_inflations":ratios,"chi_squared_10_99_percent":cut,
               "residual_consistency_max":result["residual_consistency_max"],
               "precalibration_mass_coverage":float(np.mean(mass<=z)),
               "precalibration_com_all_component_coverage":float(np.mean(com<=z)),
               "precalibration_parameter_ellipsoid_coverage":float(np.mean(maha<=cut)),
               "claim":"empirical source calibration; validation/test coverage remains measured independently"}
    result["source_uncertainty_calibration"] = summary
    return result, summary
