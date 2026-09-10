# Paper-Guided Spoofing Detector Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paper-guided physical features, a false-alarm-constrained operating point, persistent online alarms, broader route holdouts, and evidence-backed report updates without changing the legacy M0-M4 feature contracts.

**Architecture:** `src/features.py` emits legacy `f_`/`c_` columns plus isolated `e_` columns. Legacy models explicitly select `f_`/`c_`; M2e alone consumes all three families. A new pure `TemporalAlarmPolicy` is shared by offline evaluation and streaming so CLI/API behavior and reported confirmed metrics use identical state semantics.

**Tech Stack:** Python 3.12, NumPy, pandas, scikit-learn, FastAPI, pytest, joblib, PyArrow.

**Spec:** `docs/superpowers/specs/2026-09-10-paper-guided-spoofing-detector-optimization-design.md`

## Global Constraints

- Preserve flight-disjoint train/val/test and never use test data for fitting, preprocessing, feature selection, threshold selection, or model selection.
- Preserve the legacy M0-M4 feature semantics; only M2e consumes `e_` enhanced features.
- Use no fabricated HDOP, satellite-count, fix-quality, or real-flight fields.
- Do not copy paper-specific physical thresholds into this dataset.
- Add no XGBoost dependency in this iteration.
- Keep negative or degraded results and update the report only from newly generated artifacts.
- Do not batch-delete files or directories.

---

### Task 1: Enhanced physical feature extraction

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: normalized 20 Hz windows in `NORMALIZED_COLUMNS` order.
- Produces: `_enhanced_feats(cols: dict[str, np.ndarray]) -> dict[str, float]` and `e_` columns appended by `_window_feats`.

- [ ] **Step 1: Write failing tests for relative position and track/yaw behavior**

Add literal synthetic cases to `tests/test_features.py`:

```python
def test_enhanced_position_features_use_relative_motion():
    df = _df()
    out = window_extract(df).iloc[0]
    assert out["e_gps_step_mean"] > 0.0
    assert out["e_gps_step_max"] >= out["e_gps_step_mean"]
    assert out["e_gps_speed_resid_mean"] > 0.0


def test_track_yaw_wrap_does_not_create_full_circle_error():
    df = _df()
    df["lat"] = 34.0
    df["lon"] = 108.9 - np.arange(len(df)) * 1e-7
    df["yaw"] = -np.pi + 0.01
    out = window_extract(df).iloc[0]
    assert out["e_track_yaw_mean_abs"] < 0.1
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_features.py::test_enhanced_position_features_use_relative_motion tests/test_features.py::test_track_yaw_wrap_does_not_create_full_circle_error -q
```

Expected: FAIL because the `e_` columns do not exist.

- [ ] **Step 3: Implement relative ENU and circular helpers**

Add focused helpers in `src/features.py`:

```python
def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) < EPS or np.nanstd(b[ok]) < EPS:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _local_enu(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = lat[np.isfinite(lat)][0]
    lon0 = lon[np.isfinite(lon)][0]
    north = (lat - lat0) * 111320.0
    east = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    return east, north
```

Use wrapped angle differences, unwrap only finite direction sequences, and exclude samples whose reported horizontal speed is below `0.2 m/s` from heading comparisons.

- [ ] **Step 4: Run the two tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Write failing tests for magnetometer, vibration, missing sensors, and frame rotation**

```python
def test_enhanced_magnetometer_and_vibration_features_are_finite():
    df = _df()
    df["mx"] = 1.0 + 0.1 * np.sin(df["time_s"])
    df["my"] = 0.2 * np.cos(df["time_s"])
    df["mz"] = 0.5
    df["ax"] = np.sin(10 * df["time_s"])
    out = window_extract(df).iloc[0]
    assert np.isfinite(out["e_mag_norm_mean"])
    assert out["e_mag_norm_std"] > 0.0
    assert out["e_vibration_rms"] > 0.0


def test_missing_magnetometer_yields_nan_enhanced_heading_features():
    df = _df()
    df[["mx", "my", "mz"]] = np.nan
    out = window_extract(df).iloc[0]
    assert np.isnan(out["e_yaw_mag_change_corr"])
    assert np.isnan(out["e_yaw_mag_resid_std"])


def test_level_body_acceleration_matches_navigation_horizontal_acceleration():
    df = _df()
    df["vel_e"] = df["time_s"]
    df["ax"] = 1.0
    df["ay"] = 0.0
    df["az"] = 9.81
    out = window_extract(df).iloc[0]
    assert out["e_imu_gps_acc_h_mean"] < 0.05
```

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_features.py -q
```

Expected: the new enhanced-feature assertions fail while existing tests remain green.

- [ ] **Step 7: Implement magnetometer, tilt-compensated change, vibration, and navigation-frame acceleration features**

Implement `_enhanced_feats` and call it from `_window_feats`. Construct `Rz(yaw) @ Ry(pitch) @ Rx(roll)` for every finite attitude sample, rotate `[ax, ay, az]`, and compare only the horizontal navigation components with gradients of `vel_e/vel_n`. Calculate magnetometer heading after roll/pitch tilt compensation, but emit only change correlation and residual spread.

The returned mapping must include every name listed in design sections 5.1-5.3 even when values are NaN.

- [ ] **Step 8: Run feature tests and full tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_features.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add -- src/features.py tests/test_features.py
git commit -m "Add paper-guided physical consistency features"
```

---

### Task 2: Freeze legacy feature contracts and add M2e

**Files:**
- Modify: `src/models.py`
- Modify: `src/models_m2.py`
- Modify: `src/models_m4.py`
- Modify: `scripts/run_pipeline.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_models_m2.py`
- Modify: `tests/test_models_m4.py`

**Interfaces:**
- Consumes: feature frames containing `f_`, `c_`, and `e_` columns.
- Produces: `_feature_columns(X, prefixes)` accepting `str | tuple[str, ...] | None`; `fit_m2e(X_tr, y_tr, seed)`; saved `m2e.joblib`.

- [ ] **Step 1: Write failing feature-contract tests**

```python
def test_feature_prefix_tuple_excludes_enhanced_columns():
    X = pd.DataFrame({"f_a": [0.0], "c_a": [0.0], "e_a": [0.0]})
    assert _feature_columns(X, ("f_", "c_")) == ["f_a", "c_a"]


def test_m2e_adds_enhanced_columns_without_changing_m2():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"f_a": rng.normal(size=40),
                      "c_a": rng.normal(size=40),
                      "e_a": rng.normal(size=40)})
    y = pd.Series([0, 1] * 20)
    legacy = fit_m2(X, y)
    enhanced = fit_m2e(X, y)
    assert legacy.feat_names_ == ["f_a", "c_a"]
    assert enhanced.feat_names_ == ["f_a", "c_a", "e_a"]
```

Add equivalent assertions that M1b and M4 explicitly exclude `e_`.

- [ ] **Step 2: Run model tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_models.py tests/test_models_m2.py tests/test_models_m4.py -q
```

Expected: FAIL because tuple prefixes and `fit_m2e` are unsupported and legacy all-feature models include `e_`.

- [ ] **Step 3: Implement explicit prefix selection and M2e**

Change `_feature_columns` filtering to:

```python
if prefixes is None:
    return feats
if isinstance(prefixes, str):
    prefixes = (prefixes,)
return [c for c in feats if isinstance(c, str) and c.startswith(prefixes)]
```

Make `fit_m2` use `("f_", "c_")`, add `fit_m2e` using `("f_", "c_", "e_")`, make M1b pass the legacy tuple, and make M4 fit only legacy prefixes. Store `feature_family_ = "legacy"` or `"enhanced"` on fitted models.

- [ ] **Step 4: Register M2e in the pipeline**

Add `m2e` to `ALL_MODELS`; train it with `fit_m2e`; save predictions using the existing `_finalize` path. Tiny mode must include M2e. M3 raw inputs remain unchanged.

- [ ] **Step 5: Verify model and full test suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_models.py tests/test_models_m2.py tests/test_models_m4.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- src/models.py src/models_m2.py src/models_m4.py scripts/run_pipeline.py tests/test_models.py tests/test_models_m2.py tests/test_models_m4.py
git commit -m "Preserve legacy models and add enhanced M2e ablation"
```

---

### Task 3: False-alarm-constrained validation threshold

**Files:**
- Modify: `src/evaluate.py`
- Modify: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: validation prediction frame with `label` and `prob_1`.
- Produces: `select_operating_threshold(df, max_false_alarm_rate=0.10) -> tuple[float, float]`; metrics fields `threshold_policy`, `f1_threshold`, `val_false_alarm_rate`.

- [ ] **Step 1: Write failing threshold tests**

```python
def test_operating_threshold_respects_validation_false_alarm_cap():
    val = pd.DataFrame({"label": [0, 0, 0, 0, 1, 1],
                        "prob_1": [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]})
    threshold, far = select_operating_threshold(val, max_false_alarm_rate=0.25)
    assert np.isclose(threshold, 0.7)
    assert far <= 0.25


def test_operating_threshold_has_no_alarm_candidate():
    val = pd.DataFrame({"label": [0, 0, 1], "prob_1": [1.0, 1.0, 1.0]})
    threshold, far = select_operating_threshold(val, max_false_alarm_rate=0.0)
    assert threshold > 1.0
    assert far == 0.0


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_operating_threshold_rejects_single_class_validation(labels):
    with pytest.raises(ValueError, match="normal and attacked"):
        select_operating_threshold(pd.DataFrame({"label": labels, "prob_1": [0.1, 0.9]}))
```

- [ ] **Step 2: Run threshold tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -q
```

Expected: FAIL because `select_operating_threshold` and new metric fields do not exist.

- [ ] **Step 3: Implement deterministic candidate selection**

Generate sorted unique scores plus `np.nextafter(max_score, np.inf)`. For each candidate, compute validation normal-window FAR, recall, and F1. Keep feasible candidates with FAR at most the cap and choose by `(recall, f1, threshold)` descending. Return threshold and FAR.

In `evaluate_all`, require a non-empty validation split, calculate `f1_threshold = best_threshold(val)`, select the formal threshold with the new function, and store:

```python
res = {
    "threshold": threshold,
    "threshold_source": "val",
    "threshold_policy": "val_far<=0.10_max_recall",
    "f1_threshold": f1_threshold,
    "val_false_alarm_rate": val_far,
}
```

Remove train fallback for formal experiment frames. Tests without `split` must be updated to supply explicit train/val/test data.

- [ ] **Step 4: Verify evaluation and full tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- src/evaluate.py tests/test_evaluate.py
git commit -m "Select operating thresholds under validation FAR constraint"
```

---

### Task 4: Shared temporal confirmation policy and offline metrics

**Files:**
- Create: `src/alarm.py`
- Create: `tests/test_alarm.py`
- Modify: `src/evaluate.py`
- Modify: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `AlarmDecision(raw_alert: bool, confirmed_alert: bool)` and `TemporalAlarmPolicy.update(probability: float) -> AlarmDecision`.
- Consumes: ordered probability streams; `evaluate_all` resets policy for every flight.

- [ ] **Step 1: Write failing policy tests**

```python
def test_temporal_policy_requires_three_of_five_scores():
    policy = TemporalAlarmPolicy(threshold=0.7)
    states = [policy.update(p).confirmed_alert for p in [0.8, 0.1, 0.9, 0.2, 0.95]]
    assert states == [False, False, False, False, True]


def test_temporal_policy_uses_hysteresis_to_clear():
    policy = TemporalAlarmPolicy(threshold=0.5)
    for p in [0.9, 0.9, 0.9]:
        policy.update(p)
    assert policy.update(0.45).confirmed_alert is True
    assert policy.update(0.39).confirmed_alert is True
    assert policy.update(0.39).confirmed_alert is True
    assert policy.update(0.39).confirmed_alert is False


def test_temporal_policy_reset_clears_flight_history():
    policy = TemporalAlarmPolicy(threshold=0.5)
    for p in [0.9, 0.9, 0.9]:
        policy.update(p)
    policy.reset()
    assert policy.update(0.1).confirmed_alert is False
```

- [ ] **Step 2: Run alarm tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_alarm.py -q
```

Expected: collection error because `src.alarm` does not exist.

- [ ] **Step 3: Implement the pure policy**

Use a `deque(maxlen=5)` for raw threshold results, require 3 positives to enter confirmed, and count consecutive scores strictly below `0.8 * threshold` to exit after 3. Validate probabilities are finite and `threshold > 0`.

- [ ] **Step 4: Verify alarm tests GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Add failing offline-confirmation metric test**

Create an explicit train/val/test fixture with isolated one-window test spikes and assert raw false alarms are nonzero while `confirmed_false_alarm_rate == 0.0`. Also assert `confirmed_recall`, `confirmed_missed_flights_rate`, and confirmed delay fields exist.

- [ ] **Step 6: Run evaluation test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -q
```

Expected: FAIL because confirmed metrics are absent.

- [ ] **Step 7: Apply the policy per flight during evaluation**

Sort each test flight by `window_start_s`, instantiate a new policy with the frozen threshold, write `confirmed_pred`, and compute confirmed precision/recall/F1, normal-window FAR, attacked-flight miss rate, and delay without removing the existing raw metrics.

- [ ] **Step 8: Verify evaluation and full tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_alarm.py tests/test_evaluate.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 4**

```powershell
git add -- src/alarm.py src/evaluate.py tests/test_alarm.py tests/test_evaluate.py
git commit -m "Add persistent alarm policy and confirmed metrics"
```

---

### Task 5: Use confirmed alarms in streaming, CLI, and API

**Files:**
- Modify: `src/stream.py`
- Modify: `src/cli.py`
- Modify: `src/server.py`
- Modify: `tests/test_stream.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `TemporalAlarmPolicy` from Task 4.
- Produces: stream result field `confirmed_alert`; `alarm_intervals()` based on confirmed alerts; API/CLI `spoofing` based on confirmation.

- [ ] **Step 1: Write failing stream tests**

```python
def test_single_high_window_is_not_confirmed_spoofing():
    scores = iter([0.9, 0.1, 0.1, 0.1, 0.1])
    streamer = CausalStreamer(lambda row: next(scores), threshold=0.5)
    results = [r for r in (streamer.feed(row) for row in _rows(WINDOW_SIZE + 4 * STRIDE)) if r]
    assert results[0]["alert"] is True
    assert not any(r["confirmed_alert"] for r in results)
    assert streamer.alarm_intervals() == []


def test_three_of_five_high_windows_confirm_stream_alarm():
    scores = iter([0.9, 0.1, 0.9, 0.1, 0.9])
    streamer = CausalStreamer(lambda row: next(scores), threshold=0.5)
    results = [r for r in (streamer.feed(row) for row in _rows(WINDOW_SIZE + 4 * STRIDE)) if r]
    assert results[-1]["confirmed_alert"] is True
    assert len(streamer.alarm_intervals()) == 1
```

- [ ] **Step 2: Run stream tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stream.py -q
```

Expected: FAIL because `confirmed_alert` is absent and intervals use raw alerts.

- [ ] **Step 3: Integrate `TemporalAlarmPolicy` into `CausalStreamer`**

Initialize the policy from `threshold`, call it after every emitted probability, preserve `alert`, add `confirmed_alert`, and store only confirmed results for interval merging.

- [ ] **Step 4: Write a failing API test for isolated high scores**

Add this serializable test model and exercise the real API and streamer with exactly one emitted window:

```python
class ConstantProbabilityModel:
    feat_names_ = []
    decision_threshold_ = 0.5

    def predict_proba(self, X):
        return np.tile([0.1, 0.9], (len(X), 1))


def test_api_does_not_confirm_a_single_high_window(tmp_path):
    import joblib
    path = tmp_path / "constant.joblib"
    joblib.dump(ConstantProbabilityModel(), path)
    body = TestClient(create_app(model_path=str(path))).post(
        "/predict", json={"flight_id": "one-window", "rows": _rows(41)}).json()
    assert body["raw_alert_windows"] == 1
    assert body["confirmed_alert_windows"] == 0
    assert body["spoofing"] is False
```

- [ ] **Step 5: Run API test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -q
```

Expected: FAIL because the schema and confirmation behavior are absent.

- [ ] **Step 6: Update CLI/API summaries**

Set `spoofing = any(r["confirmed_alert"] for r in results)`. Return raw and confirmed alert-window counts while keeping existing keys. Keep `confidence` as maximum raw model probability and label it as such in README/report; do not call it a calibrated probability.

- [ ] **Step 7: Verify stream, API, and full tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stream.py tests/test_api.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 5**

```powershell
git add -- src/stream.py src/cli.py src/server.py tests/test_stream.py tests/test_api.py
git commit -m "Require persistent evidence for online spoofing alarms"
```

---

### Task 6: Multi-model route holdout evaluation

**Files:**
- Modify: `scripts/run_route_holdout.py`
- Create: `tests/test_route_holdout.py`

**Interfaces:**
- Produces: `fit_predict_model(name: str, data: pd.DataFrame) -> pd.DataFrame`; CLI `--models`; rows named `<model>_holdout_<route>`.
- Consumes: legacy and enhanced model fit functions plus `evaluate_all`.

- [ ] **Step 1: Write failing route-model dispatch tests**

Use a compact synthetic frame with all three routes and explicit `f_`, `c_`, `e_` columns. Assert M0, M1, M1b, M2, M2e, and M4 each return predictions preserving `split`, and an unknown name raises `ValueError("unknown route-holdout model")`.

- [ ] **Step 2: Run route test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_route_holdout.py -q
```

Expected: collection or import failure because the dispatch function is absent.

- [ ] **Step 3: Extract model dispatch and add CLI selection**

Implement explicit branches for six supported models. M0 and M4 fit only training normal windows; supervised models fit all training windows. Add:

```python
ap.add_argument("--models", nargs="+",
                choices=("m0", "m1", "m1b", "m2", "m2e", "m4"),
                default=["m0", "m1", "m1b", "m2", "m2e", "m4"])
```

Loop models inside each route and write one metrics row per pair.

- [ ] **Step 4: Verify route and full tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_route_holdout.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```powershell
git add -- scripts/run_route_holdout.py tests/test_route_holdout.py
git commit -m "Evaluate legacy and enhanced models across routes"
```

---

### Task 7: Reproduce experiments and update public documentation

**Files:**
- Modify: `README.md`
- Modify: `report/report.md`
- Regenerate: `data_processed/features_windowed.parquet` (ignored)
- Regenerate: `outputs/metrics.csv`
- Regenerate: `outputs/metrics_route_holdout.csv`
- Regenerate: `outputs/figs/*`
- Regenerate: `outputs/models/*` (ignored)
- Modify tests only if a reproduced integration defect first receives a failing regression test.

**Interfaces:**
- Consumes: completed Tasks 1-6.
- Produces: reproducible artifacts and documentation whose numbers match those artifacts.

- [ ] **Step 1: Run the complete unit/integration suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run isolated tiny smoke reproduction**

```powershell
.\.venv\Scripts\python.exe scripts\run_all.py --tiny --skip-download
```

Expected: dataset, M0/M1/M1b/M2/M2e/M4 training, threshold selection, metrics, and figures complete under `*_tiny` paths without changing full outputs.

- [ ] **Step 3: Regenerate the full enhanced dataset**

```powershell
.\.venv\Scripts\python.exe scripts\make_dataset.py
```

Expected: 89,804 rows; split counts 62,554/8,533/18,717; feature count equals 71 plus the implemented `e_` feature count; no missing split values.

- [ ] **Step 4: Run the full model pipeline**

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Expected: model files including `m2e.joblib`, refreshed `outputs/metrics.csv`, and figures. M3 may only be reported as skipped if the script emits a concrete environment/runtime error.

- [ ] **Step 5: Run the full route holdout matrix**

```powershell
.\.venv\Scripts\python.exe scripts\run_route_holdout.py
```

Expected: 18 rows for six models times three routes, each with validation-derived threshold metadata and raw/confirmed metrics.

- [ ] **Step 6: Check artifact invariants**

Run a read-only pandas check that asserts exact split counts, no overlap among flight splits, all required metric columns, M2 and M2e rows, 18 route rows, and finite AUROC wherever both test classes exist. Print the actual metrics table for report drafting.

- [ ] **Step 7: Update README and report from actual outputs**

Document:

- exact enhanced feature count and feature families;
- M2e versus M2 ablation;
- formal 10% validation-FAR threshold policy and its test behavior;
- raw versus confirmed alarm metrics and added delay;
- all-model route holdouts;
- whether research targets improved or failed;
- the local-only paper references and prohibited direct transfer of paper thresholds;
- the fact that `confidence` is an uncalibrated maximum model score.

Never copy numeric values from command plans or old report prose; use only Step 6 output.

- [ ] **Step 8: Run final verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
git diff --check
git status --short
```

Also re-run the artifact invariant check and compare every report table value against the CSVs.

- [ ] **Step 9: Commit Task 7**

```powershell
git add -- README.md report/report.md outputs/metrics.csv outputs/metrics_route_holdout.csv outputs/figs src scripts tests
git commit -m "Complete paper-guided detector optimization and rerun"
```

Do not push until the user explicitly requests publication and the final commit and remote target have been shown.
