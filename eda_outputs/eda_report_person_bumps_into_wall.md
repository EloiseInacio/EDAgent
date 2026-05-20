# EDA Report — `person_bumps_into_wall`

**Source manifest**: `meta_skel_person_bumps_into_wall.csv`
**Generated**: 2026-03-26
**Analysis basis**: manifest-level (full 2476 rows) + asset-level skeleton enrichment (50 sampled .npz files)

---

## 1. Executive Summary

This is a single-class skeleton/motion dataset of 2,476 clip-level samples representing a "person bumps into wall" action. Each row corresponds to one `.npz` clip containing COCO 17-joint pose sequences extracted from video.

**Main quality findings**:
- Data quality is generally good. Overall missing cell rate is 0.1%, limited to motion and limb-variance features, co-missing in small consistent batches — consistent with degenerate clips (zero-frame trace or no valid joint transitions).
- The FPS distribution is bimodal (~51% at ~15 fps, ~49% at ~30 fps), indicating two acquisition sources. This creates a de facto subpopulation split that is invisible in the schema.
- 4 clips have `trace_length = 0` (no frame with ≥10 visible joints), 27 have `n_frames < 10`, and 3 have `avg_confidence = 0` — all are likely unusable for training.
- Motion outliers are significant: `velocity_max` reaches 569 px/frame, `velocity_variance` reaches 20,884, with 17 clips exceeding a variance of 1,000 and 22 clips with peak velocity >200 px/frame. These are almost certainly tracker failures rather than real motion.

**Label summary**: Single class — no balance assessment applicable.

**Leakage risks**: All formal checks are low. However, the 553 unique UUID-prefixed recording sessions (mean 4.48 segments each) are not labelled as a grouping key. Any train/test split that ignores UUID boundaries will constitute entity leakage.

**Top 3 recommendations before modeling**:
1. Filter degenerate clips: remove `trace_length == 0`, `avg_confidence == 0`, and `n_frames < 6` (at 15 fps this is <0.4s).
2. Add a `uuid` / session grouping column and enforce group-disjoint splits.
3. Decide whether to stratify by FPS tier or treat them as separate subsets before combining with negative examples.

---

## 2. Analysis Context

| Item | Value |
|------|-------|
| Downstream task | Inferred: binary or multi-class action recognition / anomaly detection; exact task not provided |
| Unit of analysis | Clip-level (one row = one `.npz` skeleton sequence) |
| Label granularity | Single class present: `person_bumps_into_wall`; negative class absent from this manifest |
| Analysis scope | Full manifest (2476 rows) + asset enrichment on 50 head-sampled .npz files from `/home/eloise/nas/dataset/cap_dataset_metadata/person_bumps_into_wall/` |
| Skeleton format | COCO 17-joint pose, keypoints shape `(n_frames, 17, 3)` — x, y, confidence |

**Assumptions**:
- Each UUID prefix in the filename identifies a unique recording session or subject.
- The dataset is intended for use alongside a negative class from a separate manifest.
- `visibility` encodes mean visible joints per frame (max 17), not a binary flag.
- Limb variance units are pixel² variance of limb length across frames.

**Unresolved**:
- No subject identity column: cannot confirm whether UUID = subject or session.
- No split column: cannot assess whether any prior split was constructed.
- No negative class in this manifest: imbalance and leakage across classes cannot be evaluated here.

---

## 3. Dataset Overview

- **Rows**: 2,476
- **Columns**: 19
- **Modality**: Skeleton / motion (COCO 17-joint, `.npz`)
- **Category**: Single — `person_bumps_into_wall` (constant column)
- **Unique recordings (UUIDs)**: 553 — segments per UUID range from 1 to 9 (mean 4.48)
- **Resolution**: 512 × 910 px (dominant), with a small subset at 512 × 682 or 512 × 768
- **FPS groups**: ~15 fps (51.4% of clips) and ~30 fps (48.6%)

Each row represents one contiguous skeleton clip extracted from a longer recording. The filename pattern `<UUID>_<segment_index>.npz` links segments to their parent recording. Up to 9 segments per UUID are present, suggesting clips were extracted with a sliding or fixed window. This is clip-level, not frame-level data.

---

## 4. Schema and Metadata Interpretation

| Column | Inferred Role | Type | Notes |
|--------|---------------|------|-------|
| `category` | Categorical metadata | string | **Constant** — single value across all rows; zero-variance; drop from features |
| `filename` | Path / reference | string | Basename only (`.npz`); 100% unique; UUID prefix encodes recording session |
| `n_frames` | Numeric measurement | int | Total frame count; range 4–216 |
| `fps` | Temporal field | float | **Bimodal**: ~15 and ~30 fps tiers |
| `width` | Numeric measurement | int | **Constant** — always 512; zero-variance; drop from features |
| `height` | Numeric measurement | int | Mostly 910, minor subset at 682/768 — likely different source crops |
| `duration_seconds` | Temporal field | float | Derived: `n_frames / fps`; range 0.21–11.6s |
| `visibility` | Numeric measurement | float | Mean visible joints/frame (conf ≥ 0.3); max 17 |
| `trace_length` | Numeric measurement | int | Longest contiguous run of frames with ≥10 visible joints |
| `avg_confidence` | Numeric measurement | float | Mean keypoint confidence across all joints and frames |
| `confidence_variance` | Numeric measurement | float | Variance of keypoint confidence |
| `velocity_mean` | Numeric measurement | float | Mean joint displacement (px/frame); 6 missing |
| `velocity_max` | Numeric measurement | float | Peak single-joint displacement; 6 missing |
| `velocity_variance` | Numeric measurement | float | Variance of joint velocity; **extreme outliers** present; 6 missing |
| `bbox_area_ratio` | Numeric measurement | float | Mean bbox area / frame area |
| `limb_var_left_shoulder_elbow` | Numeric measurement | float | Limb length variance; 7 missing |
| `limb_var_right_shoulder_elbow` | Numeric measurement | float | Limb length variance; 8 missing |
| `limb_var_left_hip_knee` | Numeric measurement | float | Limb length variance; 7 missing |
| `limb_var_right_hip_knee` | Numeric measurement | float | Limb length variance; 7 missing |

**Ambiguous columns**:
- `visibility` and `trace_length` are partially redundant — both capture keypoint quality, but at different temporal granularities. Neither is a label, but low values can proxy for occlusion or poor acquisition conditions.
- The three velocity columns are all NaN for the same 6 rows, which co-occur with `trace_length = 0` or single-frame clips where no valid joint transitions exist.

---

## 5. Data Access and Integrity Checks

| Check | Result |
|-------|--------|
| Missing paths | 0 |
| Duplicate rows | 0 |
| Duplicate filenames | 0 |
| File extension | `.npz` exclusively — consistent |
| Asset access (50 sampled) | 50/50 accessible and parseable |
| Path format irregularity | None — all follow `<UUID>_<int>.npz` |

All 2,476 filenames are unique and follow a consistent pattern. The 50 sampled `.npz` files were loaded successfully; all contained valid `keypoints (n_frames, 17, 3)`, `boxes (n_frames, 4)`, `fps`, `width`, and `height` fields.

One structural note: filenames are stored as basenames only. Any code that resolves paths must prepend the base directory `/home/eloise/nas/dataset/cap_dataset_metadata/person_bumps_into_wall/`. The manifest does not encode this base path.

---

## 6. Missingness Analysis

**Overall missing cell rate**: 0.001 (very low)

| Column | Missing count | Missing rate |
|--------|--------------|--------------|
| `limb_var_right_shoulder_elbow` | 8 | 0.32% |
| `limb_var_left_shoulder_elbow` | 7 | 0.28% |
| `limb_var_left_hip_knee` | 7 | 0.28% |
| `limb_var_right_hip_knee` | 7 | 0.28% |
| `velocity_mean` | 6 | 0.24% |
| `velocity_max` | 6 | 0.24% |
| `velocity_variance` | 6 | 0.24% |
| All others | 0 | 0.0% |

**Interpretation**: Missing values in the three velocity columns are co-missing (same 6 rows), consistent with clips where no valid joint transitions exist — either single-frame clips or clips where all joints fall below confidence threshold in consecutive frames. The `extract_skeleton_meta.py` extractor returns `NaN` for velocity when no valid transitions exist (`compute_joint_velocity_stats` returns NaN when `valid_speeds.size == 0`).

Limb variance missingness (7–8 rows) follows the same pattern: `compute_limb_length_variance` returns NaN when fewer than 2 valid frames exist for a given limb. These are likely the same degenerate clips. The one-row discrepancy between limb (7–8) and velocity (6) missing suggests one additional clip where at least one limb is occluded throughout but enough valid transitions exist for velocity computation.

**Risk**: Structural — not random. These rows represent clips with severe tracking failure and should be inspected before inclusion in training data.

**Chart**: `eda_outputs/missingness_by_column.png`

---

## 7. Label Distribution

**Label column**: Not identified — `category` is constant (`person_bumps_into_wall`) across all 2,476 rows.

This manifest represents a single-class export. Label balance assessment requires the paired negative-class manifest. As-is, any model trained solely on this data would learn a degenerate (always-positive) predictor.

No label distribution chart was generated.

---

## 8. Outlier and Anomaly Analysis

IQR-based outlier counts on the full 2,476-row manifest:

| Column | Outlier count | Rate | Max value | Notes |
|--------|--------------|------|-----------|-------|
| `visibility` | 478 | 19.3% | 17.0 | Artificially high rate — IQR is extremely tight (16.91–17.06); most "outliers" are valid values in [16.4, 16.9]. Not concerning. |
| `velocity_variance` | 278 | 11.2% | 20,884 | Extreme right tail; 17 clips exceed 1,000. Likely tracker failures (jitter). |
| `limb_var_right_shoulder_elbow` | 238 | 9.6% | 802 | High variance in shoulder-elbow distance across frames — arm in fast motion or tracking instability. |
| `limb_var_left_shoulder_elbow` | 234 | 9.5% | 664 | Same. |
| `limb_var_left_hip_knee` | 222 | 9.0% | 739 | — |
| `limb_var_right_hip_knee` | 194 | 7.8% | 563 | — |
| `confidence_variance` | 135 | 5.5% | 0.142 | High within-clip confidence instability — joints appear/disappear across frames. |
| `velocity_mean` | 115 | 4.6% | 65.1 | Fast-motion clips; likely valid edge cases of aggressive collision. |
| `avg_confidence` | 114 | 4.6% | min=0.0 | Low-confidence clips; 3 clips at 0.0 — full pose estimation dropout. |
| `bbox_area_ratio` | 89 | 3.6% | 0.338 | Some clips where the person occupies a large fraction of the frame. |
| `velocity_max` | 61 | 2.5% | 569 px/frame | 22 clips exceed 200 px/frame. At 30 fps, 569 px/frame = ~17,000 px/s on a 512-wide frame — physically implausible. Tracker jumps. |
| `duration_seconds` | 26 | 1.1% | 11.6s | Long outliers (>5.7s); likely multi-event clips. |
| `n_frames` | 11 | 0.44% | 216 | Consistent with long-duration outliers. |

**Chart**: `eda_outputs/hist_n_frames.png`, `hist_fps.png`, `hist_duration_seconds.png`, `hist_visibility.png`

**Key concerns**:

1. **`velocity_max` > 200 px/frame (22 clips)** and **`velocity_variance` > 1,000 (17 clips)**: These are almost certainly tracker failures where a joint coordinate teleports between frames after a detection miss. At 30 fps, a 569 px/frame displacement on a 512-wide frame is geometrically impossible for a human limb. These clips can produce large gradient noise during training if velocity features are used.

2. **`avg_confidence = 0` (3 clips)**: Full pose estimation dropout — no usable skeleton data. Should be excluded.

3. **`trace_length = 0` (4 clips)**: No qualifying frame in the clip. Clip contributes no meaningful skeleton signal.

4. **`n_frames < 10` (27 clips)**: At 15 fps, <10 frames = <0.67s. Insufficient temporal context for motion-based models. At 30 fps, <10 frames = <0.33s.

5. **FPS bimodal split**: Mean=20.9, median=15.0, p75=29.98 — ~51% of clips at ~15 fps, ~49% at ~30 fps. This is not documented in the schema. Models that use raw frame-count features (e.g., `n_frames`, `trace_length`) without normalizing by fps will observe a systematic bias between the two acquisition groups. `duration_seconds` is the fps-normalized equivalent and should be preferred.

6. **`height` variation**: Three distinct values (682, 768, 910). The 910 group is dominant; the minority groups likely represent different video crop settings or device aspect ratios. This has no impact on the skeleton keypoints (which are in absolute pixel coordinates normalized to the recorded width/height), but it affects `bbox_area_ratio` comparability across groups.

---

## 9. Leakage and Evaluation Risk Assessment

**Overall leakage risk (automated checks)**: Low (score 0.1 / 1.0)

| Check | Severity | Risk level | Finding |
|-------|----------|-----------|---------|
| Duplicate rows | 0.0 | Low | No duplicates |
| Duplicate filenames | 0.0 | Low | All 2,476 filenames unique |
| Path → label shortcuts | 0.0 | Low | No label column to correlate against |
| Proxy features | 0.0 | Low | No label column available |
| Temporal overlap across splits | 0.0 | Low | No split column identified |
| Missing group keys | 0.0 | Low | No declared grouping columns |

**Latent leakage risk — NOT captured by automated checks**:

**Entity leakage via UUID**: Each filename encodes a UUID prefix (e.g., `0004674B-2E99-41DF-8ECE-34D8D609187C`) that links multiple segments to a common recording session. There are 553 unique UUIDs with 1–9 segments each (mean 4.48). If segments from the same UUID appear in both train and test, the model can overfit to subject-specific appearance, motion style, or background. This is a high-severity risk for evaluation validity if splits are constructed without UUID-level grouping.

**Unassessable risks**:
- No negative-class manifest was provided. Cross-class entity overlap and proxy features cannot be evaluated.
- Without subject identifiers, it is unknown whether a single subject appears in multiple UUIDs (multi-session subjects).
- No temporal timestamps: if the dataset spans a real-world deployment period, temporal drift cannot be checked.

---

## 10. Modality-Specific Analysis: Skeleton / Motion

### FPS consistency
The dataset has two distinct fps tiers:
- ~15 fps: 51.4% of clips (p25=14.5, median=15.0)
- ~30 fps: 48.6% of clips (p75=29.98, max=30.21)

This is consistent with two recording hardware configurations or two video sources at different native frame rates. Clips at 15 fps have shorter `n_frames` for equivalent `duration_seconds`. Any feature that is not fps-normalized (raw frame counts, raw velocities without fps scaling) will encode acquisition hardware identity.

**Recommendation**: Always use `duration_seconds` instead of `n_frames` as the temporal extent feature. For velocity features, the current extraction already computes displacement in px/frame; at minimum, segment the dataset by fps tier before comparing velocity distributions.

### Clip duration distribution
- Mean: 2.94s, median: 2.81s, p25: 2.20s, p75: 3.60s
- Right tail: max 11.6s, 26 clips (1.1%) exceed 5.7s IQR upper bound
- Left tail: 27 clips have `n_frames < 10` (< ~0.67s at 15 fps)
- The bulk of clips are in the 2–4s range — this appears to be a segmentation design targeting the collision event itself

Asset-level enrichment on 50 samples confirms: duration mean=2.93s, std=0.76s — consistent with the full manifest.

### Keypoint completeness
- `visibility` mean: 16.66 / 17, median: 17.0 — excellent keypoint coverage in the typical case
- Asset enrichment: `mean_visible_joints` mean=16.94 / 17 — confirms nearly all joints are consistently visible
- The IQR outliers in `visibility` (478 clips, 19.3%) are artefacts of the very tight IQR, not actual quality failures — values like 16.4–16.9 are still high-quality observations

### Motion magnitude
Full-manifest velocity statistics:
- `velocity_mean`: mean=3.78, median=3.07 px/frame — moderate and broadly consistent with slow-to-moderate body motion
- `velocity_max`: mean=49.2, median=39.5 px/frame — peak displacement is significantly higher, reflecting the collision moment
- `velocity_variance`: mean=67.5, but median=10.8 and std=667 — the mean is dominated by a small number of extreme outliers. The median is a more representative central tendency.

Asset enrichment (50 samples) produces consistent statistics: velocity_mean mean=3.16, velocity_max mean=48.2 — the 50-sample subset is representative.

### Limb length variance
All four limb features (shoulder-elbow, hip-knee, both sides) have broadly consistent distributions:
- Median ~14–15 px² variance — low, consistent with stable anatomy tracking
- Right tails extend to 664–802 px² — large arm swing or detector instability
- Left-right asymmetry in outlier rates (shoulder: ~9.5%; hip-knee: ~8–9%) is modest and expected given that arm motion during a wall-bump collision is directional

### Skeletal jitter / implausible displacements
- `velocity_max` > 200 px/frame: 22 clips. At 30 fps and 512px width, this is a displacement of >39% of frame width per frame — physically impossible for a human joint. These are detector reset events (tracking loss → new detection at a different coordinate).
- `velocity_variance` > 1000: 17 clips. High variance confirms repeated jitter rather than a single jump.
- These 22 and 17 clips likely partially overlap. A conservative filter would be `velocity_max < 200 OR velocity_variance < 500` to remove jitter-dominated clips.

### Bbox coverage
- `bbox_area_ratio` mean=0.072 (7.2% of frame area), median=0.065 — person occupies a relatively small portion of the frame
- Max=0.338 — some clips where person is very close to the camera
- The asset enrichment confirms: `mean_bbox_area_ratio` mean=0.084 on the 50-sample subset

---

## 11. Metadata Extraction Methods

Verify the UUID grouping structure from filenames:

```python
import re
import pandas as pd

df = pd.read_csv("meta_skel_person_bumps_into_wall.csv")
df["uuid"] = df["filename"].str.extract(r"^([A-Za-z0-9\-]+)_\d+\.npz$")
print(df["uuid"].nunique())            # 553 unique recording sessions
print(df.groupby("uuid").size().describe())  # 1–9 segments, mean 4.48
```

Detect degenerate clips prior to training:

```python
degenerate = df[
    (df["trace_length"] == 0) |
    (df["avg_confidence"] == 0) |
    (df["n_frames"] < 10)
]
print(len(degenerate))  # ~30 clips (some overlap)
```

Detect likely tracker-failure clips based on velocity:

```python
jitter_mask = (df["velocity_max"] > 200) | (df["velocity_variance"] > 1000)
print(jitter_mask.sum())  # ~30 clips
```

Construct a UUID-grouped split (requires adding a split column):

```python
from sklearn.model_selection import GroupShuffleSplit

uuids = df["uuid"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=uuids))
```

Verify fps-normalized duration matches `n_frames / fps`:

```python
import numpy as np
diff = np.abs(df["duration_seconds"] - df["n_frames"] / df["fps"])
print(diff.max())  # should be near-zero
```

---

## 12. Key Risks for Modeling

1. **Entity leakage** *(highest concern)*: No UUID / session grouping column. Any random split will leak recording-level information to the test set, inflating evaluation metrics. Impact: unknown but potentially large depending on how much within-session variability there is.

2. **Tracker failure contamination**: ~20–30 clips with `velocity_max > 200` or `velocity_variance > 1000` represent pose estimation artifacts. Training on these will add noise to motion features and may cause gradient instability if velocity features are included.

3. **FPS-tier confound**: Two acquisition sources with ~15 and ~30 fps create a hidden subpopulation. Raw frame-count features encode this implicitly. If the negative class is acquired at a single fps tier, the model may learn fps as a label proxy.

4. **Degenerate clips**: ~30 clips with `trace_length = 0`, `avg_confidence = 0`, or `n_frames < 10` have no usable skeleton signal. Including them in training adds noise; including them in evaluation biases metrics.

5. **Constant / zero-variance features**: `category` and `width` are constant. Including them in feature vectors will not cause failures, but they waste capacity and can cause issues with normalization pipelines that divide by std.

6. **No negative class in this manifest**: Evaluation validity for the target downstream task cannot be assessed. Class balance, inter-class leakage, and proxy features are all unassessable without the paired negative manifest.

7. **Limb variance right-tail outliers** (~9–10% of clips): High values may reflect genuine collision dynamics (arm impact on wall) or tracking instability. These should not be removed without visual inspection, but they will affect distribution assumptions for any tree-based or distance-based method that uses these features.

---

## 13. Recommendations

### Immediate fixes before modeling

1. **Add a `uuid` column** by extracting the UUID prefix from `filename`. Use it as the group key for all splits. This is a five-line fix (see §11) and is the single highest-priority action.

2. **Filter degenerate clips**: Remove rows where `trace_length == 0`, `avg_confidence == 0`, or `n_frames < 6`. This eliminates ~30 rows (<1.2% of data) with no usable signal.

3. **Flag tracker-failure clips**: Mark or remove rows with `velocity_max > 200 OR velocity_variance > 1000` (~30 rows, <1.2%). Spot-check a sample visually before removing — some may represent genuine high-velocity collisions.

4. **Drop zero-variance columns** `category` and `width` from any feature matrix.

5. **Prefer `duration_seconds` over `n_frames`** as the temporal extent feature to avoid encoding acquisition fps.

### Additional checks to run

6. **Compute fps tier distribution for the negative-class manifest** and confirm it matches this manifest's bimodal split. Fps imbalance between positive and negative classes is a potential label proxy.

7. **Investigate `height` minority values** (682, 768): identify which UUIDs produce these and whether they cluster in any temporal or source-specific way.

8. **Inspect a random sample of jitter-flagged clips** (`velocity_max > 200`) visually to separate genuine extreme motion from tracker resets.

9. **Cross-reference UUID overlap** with any other action class manifests to ensure the same recording session does not contribute to multiple classes.

### Data collection / annotation improvements

10. **Add explicit subject / session identifier** as a first-class column in future manifest exports. The UUID extraction heuristic is fragile and undocumented.

11. **Add `base_path` as a manifest column or config field** to make asset resolution reproducible without modifying path values.

12. **Standardize fps at collection time** or explicitly document acquisition sources. The 15/30 fps bimodal split should be an explicit metadata field, not inferred post-hoc.

### Split redesign

13. If this dataset is combined with negatives for a binary classifier: use `GroupShuffleSplit` with UUID as the group key. Stratify by `fps` tier if the negative class has a different fps distribution.

---

## 14. Limitations of This EDA

- **Asset enrichment limited to 50 clips**: The derived skeleton statistics (§10) are based on 50 head-sampled files, not the full 2,476. The sample is representative for central tendencies but may underrepresent the full outlier distribution. Jitter counts reported in §8 are from the full manifest.

- **No subject identity**: UUID prefix is assumed to represent a unique recording session. Whether multiple UUIDs correspond to the same physical subject is unknown. Entity leakage risk may be higher than assessed if subjects appear across UUIDs.

- **No negative class**: All leakage checks involving label columns (path→label shortcuts, proxy features) returned "not applicable." The full leakage picture requires the paired negative-class manifest.

- **No split column**: Temporal overlap and split entity overlap checks could not be executed. These are currently the highest-risk unassessed dimensions.

- **Inferred column semantics**: Column roles were inferred from names and distributions. `visibility`, `trace_length`, and limb variance definitions were verified against `extract_skeleton_meta.py` source code but not against the original pose estimation pipeline.

- **Velocity units**: Reported in px/frame as computed by `compute_joint_velocity_stats`. Without a known px-to-meter calibration, physical plausibility bounds are approximations based on image-space geometry.

---

## 15. Appendix

### A. Full numeric summary (manifest, n=2476)

| Column | Count | Mean | Std | Min | p25 | Median | p75 | Max |
|--------|-------|------|-----|-----|-----|--------|-----|-----|
| `n_frames` | 2476 | 61.1 | 33.0 | 4 | 35 | 54 | 84 | 216 |
| `fps` | 2476 | 20.94 | 8.71 | 9.91 | 14.54 | 15.04 | 29.98 | 30.21 |
| `width` | 2476 | 512.0 | 0.0 | 512 | 512 | 512 | 512 | 512 |
| `height` | 2476 | 906.4 | 23.8 | 682 | 910 | 910 | 910 | 910 |
| `duration_seconds` | 2476 | 2.94 | 1.06 | 0.21 | 2.20 | 2.81 | 3.60 | 11.60 |
| `visibility` | 2476 | 16.66 | 1.62 | 0.0 | 16.96 | 17.0 | 17.0 | 17.0 |
| `trace_length` | 2476 | 59.6 | 33.6 | 0 | 33 | 53 | 84 | 216 |
| `avg_confidence` | 2476 | 0.704 | 0.095 | 0.0 | 0.679 | 0.718 | 0.750 | 0.866 |
| `confidence_variance` | 2476 | 0.015 | 0.011 | 0.0 | 0.010 | 0.013 | 0.016 | 0.142 |
| `velocity_mean` | 2470 | 3.78 | 3.14 | 0.79 | 2.17 | 3.07 | 4.54 | 65.09 |
| `velocity_max` | 2470 | 49.2 | 44.0 | 3.38 | 22.4 | 39.5 | 65.2 | 569.3 |
| `velocity_variance` | 2470 | 67.5 | 667.5 | 0.29 | 4.69 | 10.81 | 25.56 | 20884 |
| `bbox_area_ratio` | 2476 | 0.072 | 0.038 | 0.0 | 0.045 | 0.065 | 0.089 | 0.338 |
| `limb_var_left_shoulder_elbow` | 2469 | 33.0 | 55.4 | 0.06 | 5.36 | 14.82 | 36.26 | 664.2 |
| `limb_var_right_shoulder_elbow` | 2468 | 32.0 | 54.2 | 0.06 | 5.54 | 13.96 | 34.84 | 801.7 |
| `limb_var_left_hip_knee` | 2469 | 26.8 | 40.1 | 0.12 | 6.20 | 13.90 | 30.71 | 738.9 |
| `limb_var_right_hip_knee` | 2469 | 26.5 | 37.4 | 0.16 | 6.70 | 14.62 | 31.89 | 563.0 |

### B. Asset enrichment summary (n=50 sampled clips)

| Column | Mean | Std | Min | Median | Max |
|--------|------|-----|-----|--------|-----|
| `duration_seconds` | 2.93 | 0.76 | 1.60 | 2.90 | 4.80 |
| `fps` | 25.16 | 7.31 | 10.0 | 29.98 | 30.01 |
| `mean_visible_joints` | 16.94 | 0.15 | 16.38 | 17.0 | 17.0 |
| `trace_length` | 75.24 | 31.96 | 16 | 75 | 144 |
| `avg_keypoint_confidence` | 0.724 | 0.046 | 0.641 | 0.722 | 0.844 |
| `mean_bbox_area_ratio` | 0.084 | 0.038 | 0.028 | 0.082 | 0.166 |
| `velocity_mean` | 3.16 | 2.06 | 1.26 | 2.70 | 11.38 |
| `velocity_max` | 48.2 | 32.9 | 8.58 | 45.7 | 158.3 |
| `velocity_variance` | 13.25 | 14.81 | 1.14 | 9.30 | 65.17 |

### C. Degenerate clip counts

| Filter | Count | % of total |
|--------|-------|-----------|
| `trace_length == 0` | 4 | 0.16% |
| `avg_confidence == 0` | 3 | 0.12% |
| `n_frames < 10` | 27 | 1.09% |
| `velocity_max > 200` | 22 | 0.89% |
| `velocity_variance > 1000` | 17 | 0.69% |

### D. Charts produced

- `eda_outputs/missingness_by_column.png`
- `eda_outputs/hist_n_frames.png`
- `eda_outputs/hist_fps.png`
- `eda_outputs/hist_width.png`
- `eda_outputs/hist_height.png`
- `eda_outputs/hist_duration_seconds.png`
- `eda_outputs/hist_visibility.png`
- `eda_outputs/derived_asset_metadata.csv` — full skeleton metadata for 50 sampled clips
