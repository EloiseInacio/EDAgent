# EDA Report Template

## 1. Executive Summary
Provide a concise summary for a technical stakeholder.

Include:
- dataset purpose and inferred modality
- main data quality findings
- class balance summary
- major outliers or anomalies
- leakage risks
- top 3 recommendations before modeling

Rules:
- Keep this section short
- State high-confidence findings first
- Distinguish observed facts from assumptions

---

## 2. Analysis Context
Summarize the user-provided context and the assumptions used for this report.

Include:
- downstream task or application
- assumed unit of analysis (frame, clip, sequence, subject, sample)
- label granularity
- whether analysis is based only on the manifest or also on referenced files
- any unresolved ambiguities

Format:
- Explicitly label assumptions as assumptions
- Be concise and thorough

---

## 3. Dataset Overview
Describe the dataset at a high level.

Include:
- number of rows
- number of columns
- column names and inferred roles
- detected file path columns
- detected label columns
- detected grouping or split columns
- inferred modality from filenames, schema, and user context

Suggested commentary:
- Explain what one row likely represents
- Mention whether the dataset appears frame-level, clip-level, sequence-level, or mixed

---

## 4. Schema and Metadata Interpretation
Document how the manifest was interpreted.

Include a table or structured list with:
- column name
- inferred semantic role
- data type
- example values
- interpretation confidence
- notes

Example roles:
- identifier
- path/reference
- label/target
- grouping variable
- temporal field
- categorical metadata
- numeric measurement
- derived feature
- unknown

Commentary requirements:
- Call out ambiguous columns
- Note columns that may encode shortcuts or leakage

---

## 5. Data Access and Integrity Checks
Assess whether the referenced data appears usable.

Include:
- missing or empty paths
- duplicated paths
- duplicated samples
- unreadable file references if checked
- suspicious file extensions
- path-pattern irregularities

Interpretation:
- Explain whether issues are isolated or systematic
- State how these issues may affect downstream modeling

---

## 6. Missingness Analysis
Quantify and interpret missing data.

Include:
- missingness by column
- missingness by label
- missingness by group if relevant
- co-missingness patterns
- missingness in key metadata fields

Charts:
- bar chart of missingness by column
- heatmap or grouped summary if useful

Interpretation:
- Distinguish structural missingness from likely data-quality failures
- Highlight columns where missingness may bias training or evaluation

---

## 7. Label Distribution
Assess target balance and label structure.

Include:
- class counts
- class proportions
- missing labels
- rare labels
- label distribution by split/group if available

Charts:
- class count bar chart
- optional grouped label chart

Interpretation:
- Comment on imbalance severity
- Comment on whether labels appear aligned with the stated task
- Note any suspiciously clean or suspiciously noisy label patterns

---

## 8. Outlier and Anomaly Analysis
Identify unusual values, samples, or groups.

Include:
- numeric outliers in metadata columns
- rare categorical values
- suspicious durations, fps values, bbox sizes, or keypoint statistics
- anomalous path or naming patterns
- sample-level anomalies if computed

Charts:
- histograms or box plots for key numeric fields
- scatter plot for suspicious relationships where relevant

Interpretation:
- Distinguish likely true edge cases from likely data errors
- Avoid recommending automatic removal unless clearly justified

---

## 9. Leakage and Evaluation Risk Assessment
Evaluate whether the dataset may overstate model performance.

Always check for:
- duplicate rows
- duplicate file references
- label leakage through filenames or path structure
- subject/session/camera/environment overlap across splits
- temporal overlap across train/test if applicable
- metadata features that may proxy the label too directly

Report:
- each leakage hypothesis
- evidence supporting it
- confidence level
- likely impact on evaluation validity

Interpretation:
- Be explicit about whether leakage is confirmed, suspected, or not assessable

---

## 10. Modality-Specific Analysis
Adapt this section to the inferred data type.

### For skeleton / motion / fall-detection-style data
Consider:
- fps consistency
- clip duration distribution
- bbox width, height, area, and aspect ratio
- keypoint completeness
- missing-joint frequency
- motion magnitude or temporal variability
- jitter or implausible coordinate jumps
- per-class movement differences

### For visual datasets
Consider:
- image resolution
- bbox coverage and aspect ratio
- annotation density
- blur/noise proxies
- signal-to-noise indicators where available

### For temporal or sensor datasets
Consider:
- sampling regularity
- sequence length distribution
- drift
- missing windows
- channel-wise anomalies

### For audio datasets
Consider:
- duration distribution across clips
- sample rate consistency across files
- mono vs. stereo and channel count variation
- RMS energy distribution (loudness variation within and across classes)
- silence ratio (high values may indicate annotation misalignment or mic issues)
- clipping ratio (values near 1.0 indicate recording overload)
- spectral centroid by class (useful for speech vs. music vs. noise separation)
- zero-crossing rate distribution (high for noisy/fricative speech, low for tonal audio)
- spectral bandwidth variation per class

Rules:
- Only include analyses that are justified by the data
- State when a modality-specific check could not be performed

---

## 11. Metadata Extraction Methods
Provide concise code snippets showing how metadata was extracted or validated.

Include snippets for relevant tasks such as:
- loading the manifest
- identifying path columns
- extracting file extensions
- checking file existence
- parsing fps, bbox, or keypoint-related columns
- computing sequence-level aggregates
- detecting duplicates
- checking split/group overlap

Rules:
- Use short, runnable snippets
- Add a one-line explanation before each snippet
- Include only snippets directly tied to findings in the report

---

## 12. Key Risks for Modeling
Summarize the practical implications for model development.

Include:
- risks to training quality
- risks to evaluation validity
- risks to deployment realism
- risks from annotation noise
- risks from hidden confounders

Format:
- ordered list from highest to lowest concern

---

## 13. Recommendations
Provide clear next steps.

Split recommendations into:
- immediate fixes before modeling
- additional checks to run
- data collection or annotation improvements
- split redesign if leakage risk exists

Rules:
- Make recommendations specific and actionable
- Tie each recommendation to a finding above

---

## 14. Limitations of This EDA
State what this report could and could not assess.

Include:
- inaccessible referenced files
- assumptions made about row semantics
- missing split identifiers
- inability to verify certain leakage hypotheses
- limited confidence in inferred column meanings

Rules:
- Be transparent
- Do not overclaim

---

## 15. Appendix
Optional supporting material.

Can include:
- schema summary table
- per-column statistics
- additional charts
- excluded columns and why
- anomaly examples