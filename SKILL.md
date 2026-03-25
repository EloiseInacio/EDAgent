# Workflow

1. Start with the intake questions in [reference/questionnaire.md](reference/questionnaire.md).
2. Skip any question already answered by the user or inferable from the uploaded CSV.
3. Ask no more than 4 questions in the first turn.
4. Run `script/profile_manifest.py` on the uploaded CSV to infer column roles, identify candidate path/label/grouping fields, summarize schema, and write `manifest_profile.json`.
5. Use the manifest profile to decide which analyses are supported and which modality-specific branches are relevant.
6. Run `script/run_core_eda.py` using the uploaded CSV and `manifest_profile.json` to compute missingness, label distribution, numeric summaries, outlier checks, grouped summaries, and charts, and to write `eda_summary.json` and chart outputs.
7. When asset-level metadata is relevant and accessible, let `script/run_core_eda.py` enrich the analysis using:
   - `extract_meta.py` for video metadata
   - `extract_skeleton_meta.py` for skeleton, bbox, fps, and motion-related metadata
   - `extract_text_meta.py` for text-path or text-column metadata
8. Run `script/check_leakage.py` using the uploaded CSV and `manifest_profile.json` to detect duplicate rows, duplicate file references, split/entity overlap, path-based label shortcuts, proxy features, temporal overlap risks, and missing grouping identifiers, and to write `leakage_report.json`.
9. Generate the final EDA report using the structure in [reference/report_template.md](reference/report_template.md).
10. Ground the report in outputs from:
    - `manifest_profile.json`
    - `eda_summary.json`
    - `leakage_report.json`
    - any derived asset metadata and charts produced during analysis
11. Omit unsupported sections instead of fabricating content.
12. Clearly separate observations, assumptions, and recommendations throughout the report.
13. Include concise code snippets in the report only for metadata extraction or validation methods that directly support reported findings.

## Execution rules

- Always run `script/profile_manifest.py` first.
- Use `manifest_profile.json` as the shared input for downstream analysis scripts.
- Run `script/run_core_eda.py` before writing the report.
- Run `script/check_leakage.py` before writing the leakage and evaluation-risk sections.
- Use helper extractors only when the manifest and accessible files support them.
- If underlying assets are inaccessible, continue with manifest-level analysis and state the limitation explicitly.