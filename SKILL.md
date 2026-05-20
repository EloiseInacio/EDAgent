# Workflow

1. Start with the intake questions in [reference/questionnaire.md](reference/questionnaire.md).
2. Skip any question already answered by the user or inferable from the data source.
3. Ask no more than 4 questions in the first turn.
4. **SQL sources only:** if the user provides a SQL database instead of a CSV file, ask the SQL follow-up questions (connection string, primary table, joins, filters), then run `script/load_sql.py` (via the `materialize_sql_manifest` MCP tool) to produce a CSV manifest. Use the returned path as the manifest for all subsequent steps. For CSV sources, skip this step.
5. Run `script/data_profile.py` on the manifest CSV to infer column roles, identify candidate path/label/grouping fields, summarize schema, and write `eda_outputs/manifest_profile.json`.
6. Use the manifest profile to decide which analyses are supported and which modality-specific branches are relevant.
7. Run `script/run_core_eda.py` using the manifest CSV and `eda_outputs/manifest_profile.json` to compute missingness, label distribution, numeric summaries, outlier checks, grouped summaries, and charts, and to write `eda_outputs/eda_summary.json` and chart outputs.
8. When asset-level metadata is relevant and accessible, let `script/run_core_eda.py` enrich the analysis using:
   - `extract_meta.py` for video metadata
   - `extract_skeleton_meta.py` for skeleton, bbox, fps, and motion-related metadata
   - `extract_text_meta.py` for text-path or text-column metadata
9. Run `script/check_leakage.py` using the manifest CSV and `eda_outputs/manifest_profile.json` to detect duplicate rows, duplicate file references, split/entity overlap, path-based label shortcuts, proxy features, temporal overlap risks, and missing grouping identifiers, and to write `eda_outputs/leakage_report.json`.
10. Generate the final EDA report using the structure in [reference/report_template.md](reference/report_template.md).
11. Ground the report in outputs from:
    - `eda_outputs/manifest_profile.json`
    - `eda_outputs/eda_summary.json`
    - `eda_outputs/leakage_report.json`
    - any derived asset metadata and charts produced during analysis
12. Omit unsupported sections instead of fabricating content.
13. Clearly separate observations, assumptions, and recommendations throughout the report.
14. Include concise code snippets in the report only for metadata extraction or validation methods that directly support reported findings.

## Execution rules

- Always run `script/data_profile.py` first.
- Use `eda_outputs/manifest_profile.json` as the shared input for downstream analysis scripts.
- Run `script/run_core_eda.py` before writing the report.
- Run `script/check_leakage.py` before writing the leakage and evaluation-risk sections.
- Use helper extractors only when the manifest and accessible files support them.
- If underlying assets are inaccessible, continue with manifest-level analysis and state the limitation explicitly.