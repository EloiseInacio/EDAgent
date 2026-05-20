# EDAgent — workflow for Claude Code

This project is an EDA agent for manifest-based ML datasets. When a user gives you a CSV or SQL database and asks for dataset analysis, follow the steps below exactly.

## Pipeline

**0. Read before acting**
Fetch `edagent://questionnaire` and `edagent://workflow` from the MCP server before the first analysis in a session.

**1. Intake (before running any script)**
Ask the questions from `edagent://questionnaire`. Rules:
- Skip any question already answered by the user or clearly inferable from the data source.
- Ask no more than 4 questions per turn.
- Do not run scripts until you have enough context to interpret the results.

**1b. Materialize SQL manifest (SQL sources only)**
If the user provides a SQL database instead of a CSV file:
- Ask the SQL follow-up questions from `edagent://questionnaire` (connection string, primary table, joins, filters).
- Call `materialize_sql_manifest` with the connection details.
- Use the `manifest_path` returned in the JSON output as the `manifest_path` argument for all subsequent steps.
- For CSV sources, skip this step entirely — proceed directly to `profile_manifest`.

**2. profile_manifest — always first**
Run this before any other script. Use the output to:
- Confirm which columns are paths, labels, grouping variables, and temporal fields.
- Determine the inferred modality (skeleton/motion, video, image, text, tabular).
- Decide which modality-specific analysis branches are relevant in the next step.

**3. run_eda**
Run after profiling. Set `include_asset_metadata=True` only when:
- The profile reports a non-tabular modality, and
- The referenced asset files are locally accessible.

If assets are inaccessible, continue with manifest-level analysis and note the limitation explicitly.

**4. check_leakage**
Always run before writing the report. Do not skip this step.

**5. Write the report**
Use the structure from `edagent://report-template` (15 sections). Rules:
- Omit sections that are not supported by the data — do not fabricate content.
- Clearly label what is an observed fact, what is an assumption, and what is a recommendation.
- Tie every recommendation to a specific finding.
- Include code snippets only for metadata extraction or validation methods that directly support a reported finding.

## Epistemic discipline

These rules apply throughout:
- Confirmed finding: directly observed in the data or script output.
- Assumption: inferred from incomplete information — label it explicitly.
- Not assessable: state why (missing split column, inaccessible files, etc.).

Never overstate confidence. Never fill unsupported sections with generic commentary.
