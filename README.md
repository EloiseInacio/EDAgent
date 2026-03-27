# EDA Agent

An interactive exploratory data analysis agent for manifest-based datasets.

This agent is designed for workflows where the primary input is a **CSV manifest** that points to underlying assets or contains metadata columns directly. It asks a small number of intake questions to understand the downstream task, infers the structure of the manifest, runs reusable analysis scripts, and produces a senior-data-scientist-style EDA report with charts, commentary, leakage checks, and metadata extraction snippets.

## What it does

The agent is built to support EDA requests such as:

* “Write me an EDA report for this dataset.”
* “Analyze this CSV of file paths for fall detection.”
* “Check missingness, class balance, outliers, and leakage risks.”
* “Inspect a manifest containing skeleton keypoints, bbox metadata, fps, or text data.”

It is especially useful when:

* one row represents a frame, clip, sequence, subject, or sample
* the manifest contains path references to videos, skeleton files, text files, or other derived assets
* the user wants both quantitative checks and narrative interpretation
* the user needs modeling-risk assessment, not just descriptive statistics

## Core capabilities

### Interactive intake

The agent starts by asking a short set of targeted questions to clarify:

* the downstream task or application
* the unit of analysis for each row
* the label definition and granularity
* relevant split or grouping variables

It skips questions that are already answered by the user or inferable from the uploaded CSV.

### Manifest understanding

The agent profiles the uploaded manifest to infer:

* path columns
* label columns
* grouping or split columns
* temporal columns
* identifier columns
* likely modality

### Core EDA

The agent runs standard EDA checks including:

* missingness analysis
* label distribution
* numeric summaries
* outlier detection
* grouped summaries
* chart generation

### Leakage and evaluation risk analysis

The agent explicitly checks for:

* duplicate rows
* duplicate file references
* overlap across split and grouping identifiers
* path- or filename-based label shortcuts
* metadata features that may proxy the label
* temporal overlap risks
* missing grouping keys that can hide leakage

### Modality-aware metadata extraction

When supported by the manifest and accessible files, the agent can enrich EDA using helper extractors for:

* video metadata
* skeleton, bbox, fps, and motion metadata
* text metadata

### Report generation

The final report follows a structured template and includes:

* executive summary
* context and assumptions
* schema and metadata interpretation
* missingness and label analysis
* outlier findings
* leakage risks
* modality-specific findings
* concise code snippets for metadata extraction or validation
* actionable recommendations

## Repository structure

A typical skill layout looks like this:

```text
eda-agent/
├── SKILL.md
├── README.md
├── env.yaml
├── agents/
│   └── openai.yaml
├── scripts/
│   ├── profile_manifest.py
│   ├── run_core_eda.py
│   ├── check_leakage.py
│   ├── extract_meta.py
│   ├── extract_skeleton_meta.py
│   └── extract_text_meta.py
└── references/
    ├── questionnaire.md
    └── report-template.md
```

## Main scripts

### `scripts/profile_manifest.py`

Profiles the uploaded CSV manifest and writes `manifest_profile.json`.

Responsibilities:

* load the CSV manifest
* infer semantic roles for columns
* summarize missingness and uniqueness
* identify candidate path, label, grouping, temporal, and ID columns
* summarize file extensions for path fields
* estimate likely dataset modality

Primary output:

* `manifest_profile.json`

### `scripts/run_core_eda.py`

Runs core EDA and writes structured outputs for the report.

Responsibilities:

* compute missingness
* compute label distribution
* summarize numeric columns
* flag IQR-based outliers
* compute grouped summaries
* generate charts
* optionally derive asset-level metadata from accessible files

Primary outputs:

* `eda_outputs/eda_summary.json`
* `eda_outputs/eda_summary.md`
* chart PNG files
* optional derived asset metadata files

### `scripts/check_leakage.py`

Runs reusable leakage and evaluation-risk checks.

Responsibilities:

* detect duplicate rows
* detect duplicate paths
* check overlap across split and group identifiers
* identify filename or path tokens correlated with labels
* flag likely proxy features
* evaluate temporal overlap heuristics

Primary outputs:

* `leakage_report.json`
* `leakage_report.md`

## Metadata helper modules

### `extract_meta.py`

Used for video-oriented metadata extraction.

Example use cases:

* video duration and dimensions
* optical flow amplitude
* frame-level signal proxies

### `extract_skeleton_meta.py`

Used for skeleton and motion metadata extraction.

Example use cases:

* duration from pose sequences
* keypoint visibility and confidence
* bbox area ratios
* velocity or jitter proxies
* limb-length stability

### `extract_text_meta.py`

Used for textual metadata extraction.

Example use cases:

* document length metrics
* lexical diversity
* repetition metrics
* noise and cleanliness signals
* readability proxies
* simple label-keyword overlap heuristics

## Workflow

The agent follows this high-level workflow:

1. Start with intake questions from `references/questionnaire.md`.
2. Skip questions already answered by the user or inferable from the uploaded CSV.
3. Ask no more than 4 questions in the first turn.
4. Run `scripts/profile_manifest.py` to create `manifest_profile.json`.
5. Use the manifest profile to decide which modality-specific analyses are supported.
6. Run `scripts/run_core_eda.py` using the manifest and `manifest_profile.json`.
7. When relevant and accessible, let `run_core_eda.py` call helper extractors for video, skeleton, or text metadata.
8. Run `scripts/check_leakage.py` using the manifest and `manifest_profile.json`.
9. Generate the final report using `references/report-template.md`.
10. Ground conclusions in script outputs and explicitly state any limitations.

## Supported data patterns

The agent is currently best suited for:

* CSV manifests with file paths
* manifests with path columns plus metadata columns
* structured video or motion datasets
* skeleton / bbox / fps metadata workflows
* text datasets with inline text columns or text file paths
* supervised learning settings where label balance and leakage matter

## Example use cases

### Fall detection from skeleton metadata

Input:

* CSV manifest with paths to pose files or derived data
* columns containing bbox, fps, labels, split info, or subject identifiers

Useful checks:

* class balance
* missing keypoints or low-confidence joints
* abnormal fps or bbox patterns
* subject overlap across splits
* path names that encode the label

### Text classification dataset

Input:

* CSV manifest with inline text or paths to text, JSON, or JSONL files
* labels and grouping columns

Useful checks:

* document length distribution
* lexical diversity
* repetitive or noisy samples
* label imbalance
* label tokens appearing directly in text or path structure

## Environment setup

Create the conda environment:

```bash
conda env create -f env.yaml
```

Activate it:

```bash
conda activate eda-agent
```

Update it after editing `env.yaml`:

```bash
conda env update -f env.yaml --prune
```

## Running the pipeline manually

You can test the backend scripts independently before packaging the skill.

### 1. Profile the manifest

```bash
python scripts/profile_manifest.py data/your_manifest.csv
```

### 2. Run core EDA

```bash
python scripts/run_core_eda.py data/your_manifest.csv \
  --profile-json manifest_profile.json \
  --include-asset-metadata
```

### 3. Run leakage checks

```bash
python scripts/check_leakage.py data/your_manifest.csv \
  --profile-json manifest_profile.json
```

## Using with Claude Code

The agent integrates with Claude Code via an MCP server that exposes the EDA pipeline as tools and resources.

### Setup

1. Create and activate the conda environment (see [Environment setup](#environment-setup)).
2. Open this project directory in Claude Code. The `.mcp.json` at the project root registers the MCP server automatically — no additional configuration is needed.

### What the MCP server provides

**Resources** — fetched by Claude at the start of each session:

| URI | Content |
|-----|---------|
| `edagent://workflow` | Pipeline instructions and workflow rules |
| `edagent://questionnaire` | Intake questions to ask before analysis |
| `edagent://report-template` | 15-section report structure |

**Tools** — called by Claude during analysis:

| Tool | Step | Description |
|------|------|-------------|
| `profile_manifest` | 1 | Infer column roles and dataset modality, write `manifest_profile.json` |
| `run_eda` | 2 | Compute statistics, detect outliers, generate charts, write `eda_summary.json` |
| `check_leakage` | 3 | Detect duplicates, split overlap, proxy features, write `leakage_report.json` |
| `read_eda_file` | any | Read any output file produced by the pipeline |

### Usage

With the project open in Claude Code, give Claude a CSV path or ask for a dataset analysis:

```
Analyze data/my_manifest.csv — it's a fall detection dataset with skeleton pose files.
```

Claude will:

1. Fetch the workflow and questionnaire from the MCP server.
2. Ask a short set of intake questions (up to 4 per turn), skipping anything already clear from the CSV.
3. Call `profile_manifest`, `run_eda`, and `check_leakage` in sequence.
4. Write the final report using the 15-section template, grounding every claim in script output.

The `CLAUDE.md` in this repository encodes the workflow rules for Claude — it is read automatically and requires no manual steps.

### MCP server transport

The server runs over stdio via:

```bash
conda run -n eda-agent python mcp_server.py
```

This is handled automatically by `.mcp.json`. To run the server manually for debugging:

```bash
conda activate eda-agent
python mcp_server.py
```

## Using with OpenAI

The MCP server supports an HTTP/SSE transport mode for use with the OpenAI Agents SDK and the ChatGPT desktop app. Both require the server to be running locally — the EDA scripts execute on the local filesystem and expect the manifest and asset files to be locally accessible.

### Prerequisites

Create and activate the conda environment (see [Environment setup](#environment-setup)), then start the MCP server in SSE mode:

```bash
conda activate eda-agent
python mcp_server.py --transport sse --port 8000
```

The server will listen at `http://127.0.0.1:8000/sse`. Keep this process running while using the agent.

### OpenAI Agents SDK

Connect an MCP client to the running server and pass the system prompt from `reference/system_prompt_openai.md`:

```python
from agents import Agent, MCPServerSse

async def main():
    async with MCPServerSse(url="http://127.0.0.1:8000/sse") as mcp:
        with open("reference/system_prompt_openai.md") as f:
            system_prompt = f.read()
        agent = Agent(
            name="EDAgent",
            instructions=system_prompt,
            mcp_servers=[mcp],
        )
        # run your agent loop here
```

The agent will have access to `profile_manifest`, `run_eda`, `check_leakage`, and `read_eda_file` as tools.

### ChatGPT desktop app

1. Start the MCP server in SSE mode (see above).
2. In the ChatGPT desktop app, open **Settings → MCP Servers** and add a new server with the URL `http://127.0.0.1:8000/sse`.
3. Start a new conversation and paste the contents of `reference/system_prompt_openai.md` as the system prompt (or into the first message if system prompts are not configurable).
4. Provide the path to your CSV manifest and the agent will follow the EDA workflow.

### System prompt

`reference/system_prompt_openai.md` contains the full workflow instructions for OpenAI-based clients. It consolidates intake questions, tool execution order, decision rules, epistemic discipline, and the 15-section report structure — everything that Claude Code loads automatically via MCP resources but that OpenAI clients require as an explicit system prompt.

### Limitation

Both integrations require the MCP server to run on the same machine as the data. Remote or cloud-hosted deployments are not supported by the current setup — the EDA scripts read files from the local filesystem only.

## Design principles

This agent is built around a few strong design principles:

### Separate mechanics from reasoning

Use scripts for deterministic computation and use the skill instructions for:

* intake questions
* modality routing
* report writing
* interpretation
* recommendations

### Be explicit about uncertainty

The agent should distinguish between:

* confirmed findings from the data
* likely inferences
* unsupported or unassessable checks

### Optimize for modeling usefulness

The goal is not just to summarize a dataset. The goal is to surface issues that matter for:

* data validity
* evaluation quality
* deployment realism
* annotation quality
* hidden confounding factors

## Current limitations

The current version assumes:

* the primary entry point is a CSV manifest
* uploaded files are the main source of data
* helper extractors are available locally when asset-level metadata is needed

The agent may be limited when:

* referenced assets are inaccessible
* row semantics are ambiguous and not clarified during intake
* split or grouping identifiers are missing
* modality-specific metadata is absent or inconsistently encoded

## License and usage

Adapt this agent to your own workflows, datasets, and domain-specific checks. The current version is a strong starting point for manifest-based EDA in computer vision, motion analysis, and text-heavy pipelines.
