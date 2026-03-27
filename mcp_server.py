"""MCP server exposing EDAgent pipeline as tools for Claude Code."""

import json
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).parent
SCRIPT_DIR = PROJECT_ROOT / "script"

mcp = FastMCP("EDAgent")


@mcp.resource("edagent://workflow")
def get_workflow() -> str:
    """EDAgent workflow instructions (SKILL.md). Read before starting any EDA session."""
    return (PROJECT_ROOT / "SKILL.md").read_text()


@mcp.resource("edagent://questionnaire")
def get_questionnaire() -> str:
    """Intake questions to ask the user before running analysis."""
    return (PROJECT_ROOT / "reference" / "questionnaire.md").read_text()


@mcp.resource("edagent://report-template")
def get_report_template() -> str:
    """15-section EDA report template. Use this to structure the final report."""
    return (PROJECT_ROOT / "reference" / "report_template.md").read_text()


def _run(args: list[str], cwd: Path = PROJECT_ROOT) -> tuple[int, str, str]:
    result = subprocess.run(
        [sys.executable] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.returncode, result.stdout, result.stderr


def _read_output(path: Path) -> str:
    if not path.exists():
        return f"[file not found: {path}]"
    return path.read_text()


@mcp.tool()
def profile_manifest(
    manifest_path: str,
    output_dir: str = "eda_outputs",
    check_path_exists: bool = False,
    base_path: str = "",
) -> str:
    """Step 1/3 — always run before run_eda or check_leakage.
    Infers column roles (path, label, grouping, temporal, identifier), detects dataset modality
    (skeleton/motion, video, image, text, tabular), and writes manifest_profile.json.
    Use the profile output to decide which modality-specific branches are relevant in run_eda."""
    manifest = Path(manifest_path)
    out_dir = manifest.parent / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_json = out_dir / "manifest_profile.json"

    args = [
        str(SCRIPT_DIR / "data_profile.py"),
        str(manifest),
        "--output", str(profile_json),
    ]
    if check_path_exists:
        args.append("--check-path-exists")
    if base_path:
        args += ["--base-path", base_path]

    rc, stdout, stderr = _run(args)
    if rc != 0:
        return f"[data_profile.py failed]\nstdout: {stdout}\nstderr: {stderr}"

    return _read_output(profile_json)


@mcp.tool()
def run_eda(
    manifest_path: str,
    output_dir: str = "eda_outputs",
    include_asset_metadata: bool = False,
    base_path: str = "",
) -> str:
    """Step 2/3 — run after profile_manifest, before check_leakage.
    Computes missingness, label distribution, numeric summaries, outlier detection, grouped
    summaries, and charts. Writes eda_summary.json and eda_summary.md.
    Set include_asset_metadata=True when profile_manifest reports a non-tabular modality
    (skeleton/motion, video, text) and the referenced asset files are locally accessible.
    If assets are inaccessible, omit this flag and note the limitation in the report."""
    manifest = Path(manifest_path)
    out_dir = manifest.parent / output_dir
    profile_json = out_dir / "manifest_profile.json"

    args = [
        str(SCRIPT_DIR / "run_core_eda.py"),
        str(manifest),
        "--profile-json", str(profile_json),
        "--output-dir", str(out_dir),
    ]
    if include_asset_metadata:
        args.append("--include-asset-metadata")
    if base_path:
        args += ["--base-path", base_path]

    rc, stdout, stderr = _run(args)
    if rc != 0:
        return f"[run_core_eda.py failed]\nstdout: {stdout}\nstderr: {stderr}"

    summary_json = out_dir / "eda_summary.json"
    summary_md = out_dir / "eda_summary.md"
    parts = []
    if summary_json.exists():
        parts.append("=== eda_summary.json ===\n" + _read_output(summary_json))
    if summary_md.exists():
        parts.append("=== eda_summary.md ===\n" + _read_output(summary_md))
    return "\n\n".join(parts) if parts else f"[no output files found]\nstdout: {stdout}\nstderr: {stderr}"


@mcp.tool()
def check_leakage(
    manifest_path: str,
    output_dir: str = "eda_outputs",
) -> str:
    """Step 3/3 — run after run_eda, before writing the report.
    Checks for: duplicate rows/paths, entity overlap across splits, filename-token to label
    correlation, proxy features encoding the label, and temporal overlap risks.
    Returns leakage_report.json and leakage_report.md. Each finding includes a severity score,
    risk level, and recommendation — use these directly in report section 9."""
    manifest = Path(manifest_path)
    out_dir = manifest.parent / output_dir
    profile_json = out_dir / "manifest_profile.json"
    leakage_json = out_dir / "leakage_report.json"
    leakage_md = out_dir / "leakage_report.md"

    args = [
        str(SCRIPT_DIR / "check_leakage.py"),
        str(manifest),
        "--profile-json", str(profile_json),
        "--output", str(leakage_json),
        "--markdown-output", str(leakage_md),
    ]

    rc, stdout, stderr = _run(args)
    if rc != 0:
        return f"[check_leakage.py failed]\nstdout: {stdout}\nstderr: {stderr}"

    parts = []
    if leakage_json.exists():
        parts.append("=== leakage_report.json ===\n" + _read_output(leakage_json))
    if leakage_md.exists():
        parts.append("=== leakage_report.md ===\n" + _read_output(leakage_md))
    return "\n\n".join(parts) if parts else f"[no output files found]\nstdout: {stdout}\nstderr: {stderr}"


@mcp.tool()
def read_eda_file(file_path: str) -> str:
    """Read any EDA output file (JSON, markdown, chart path list, etc.)."""
    return _read_output(Path(file_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
