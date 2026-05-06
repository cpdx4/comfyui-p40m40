"""
scripts/triage_upstream.py
===========================
Analyses new upstream ComfyUI commits and categorises each as:
  AUTO_ACCEPT  — safe to cherry-pick without patching
  PATCH        — cherry-pick + re-run patches/apply_all.py
  SKIP         — incompatible; do not apply

Used by CI (upstream_check.yml) and can be run locally:
  python scripts/triage_upstream.py
  python scripts/triage_upstream.py --output report.md --json report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFYUI_DIR = REPO_ROOT / "ComfyUI"

# Same patterns as upstream_merge.sh (keep in sync)
SKIP_PATTERNS = [
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
    "comfy_kitchen", "import triton", "@triton.jit", "sm_90", "sm_89",
    "cuda_12", "CUDA 12", "flash_attn_cuda", "triton_attention",
]

PATCH_PATTERNS = [
    "torch.compile(", "enable_flash=True", "enable_mem_efficient=True",
    "bfloat16", "sdpa_kernel(", "scale=",
]

SAFE_FILE_EXTENSIONS = {".js", ".ts", ".css", ".scss", ".html", ".md", ".json"}
SAFE_FILE_PREFIXES = ("web/",)


def git(*args: str, cwd: Path = COMFYUI_DIR) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    return result.stdout.strip()


def get_new_commits() -> List[str]:
    current = git("rev-parse", "HEAD")
    try:
        upstream = git("rev-parse", "upstream/master")
    except Exception:
        return []
    if current == upstream:
        return []
    commits = git("log", "--format=%H", f"{current}..{upstream}")
    return [c for c in commits.splitlines() if c]


def get_diff(commit: str) -> str:
    return git("show", "--unified=0", commit)


def get_changed_files(commit: str) -> List[str]:
    out = git("diff-tree", "--no-commit-id", "-r", "--name-only", commit)
    return [f for f in out.splitlines() if f]


def is_safe_file(path: str) -> bool:
    p = Path(path)
    if p.suffix in SAFE_FILE_EXTENSIONS:
        return True
    for prefix in SAFE_FILE_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def categorize(commit: str) -> str:
    changed_files = get_changed_files(commit)
    if all(is_safe_file(f) for f in changed_files):
        return "AUTO_ACCEPT"

    diff = get_diff(commit)

    for pattern in SKIP_PATTERNS:
        if pattern in diff:
            return "SKIP"

    for pattern in PATCH_PATTERNS:
        if pattern in diff:
            return "PATCH"

    return "AUTO_ACCEPT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Write markdown report to this file")
    parser.add_argument("--json", help="Write JSON results to this file")
    args = parser.parse_args()

    commits = get_new_commits()
    if not commits:
        print("No new commits upstream.")
        if args.json:
            Path(args.json).write_text("[]")
        return

    results: List[Dict] = []
    for commit in reversed(commits):  # oldest first
        subject = git("log", "--format=%s", "-1", commit)
        short = commit[:8]
        category = categorize(commit)
        results.append({"hash": short, "full_hash": commit, "subject": subject, "category": category})
        icon = {"AUTO_ACCEPT": "✅", "PATCH": "🔧", "SKIP": "⛔"}.get(category, "?")
        print(f"  {icon} [{short}] [{category}] {subject}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))

    if args.output:
        auto = [r for r in results if r["category"] == "AUTO_ACCEPT"]
        patch = [r for r in results if r["category"] == "PATCH"]
        skip = [r for r in results if r["category"] == "SKIP"]

        lines = [
            f"## Upstream Triage Report ({len(results)} commits)\n",
            f"- ✅ Auto-accept: {len(auto)}",
            f"- 🔧 Needs patch: {len(patch)}",
            f"- ⛔ Skipped:     {len(skip)}\n",
        ]
        if skip:
            lines += ["\n### ⛔ Skipped (incompatible)"] + [
                f"- `{r['hash']}` {r['subject']}" for r in skip
            ]
        if patch:
            lines += ["\n### 🔧 Needs Patching"] + [
                f"- `{r['hash']}` {r['subject']}" for r in patch
            ]
        if auto:
            lines += ["\n### ✅ Auto-Accept"] + [
                f"- `{r['hash']}` {r['subject']}" for r in auto
            ]
        Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
