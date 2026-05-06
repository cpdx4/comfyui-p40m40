#!/usr/bin/env bash
# scripts/upstream_merge.sh
# =========================
# Safely merge upstream ComfyUI commits into the local submodule while
# automatically skipping commits that introduce incompatible features.
#
# Usage:
#   ./scripts/upstream_merge.sh              # fetch + merge compatible commits
#   ./scripts/upstream_merge.sh --dry-run    # show what would happen, no changes
#   ./scripts/upstream_merge.sh --interactive # pause at each PATCH commit for review
#
# Requirements: git, python3, jq (optional, for pretty output)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFYUI_DIR="$REPO_ROOT/ComfyUI"
UPSTREAM_REMOTE="upstream"
UPSTREAM_BRANCH="master"
LOG_FILE="$REPO_ROOT/upstream_merge.log"
SKIP_LOG="$REPO_ROOT/skip_log.txt"

DRY_RUN=false
INTERACTIVE=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        --interactive) INTERACTIVE=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Patterns that indicate a commit MUST be skipped or patched
# ---------------------------------------------------------------------------

# Commits with ANY of these patterns in their diff are SKIPPED entirely
SKIP_PATTERNS=(
    "float8_e4m3fn"
    "float8_e5m2"
    "float8_e4m3fnuz"
    "float8_e5m2fnuz"
    "comfy_kitchen"
    "import triton"
    "@triton.jit"
    "sm_90"
    "sm_89"
    "cuda_12"
    "CUDA 12"
    "flash_attn_cuda"
    "triton_attention"
)

# Commits with these patterns need a patch run after cherry-pick
NEEDS_PATCH_PATTERNS=(
    "torch.compile("
    "enable_flash=True"
    "enable_mem_efficient=True"
    "bfloat16"
    "sdpa_kernel("
    "scale=.*scaled_dot_product"
)

# Files that are always safe to accept regardless of content
SAFE_FILE_PATTERNS=(
    "^web/"
    "^comfyui_version"
    "\.js$"
    "\.ts$"
    "\.css$"
    "\.scss$"
    "\.html$"
    "\.md$"
    "\.json$"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_skip() { echo "$*" >> "$SKIP_LOG"; }

commit_matches_pattern() {
    local commit="$1"; shift
    local patterns=("$@")
    local diff
    diff=$(git -C "$COMFYUI_DIR" show --unified=0 "$commit" 2>/dev/null || true)
    for pattern in "${patterns[@]}"; do
        if echo "$diff" | grep -q "$pattern"; then
            return 0
        fi
    done
    return 1
}

commit_only_safe_files() {
    local commit="$1"
    local changed_files
    changed_files=$(git -C "$COMFYUI_DIR" diff-tree --no-commit-id -r --name-only "$commit" 2>/dev/null)
    while IFS= read -r file; do
        local is_safe=false
        for pattern in "${SAFE_FILE_PATTERNS[@]}"; do
            if echo "$file" | grep -qE "$pattern"; then
                is_safe=true
                break
            fi
        done
        if ! $is_safe; then
            return 1  # At least one non-safe file
        fi
    done <<< "$changed_files"
    return 0
}

categorize_commit() {
    local commit="$1"
    local subject
    subject=$(git -C "$COMFYUI_DIR" log --format="%s" -1 "$commit")

    # Auto-accept if only safe files changed
    if commit_only_safe_files "$commit"; then
        echo "AUTO_ACCEPT"
        return
    fi

    # Skip if any skip pattern found
    if commit_matches_pattern "$commit" "${SKIP_PATTERNS[@]}"; then
        echo "SKIP"
        return
    fi

    # Needs patching if any patch pattern found
    if commit_matches_pattern "$commit" "${NEEDS_PATCH_PATTERNS[@]}"; then
        echo "PATCH"
        return
    fi

    echo "AUTO_ACCEPT"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "=== ComfyUI P40/M40 Upstream Merge ==="
log "DRY_RUN=$DRY_RUN  INTERACTIVE=$INTERACTIVE"

# Ensure we're in the submodule
if [ ! -d "$COMFYUI_DIR/.git" ] && [ ! -f "$COMFYUI_DIR/.git" ]; then
    log "ERROR: ComfyUI submodule not initialized."
    log "Run: git submodule update --init --recursive"
    exit 1
fi

# Add upstream remote if not present
if ! git -C "$COMFYUI_DIR" remote | grep -q "^$UPSTREAM_REMOTE$"; then
    log "Adding upstream remote: https://github.com/comfyanonymous/ComfyUI.git"
    git -C "$COMFYUI_DIR" remote add "$UPSTREAM_REMOTE" \
        "https://github.com/comfyanonymous/ComfyUI.git"
fi

# Fetch upstream
log "Fetching from upstream..."
git -C "$COMFYUI_DIR" fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" 2>&1 | tee -a "$LOG_FILE"

# Get the range of new commits
CURRENT_HEAD=$(git -C "$COMFYUI_DIR" rev-parse HEAD)
UPSTREAM_HEAD=$(git -C "$COMFYUI_DIR" rev-parse "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH")

if [ "$CURRENT_HEAD" = "$UPSTREAM_HEAD" ]; then
    log "Already up to date."
    exit 0
fi

NEW_COMMITS=$(git -C "$COMFYUI_DIR" log --format="%H" "$CURRENT_HEAD..$UPSTREAM_HEAD" | tac)
COMMIT_COUNT=$(echo "$NEW_COMMITS" | grep -c . || true)
log "Found $COMMIT_COUNT new commits to process."

APPLIED=0
SKIPPED=0
PATCHED=0
FAILED=0
CONFLICTS=()

while IFS= read -r commit; do
    [ -z "$commit" ] && continue

    SUBJECT=$(git -C "$COMFYUI_DIR" log --format="%s" -1 "$commit")
    SHORT=$(git -C "$COMFYUI_DIR" log --format="%h" -1 "$commit")
    CATEGORY=$(categorize_commit "$commit")

    log "[$SHORT] [$CATEGORY] $SUBJECT"

    case "$CATEGORY" in
        SKIP)
            log "  → Skipping (incompatible: FP8/Triton/comfy_kitchen/CUDA12)"
            log_skip "$(date -I) SKIP $SHORT $SUBJECT"
            SKIPPED=$((SKIPPED + 1))
            ;;

        AUTO_ACCEPT)
            if $DRY_RUN; then
                log "  → Would cherry-pick (auto-accept)"
            else
                if git -C "$COMFYUI_DIR" cherry-pick --no-commit "$commit" 2>&1 | tee -a "$LOG_FILE"; then
                    git -C "$COMFYUI_DIR" commit --no-edit \
                        -m "upstream: $SUBJECT" \
                        -m "[P40-COMPAT auto-accept] cherry-pick $SHORT" \
                        2>&1 | tee -a "$LOG_FILE"
                    APPLIED=$((APPLIED + 1))
                else
                    log "  ERROR: Cherry-pick conflict on $SHORT"
                    git -C "$COMFYUI_DIR" cherry-pick --abort 2>/dev/null || true
                    CONFLICTS+=("$SHORT: $SUBJECT")
                    FAILED=$((FAILED + 1))
                fi
            fi
            ;;

        PATCH)
            if $DRY_RUN; then
                log "  → Would cherry-pick + re-apply patches"
                continue
            fi

            if $INTERACTIVE; then
                log "  PATCH commit — showing diff:"
                git -C "$COMFYUI_DIR" show --stat "$commit" | head -30
                echo -n "  Apply and re-patch? [y/N/s(skip)] "
                read -r response
                case "$response" in
                    [yY]) ;;
                    [sS])
                        log "  → User skipped"
                        SKIPPED=$((SKIPPED + 1))
                        continue
                        ;;
                    *)
                        log "  → User declined"
                        FAILED=$((FAILED + 1))
                        continue
                        ;;
                esac
            fi

            log "  → Cherry-picking + re-applying patches..."
            if git -C "$COMFYUI_DIR" cherry-pick --no-commit "$commit" 2>&1 | tee -a "$LOG_FILE"; then
                git -C "$COMFYUI_DIR" commit --no-edit \
                    -m "upstream: $SUBJECT" \
                    -m "[P40-COMPAT patched] cherry-pick $SHORT" \
                    2>&1 | tee -a "$LOG_FILE"

                # Re-apply source patches
                log "  → Re-running patches/apply_all.py..."
                python "$REPO_ROOT/patches/apply_all.py" 2>&1 | tee -a "$LOG_FILE" || {
                    log "  WARNING: Some patches failed after cherry-pick. Manual review needed."
                }
                PATCHED=$((PATCHED + 1))
            else
                log "  ERROR: Conflict applying PATCH commit $SHORT"
                git -C "$COMFYUI_DIR" cherry-pick --abort 2>/dev/null || true
                CONFLICTS+=("$SHORT: $SUBJECT")
                FAILED=$((FAILED + 1))
            fi
            ;;
    esac
done <<< "$NEW_COMMITS"

# ---------------------------------------------------------------------------
# Post-merge validation
# ---------------------------------------------------------------------------
if ! $DRY_RUN && [ $APPLIED -gt 0 ] || [ $PATCHED -gt 0 ]; then
    log ""
    log "Running compatibility validation..."
    python "$REPO_ROOT/scripts/validate_compat.py" 2>&1 | tee -a "$LOG_FILE" || {
        log "WARNING: Validation found issues. Review output above."
    }
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=== Merge Summary ==="
log "  Applied (auto):  $APPLIED"
log "  Applied (patch): $PATCHED"
log "  Skipped:         $SKIPPED"
log "  Failed/Conflict: $FAILED"

if [ ${#CONFLICTS[@]} -gt 0 ]; then
    log ""
    log "Commits requiring manual resolution:"
    for c in "${CONFLICTS[@]}"; do
        log "  !! $c"
    done
fi

log ""
log "Full log: $LOG_FILE"
log "Skip log: $SKIP_LOG"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
