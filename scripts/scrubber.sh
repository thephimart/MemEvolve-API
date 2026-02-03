#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo

confirm() {
  local prompt="$1"
  read -r -p "$prompt (y/N) " reply
  reply=${reply:-N}
  [[ "$reply" =~ ^[yY]$ ]]
}

# -------------------------
# PYC FILES
# -------------------------
mapfile -t PYC_FILES < <(
  find "$PROJECT_ROOT" -type f -name "*.pyc"
)

if (( ${#PYC_FILES[@]} > 0 )); then
  echo ".pyc files to be deleted:"
  echo "-------------------------"
  printf '%s\n' "${PYC_FILES[@]}"
  echo "Total: ${#PYC_FILES[@]}"
  echo

  if confirm "Delete .pyc files?"; then
    rm -f "${PYC_FILES[@]}"
    echo ".pyc files deleted ✅"
  else
    echo ".pyc cleanup skipped."
  fi
  echo
else
  echo "No .pyc files found."
  echo
fi

# -------------------------
# LOG FILES
# Includes files directly under logs/ AND any subdirectories
# -------------------------
mapfile -t LOG_FILES < <(
  find "$PROJECT_ROOT/logs" -mindepth 1 -type f 2>/dev/null || true
)

if (( ${#LOG_FILES[@]} > 0 )); then
  echo "Log files to be deleted:"
  echo "------------------------"
  printf '%s\n' "${LOG_FILES[@]}"
  echo "Total: ${#LOG_FILES[@]}"
  echo

  if confirm "Delete log files?"; then
    rm -f "${LOG_FILES[@]}"
    echo "Log files deleted ✅"
  else
    echo "Log cleanup skipped."
  fi
  echo
else
  echo "No log files found."
  echo
fi

# -------------------------
# DATA FILES
# Global option first, then per-subdir fallback
# -------------------------
DATA_DIR="$PROJECT_ROOT/data"

mapfile -t ALL_DATA_FILES < <(
  find "$DATA_DIR" -mindepth 1 -type f 2>/dev/null || true
)

if (( ${#ALL_DATA_FILES[@]} == 0 )); then
  echo "No data files found."
  echo
else
  echo "All data files that could be deleted:"
  echo "-------------------------------------"
  printf '%s\n' "${ALL_DATA_FILES[@]}"
  echo "Total: ${#ALL_DATA_FILES[@]}"
  echo

  if confirm "Delete ALL data files?"; then
    rm -f "${ALL_DATA_FILES[@]}"
    echo "All data files deleted ✅"
    echo
  else
    echo "Global data cleanup skipped. Proceeding per subdirectory…"
    echo

    DATA_SUBDIRS=(
      "endpoint_metrics"
      "evolution"
      "evolution/evolution_backups"
      "memory"
      "metrics"
      "taskcraft"
      "webwalkerqa"
      "xbench"
    )

    for subdir in "${DATA_SUBDIRS[@]}"; do
      TARGET_DIR="$DATA_DIR/$subdir"
      [[ -d "$TARGET_DIR" ]] || continue

      mapfile -t DATA_FILES < <(
        find "$TARGET_DIR" -type f 2>/dev/null || true
      )

      if (( ${#DATA_FILES[@]} == 0 )); then
        echo "No files found in data/$subdir"
        echo
        continue
      fi

      echo "Files to be deleted in data/$subdir:"
      echo "-------------------------------------"
      printf '%s\n' "${DATA_FILES[@]}"
      echo "Total: ${#DATA_FILES[@]}"
      echo

      if confirm "Delete files in data/$subdir?"; then
        rm -f "${DATA_FILES[@]}"
        echo "data/$subdir cleaned ✅"
      else
        echo "Skipped data/$subdir"
      fi
      echo
    done
  fi
fi

echo "Cleanup finished 🧹"
