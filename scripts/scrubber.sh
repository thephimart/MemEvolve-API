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
# -------------------------
mapfile -t LOG_FILES < <(
  find "$PROJECT_ROOT/logs" -mindepth 2 -type f 2>/dev/null || true
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
# -------------------------
mapfile -t DATA_FILES < <(
  find "$PROJECT_ROOT/data" -mindepth 2 -type f 2>/dev/null || true
)

if (( ${#DATA_FILES[@]} > 0 )); then
  echo "Data files to be deleted:"
  echo "-------------------------"
  printf '%s\n' "${DATA_FILES[@]}"
  echo "Total: ${#DATA_FILES[@]}"
  echo

  if confirm "Delete data files?"; then
    rm -f "${DATA_FILES[@]}"
    echo "Data files deleted ✅"
  else
    echo "Data cleanup skipped."
  fi
  echo
else
  echo "No data files found."
  echo
fi

echo "Cleanup finished 🧹"
