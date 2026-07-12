#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/existing/Phystwin-Reduction-Experiment-1"
  exit 2
fi

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(cd "$1" && pwd)"

if [[ ! -d "$TARGET/.git" ]]; then
  echo "Target is not a Git repository: $TARGET" >&2
  exit 1
fi

find "$TARGET" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
for item in "$SOURCE"/.[!.]* "$SOURCE"/..?* "$SOURCE"/*; do
  [[ -e "$item" ]] || continue
  [[ "$(basename "$item")" == ".git" ]] && continue
  cp -a "$item" "$TARGET"/
done

cd "$TARGET"
git add -A
git status

echo
echo 'Review the status, then run:'
echo 'git commit -m "Refactor Colab notebook into server-ready Python scripts"'
echo 'git push origin main'
