#!/usr/bin/env bash
# Symlink mmplatform plugin into the default FiftyOne plugins directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="${SCRIPT_DIR}/mmplatform-cvat"
PLUGINS_DIR="$(python3 -c "import fiftyone as fo; print(fo.config.plugins_dir)")"

mkdir -p "${PLUGINS_DIR}"
TARGET="${PLUGINS_DIR}/mmplatform-cvat"

if [[ -e "${TARGET}" && ! -L "${TARGET}" ]]; then
  echo "Already exists and is not a symlink: ${TARGET}" >&2
  exit 1
fi

ln -sfn "${PLUGIN_SRC}" "${TARGET}"
echo "Installed: ${TARGET} -> ${PLUGIN_SRC}"
echo "Restart FiftyOne App to load the plugin."
