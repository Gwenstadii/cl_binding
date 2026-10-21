#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p .tools .server

# Keep this configuration local to this process and project.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
uv_bin="$project_root/.tools/uv-package/bin/uv"

if [[ ! -x "$uv_bin" ]]; then
    python3 -m pip install --disable-pip-version-check --timeout 30 --retries 1 \
        --index-url "$UV_DEFAULT_INDEX" --target "$project_root/.tools/uv-package" uv
fi
if [[ ! -x .venv/bin/python ]]; then
    "$uv_bin" venv --python 3.12 --seed .venv
fi

# Install GPU wheels from the official source, then resolve the remaining stack.
"$uv_bin" pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.10.0' 'torchvision==0.25.0'
"$uv_bin" pip install --python .venv/bin/python \
    --index-url "$UV_DEFAULT_INDEX" \
    --constraint requirements/scutlab-cu128.txt \
    -e './third_party/lerobot[smolvla,training,evaluation,libero]' h5py
"$uv_bin" pip check --python .venv/bin/python
.venv/bin/python scripts/verify_scutlab.py
"$uv_bin" pip freeze --python .venv/bin/python > .server/requirements.freeze.txt
