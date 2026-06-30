#!/bin/bash

set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/home/hanfeig/conda-envs/mei}
PYTHON_BIN=${ENV_ROOT}/bin/python
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected an existing conda environment at ${ENV_ROOT}." >&2
  echo "Create it first with:" >&2
  echo "/opt/anaconda3-2023.03/bin/conda create -p ${ENV_ROOT} python=3.11 -y" >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  torch==2.7.1 \
  torchvision==0.22.1
"${PYTHON_BIN}" -m pip install \
  numpy==2.3.3 \
  Pillow==11.3.0 \
  scipy==1.16.0 \
  pandas==2.3.1 \
  open_clip_torch==3.3.0 \
  huggingface_hub

"${PYTHON_BIN}" -c \
  "import numpy, open_clip, pandas, PIL, scipy, torch, torchvision"

echo "Environment is ready: ${ENV_ROOT}"
