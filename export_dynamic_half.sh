#!/bin/bash

set -euxo pipefail

python export.py \
    --dynamic \
    --optimization-profile trt_opt_profile.yaml \
    --workspace 12 --device 1 \
    --weights dolphin_single_det_base_single_0419.pt \
    --include engine \
    --half \
    --output-prefix dolphin_single_det_base_single_0419_all_dynamic_half

