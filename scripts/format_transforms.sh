#!/bin/bash

# === Parse Arguments ===
SAVE_ROOT="${1:-$(pwd)}"       # Default to current directory if not provided
DATA_ROOT="$2"

if [[ -z "$DATA_ROOT" ]]; then
    echo "Usage: $0 [save_root (optional)] <data_root>"
    exit 1
fi

# === Run Python Code ===
python3 - <<EOF
import json
import os
import sys

save_root = "${SAVE_ROOT}"
data_root = "${DATA_ROOT}"

with open(os.path.join(data_root, "transforms.json")) as handle:
    transform = json.load(handle)

transform["datadir"] = data_root
for i in range(len(transform['frames'])):
    transform['frames'][i]['file_path'] = f"MakoSpectrometer-t{i:04d}.img.hdr"

os.makedirs(save_root, exist_ok=True)

with open(os.path.join(save_root, "transforms.json"), 'w') as handle:
    json.dump(transform, handle, indent=4)

print(f"✅ Saved updated transforms.json to: {save_root}")
EOF
