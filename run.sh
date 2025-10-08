#!/bin/bash
# Quick run script - use this after setup is complete

# Activate environment (from /data - the persistent volume)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data/diffbir_env

# Navigate to DiffBIR directory if not already there
if [ ! -f "run_gradio.py" ]; then
    cd /data/DiffBIR
fi

# Run DiffBIR
python run_gradio.py --captioner llava --share