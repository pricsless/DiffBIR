#!/bin/bash
set -e  # Exit on error

echo "=================================="
echo "DiffBIR Quick Setup for Vast.ai"
echo "=================================="

# Create a minimal environment.yml without pip dependencies
# IMPORTANT: Install to /workspace so it's saved on the volume
echo "Creating minimal conda environment..."
cat > ../environment-minimal.yml << 'EOL'
name: diffbir
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - ffmpeg
  - git
  - curl
  - wget
EOL

# Navigate to workspace root
cd ..

# Create conda environment in /workspace (so it persists on volume)
echo "Setting up conda environment in /workspace..."
conda env create -f environment-minimal.yml -p /workspace/diffbir_env -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /workspace/diffbir_env

# Verify Python version
echo "Python version:"
python --version

# Go back to DiffBIR directory
cd DiffBIR

# Install PyTorch packages with proper index URL
echo "Installing PyTorch packages..."
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install xformers
echo "Installing xformers..."
pip install xformers==0.0.25.post1

# Install remaining packages
echo "Installing remaining dependencies..."
pip install omegaconf==2.3.0 \
    accelerate==0.28.0 \
    einops==0.7.0 \
    opencv-python==4.9.0.80 \
    scipy==1.12.0 \
    ftfy==6.2.0 \
    regex==2023.12.25 \
    python-dateutil==2.9.0.post0 \
    timm==0.9.16 \
    pytorch-lightning==2.2.1 \
    tensorboard==2.16.2 \
    protobuf==4.25.3 \
    lpips==0.1.4 \
    facexlib==0.3.0 \
    gradio \
    polars==1.12.0 \
    torchsde==0.2.6 \
    bitsandbytes==0.44.1 \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    sentencepiece==0.1.99 \
    fairscale==0.4.4

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "IMPORTANT: Environment installed to /workspace/diffbir_env (saved on volume)"
echo ""
echo "To run DiffBIR, use:"
echo "  conda activate /workspace/diffbir_env"
echo "  cd /workspace/DiffBIR"
echo "  python run_gradio.py --captioner llava --share"
echo ""
echo "Or run the quick start script:"
echo "  bash run.sh"
echo ""

# Ask if user wants to start immediately
read -p "Do you want to start DiffBIR now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python run_gradio.py --captioner llava --share
fi