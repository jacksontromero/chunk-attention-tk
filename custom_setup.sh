#!/bin/bash
# setup_cuda_env.sh
# Sets up conda environment with CUDA 12.4, PyTorch 2.5.1, and Python 3.10

set -e  # Exit on error

ENV_NAME="myenv"

echo "=== Setting up $ENV_NAME environment ==="

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing $ENV_NAME environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n $ENV_NAME -y
fi

# Create fresh environment with Python 3.10
echo "Creating environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Updating packages"
sudo apt-get update && sudo apt install build-essential cmake

# Install CUDA 12.4 toolkit with explicit channel specification to avoid version mixing
echo "Installing CUDA 12.4 toolkit..."
conda install \
    -c nvidia/label/cuda-12.4.1 \
    cuda-toolkit=12.4.1 \
    cuda-nvcc=12.4 \
    cuda-cudart-dev=12.4 \
    cuda-libraries-dev=12.4 \
    cuda-nvtx=12.4 \
    -c nvidia/label/cuda-12.4.1 \
    -y

# Remove conda's binutils to avoid glibc conflicts with system linker
echo "Removing conda binutils to use system linker..."
conda remove binutils_linux-64 ld_impl_linux-64 --force -y 2>/dev/null || true

# Install PyTorch 2.5.1 with CUDA 12.4 via pip
echo "Installing PyTorch 2.5.1 with CUDA 12.4..."
python3 -m pip install --upgrade pip mypy
pip install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Unset conda compiler variables to use system compilers
echo "Configuring to use system compilers..."
conda env config vars set CC="" -n $ENV_NAME
conda env config vars set CXX="" -n $ENV_NAME

# Reactivate to apply changes
conda deactivate
conda activate $ENV_NAME

# Verify installation
echo ""
echo "=== Verification ==="
echo "Python version:"
python --version

echo ""
echo "NVCC version:"
nvcc --version | grep "release"

echo ""
echo "PyTorch installation:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo ""
echo "Compilers being used:"
echo "  gcc: $(which gcc)"
echo "  g++: $(which g++)"
echo "  ld: $(which ld)"
echo "  nvcc: $(which nvcc)"

echo ""
echo "=== Setup complete! ==="
echo "To use this environment, run: conda activate $ENV_NAME"
echo ""
echo "To build your project, ensure you unset CC/CXX/LD variables and use:"
echo "  unset CC CXX LD"
echo "  cmake -S . -B build -DTORCH=\$CONDA_PREFIX/lib/python3.10/site-packages/torch/ -DUSE_CUDA=ON ..."

echo "ThunderKittens env setup"
export CUDAHOSTCXX=/usr/bin/g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
