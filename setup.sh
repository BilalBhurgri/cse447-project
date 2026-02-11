#!/bin/bash
# Setup script for CSE447 Project

echo "Setting up CSE447 Project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install tqdm>=4.65.0

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the virtual environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To test the model, run:"
echo "    python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt"
