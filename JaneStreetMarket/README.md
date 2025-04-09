# Create a virtual environment
python3 -m venv ../.venv

# Activate the virtual environment
source ../.venv/bin/activate

# Install dependencies
pip install pandas polars kaggle-evaluation

# Verify installation
python -c "import pandas; import polars; import kaggle_evaluation; print('All dependencies installed successfully')"
