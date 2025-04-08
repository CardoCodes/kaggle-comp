# March Madness Kaggle Competition

## Data Setup

This project uses the data from the [March Madness Kaggle Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025).

This will require you to have a kaggle account and a API key that can be found [here](https://www.kaggle.com/settings/).

Once you have the kaggle.json file, move it to the correct directory.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then you can download the data using the following command:

```bash
kaggle competitions download -c march-machine-learning-mania-2025
```

Once you have installed the kaggle dataset, you can unzip the file and move it to the correct directory.

```bash
unzip march-machine-learning-mania-2025.zip
mv *.csv ./data
```

## Setup Guides

### Windows Setup
Use the following pip install command to install all the packages needed for this project.

```bash
pip install numpy pandas matplotlib seaborn plotly xgboost scikit-learn
```

or alternatively you can install using the requirements.txt file.

```bash
pip install -r requirements.txt
```

### Linux Setup

#### Option 1: Using a Virtual Environment (Recommended)

Creating a virtual environment is the recommended approach for data science projects as it keeps dependencies isolated and avoids conflicts with system packages.

1. Install required system packages:
```bash
sudo apt install python3-full python3-venv
```

2. Create a virtual environment in your project directory:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Install the required packages:
```bash
pip install numpy pandas matplotlib seaborn plotly xgboost scikit-learn
```

or using requirements.txt:
```bash
pip install -r requirements.txt
```

5. Run Jupyter (if needed):
```bash
jupyter notebook
```

When you're done working, deactivate the virtual environment:
```bash
deactivate
```

#### Option 2: System-wide Installation

If you prefer to install packages system-wide (not recommended for most data science work):

```bash
sudo apt install python3-pandas python3-numpy python3-matplotlib python3-seaborn python3-plotly python3-xgboost python3-sklearn python3-setuptools
```

Note: Some packages like plotly might not be available in the default Ubuntu repositories or might be under different names. You might need to install some packages via pip even with this approach.

## Repository Structure

### Jupyter Notebook (.ipynb)
The Jupyter Notebook in this repository is specifically designed for Kaggle submissions. It:

- Provides an interactive environment to run predictions
- Contains visualizations and explanations of the model's performance
- Is formatted to be compatible with Kaggle's notebook environment
- Serves as both documentation and an executable pipeline for competition submissions

For Kaggle competition purposes, you should use the notebook, which imports and utilizes the functionality from the Python modules while presenting results in a format suitable for the competition.

## Data Preparation and Model Selection

### Data Preparation

The March Madness tournament data requires significant preparation to be useful for prediction. Our approach involves:

1. **Loading and merging multiple data sources** - We combine regular season results, tournament outcomes, team stats, and seeds:

```python
def load_data(self):
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in {self.data_dir}/ directory")
    
    # Load seeds, teams, and game results
    self.seeds = pd.read_csv(os.path.join(self.data_dir, 'MNCAATourneySeeds.csv'))
    self.teams = pd.read_csv(os.path.join(self.data_dir, 'MTeams.csv'))
    self.game_results = pd.read_csv(os.path.join(self.data_dir, 'MRegularSeasonCompactResults.csv'))
```

2. **Feature engineering** - We create differential features between teams to capture competitive advantages:

```python
# Calculate differences between team statistics
features_df['FGMDiff'] = features_df['Team1_FGM'] - features_df['Team2_FGM']
features_df['FGADiff'] = features_df['Team1_FGA'] - features_df['Team2_FGA']
features_df['FGPctDiff'] = features_df['Team1_FGPct'] - features_df['Team2_FGPct']
features_df['FGM3Diff'] = features_df['Team1_FGM3'] - features_df['Team2_FGM3']
features_df['FGA3Diff'] = features_df['Team1_FGA3'] - features_df['Team2_FGA3']
```

3. **Data normalization** - Handling missing values and ensuring data consistency:

```python
# Handle missing values
features_df = features_df.fillna(0)
```

### Model Selection

We chose **XGBoost** for our prediction model for several key reasons:

1. **Performance on binary classification problems** - XGBoost consistently performs well on win/loss prediction:

```python
def create_model(self):
    # Create an XGBoost model for win probability
    self.model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',  # For probability outputs
        random_state=42
    )
```

2. **Feature importance analysis** - XGBoost provides valuable insights into what drives predictions:

```python
# Feature importance analysis shows shooting efficiency is critical
"""
===== FEATURE IMPORTANCE =====
         Feature  Importance
40     FGPctDiff    0.404768  # Field Goal % difference is most important
50       AstDiff    0.161311  # Assist difference is second most important  
38       FGMDiff    0.062710
44       FTMDiff    0.061543
45       FTADiff    0.055830
"""
```