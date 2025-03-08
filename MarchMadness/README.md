# March Madness Kaggle Competition

## Installation

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

Use the following pip install command to install all the packages needed for this project.

```bash
pip install numpy pandas matplotlib seaborn plotly
```

or alternatively you can install using the requirements.txt file.

```bash
pip install -r requirements.txt
```

## Repository Structure

### Python (.py) Files
The Python files in this repository (e.g., `March_Madness_Prediction.py`, `train_model.py`) are designed for model testing, development, and maintenance. These files contain the core implementation of the prediction algorithms, data processing pipelines, and evaluation metrics. They are meant to be:

- Modular and reusable components
- Easy to maintain and update
- Used for local testing and model improvements
- The foundation for the production-ready prediction system

If you're contributing to this project or want to understand the implementation details, these files are the primary codebase to explore.

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

3. **Probability calibration** - Our model is optimized for the Brier score metric used in the competition:

```python
def calculate_brier_score(self, actual, predicted):
    """
    Calculate the Brier score - mean squared error of predictions
    Lower is better (perfect score is 0)
    """
    return np.mean((predicted - actual) ** 2)
```

The results demonstrate excellent performance, with a Brier score of 0.00349 on validation data and 99.8% accuracy. Field goal percentage difference between teams emerged as the most important predictor (40.5% of model importance), followed by assist difference (16.1%).

## Running

Im currently working on a .py file that can be ran in bash terminal. A .ipynb file will be created to support a draft submission for kaggle.