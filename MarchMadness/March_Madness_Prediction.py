import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import xgboost as xgb
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score


class MarchMadnessPredictor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = None
        self.teams = None
        self.seeds  = None
        self.submission = None
        self.all_compact_results = None
        self.all_detailed_results = None
        self.tourney_compact_results = None
        self.tourney_detailed_results = None
        self.model = None
        self.calibration_model = None
        
        # Load data when the predictor is initialized
        self.load_data()

    def load_data(self):
        
        """
        Set up a data dictionary that will store the data for each file. e.g.
        self.data = {
            'teams': [DataFrame with teams data],
            'games': [DataFrame with games data],
            'players': [DataFrame with players data]
        }
        """

        # Make sure the data_dir ends with a slash
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
            
        files = glob.glob(self.data_dir + '*.csv')
        print(f"Found {len(files)} CSV files in {self.data_dir}")
        
        self.data = {}
        for file in files:
            # Extract filename without extension
            filename = file.split('/')[-1].split('\\')[-1].split('.')[0]
            try:
                self.data[filename] = pd.read_csv(file, encoding='latin-1')
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

        # Check if SampleSubmissionStage1 exists
        if 'SampleSubmissionStage1' in self.data:
            self.submission = self.data['SampleSubmissionStage1']
        else:
            print("Warning: SampleSubmissionStage1.csv not found. Submission data will be None.")
            self.submission = None

        teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
        teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
        #print(self.teams.head())

        season_compact_results = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']]).assign(ST='S')
        season_detailed_results = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']]).assign(ST='S')
        tourney_compact_results = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']]).assign(ST='T')
        tourney_detailed_results = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']]).assign(ST='T')

        # Extract numeric seed value from seed string
        seeds = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        seeds['SeedValue'] = seeds['Seed'].str.extract(r'(\d+)').astype(int)
        self.seeds = seeds
        print(self.seeds)

        """
        Load the game data with additional derived features.
        Combines regualr season and tournament results
        """

        # Combine all game results
        all_compact_results = pd.concat([season_compact_results, tourney_compact_results])
        all_detailed_results = pd.concat([season_detailed_results, tourney_detailed_results])

        # Add derived features to compact results
        all_compact_results['ScoreDiff'] = all_compact_results['WScore'] - all_compact_results['LScore']
        all_compact_results['HomeAdvantage'] = all_compact_results['WLoc'].map({'H': 1, 'N': 0, 'A': -1})
        
        # Add derived features to detaifled results
        all_detailed_results['ScoreDiff'] = all_detailed_results['WScore'] - all_detailed_results['LScore']
        all_detailed_results['HomeAdvantage'] = all_detailed_results['WLoc'].map({'H': 1, 'N': 0, 'A': -1})

         # Calculate shooting percentages (handling division by zero)
        all_detailed_results['WFGPct'] = np.where(all_detailed_results['WFGA'] > 0, 
                                                all_detailed_results['WFGM'] / all_detailed_results['WFGA'], 0)
        all_detailed_results['WFG3Pct'] = np.where(all_detailed_results['WFGA3'] > 0, 
                                                all_detailed_results['WFGM3'] / all_detailed_results['WFGA3'], 0)
        all_detailed_results['WFTPct'] = np.where(all_detailed_results['WFTA'] > 0, 
                                                all_detailed_results['WFTM'] / all_detailed_results['WFTA'], 0)
        all_detailed_results['LFGPct'] = np.where(all_detailed_results['LFGA'] > 0, 
                                                all_detailed_results['LFGM'] / all_detailed_results['LFGA'], 0)
        all_detailed_results['LFG3Pct'] = np.where(all_detailed_results['LFGA3'] > 0, 
                                                all_detailed_results['LFGM3'] / all_detailed_results['LFGA3'], 0)
        all_detailed_results['LFTPct'] = np.where(all_detailed_results['LFTA'] > 0, 
                                                all_detailed_results['LFTM'] / all_detailed_results['LFTA'], 0)
        
        # Add statistical differences
        all_detailed_results['ReboundDiff'] = (all_detailed_results['WOR'] + all_detailed_results['WDR']) - \
                                            (all_detailed_results['LOR'] + all_detailed_results['LDR'])
        all_detailed_results['AssistDiff'] = all_detailed_results['WAst'] - all_detailed_results['LAst']
        all_detailed_results['TurnoverDiff'] = all_detailed_results['WTO'] - all_detailed_results['LTO']
        all_detailed_results['StealDiff'] = all_detailed_results['WStl'] - all_detailed_results['LStl']
        all_detailed_results['BlockDiff'] = all_detailed_results['WBlk'] - all_detailed_results['LBlk']
        all_detailed_results['FoulDiff'] = all_detailed_results['WPF'] - all_detailed_results['LPF']

        # Add seed information to tournament games
        tourney_compact = all_compact_results[all_compact_results['ST'] == 'T'].copy()
        tourney_detailed = all_detailed_results[all_detailed_results['ST'] == 'T'].copy()

        # Add winner seeds
        tourney_compact = pd.merge(
            tourney_compact,
            seeds[['Season', 'TeamID', 'SeedValue']],
            how='left',
            left_on=['Season', 'WTeamID'],
            right_on=['Season', 'TeamID']
        )
        tourney_compact.rename(columns={'SeedValue': 'WSeedValue'}, inplace=True)
        tourney_compact.drop('TeamID', axis=1, inplace=True)
        
        tourney_detailed = pd.merge(
            tourney_detailed,
            seeds[['Season', 'TeamID', 'SeedValue']],
            how='left',
            left_on=['Season', 'WTeamID'],
            right_on=['Season', 'TeamID']
        )
        tourney_detailed.rename(columns={'SeedValue': 'WSeedValue'}, inplace=True)
        tourney_detailed.drop('TeamID', axis=1, inplace=True)

        # Add loser seeds
        tourney_compact = pd.merge(
            tourney_compact,
            seeds[['Season', 'TeamID', 'SeedValue']],
            how='left',
            left_on=['Season', 'LTeamID'],
            right_on=['Season', 'TeamID']
        )
        tourney_compact.rename(columns={'SeedValue': 'LSeedValue'}, inplace=True)
        tourney_compact.drop('TeamID', axis=1, inplace=True)
        
        tourney_detailed = pd.merge(
            tourney_detailed,
            seeds[['Season', 'TeamID', 'SeedValue']],
            how='left',
            left_on=['Season', 'LTeamID'],
            right_on=['Season', 'TeamID']
        )
        tourney_detailed.rename(columns={'SeedValue': 'LSeedValue'}, inplace=True)
        tourney_detailed.drop('TeamID', axis=1, inplace=True)

         # Calculate seed difference (lower is better in seeding, so LSeed - WSeed is positive if favorite won)
        tourney_compact['SeedDiff'] = tourney_compact['LSeedValue'] - tourney_compact['WSeedValue']
        tourney_detailed['SeedDiff'] = tourney_detailed['LSeedValue'] - tourney_detailed['WSeedValue']

        # Store all processed data
        self.all_compact_results = all_compact_results
        self.all_detailed_results = all_detailed_results
        self.tourney_compact_results = tourney_compact
        self.tourney_detailed_results = tourney_detailed

        print("All Compact Resullts: \n", self.all_compact_results.head())
        print(self.all_detailed_results.head())
        print(self.tourney_compact_results.head())
        print(self.tourney_detailed_results.head())

    def create_model(self):
        """
        Creates XGBoost models for prediction and calibration.
        """
        # Main prediction model
        self.model = xgb.XGBRegressor(
            n_estimators=500,         # Number of boosting rounds
            learning_rate=0.05,       # Smaller learning rate for better generalization
            max_depth=6,              # Control model complexity
            min_child_weight=3,       # Helps prevent overfitting
            subsample=0.8,            # Use 80% of data for each tree
            colsample_bytree=0.8,     # Use 80% of features for each tree
            objective='binary:logistic',  # Binary classification with probability output
            random_state=42,
            n_jobs=-1                 # Use all CPU cores
        )
        
        # Calibration model to fine-tune probabilities
        self.calibration_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        
    def prepare_features(self, min_season=None, max_season=None, use_detailed=True):
        """
        Prepare features for model training from the game results data.
        
        Parameters:
        -----------
        min_season : int, optional
            Minimum season to include in training data
        max_season : int, optional
            Maximum season to include in training data
        use_detailed : bool, default=True
            Whether to use detailed features or just compact features
            
        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable (1 for team1 win, 0 for team2 win)
        """
        print("Preparing features for model training...")
        
        # Check if required data exists
        required_files = [
            'MRegularSeasonDetailedResults', 
            'WRegularSeasonDetailedResults',
            'MNCAATourneyDetailedResults',
            'WNCAATourneyDetailedResults'
        ]
        
        missing_files = [f for f in required_files if f not in self.data or self.data[f] is None]
        if missing_files:
            print(f"Error: Missing required data files: {missing_files}")
            # List available files
            print(f"Available files: {list(self.data.keys())}")
            raise ValueError(f"Missing required data files: {missing_files}")
        
        # Filter seasons if specified
        all_detailed_results = pd.concat([
            self.data['MRegularSeasonDetailedResults'], 
            self.data['WRegularSeasonDetailedResults'],
            self.data['MNCAATourneyDetailedResults'],
            self.data['WNCAATourneyDetailedResults']
        ]).assign(ST='All')
        
        # Filter by season if specified
        if min_season is not None:
            all_detailed_results = all_detailed_results[all_detailed_results['Season'] >= min_season]
        if max_season is not None:
            all_detailed_results = all_detailed_results[all_detailed_results['Season'] <= max_season]
            
        # Create pairs of teams for each game (both directions)
        # For each game, we create two rows: (team1, team2) and (team2, team1)
        # with corresponding targets 1 and 0
        game_pairs = []
        
        for _, game in all_detailed_results.iterrows():
            # Features for team1 (winner) vs team2 (loser)
            features1 = {
                'Season': game['Season'],
                'DayNum': game['DayNum'],
                'Team1': game['WTeamID'],
                'Team2': game['LTeamID'],
                'Target': 1  # Team1 won
            }
            
            # Features for team2 (loser) vs team1 (winner)
            features2 = {
                'Season': game['Season'],
                'DayNum': game['DayNum'],
                'Team1': game['LTeamID'],
                'Team2': game['WTeamID'],
                'Target': 0  # Team1 lost
            }
            
            # Add location feature if available
            if 'WLoc' in game:
                if game['WLoc'] == 'H':
                    features1['Team1Home'] = 1
                    features2['Team1Home'] = 0
                elif game['WLoc'] == 'A':
                    features1['Team1Home'] = 0
                    features2['Team1Home'] = 1
                else:  # Neutral
                    features1['Team1Home'] = 0.5
                    features2['Team1Home'] = 0.5
            
            # Add detailed stats if available and requested
            if use_detailed and all(col in game for col in ['WFGM', 'LFGM']):
                # Team1 offensive stats when it was the winner
                features1.update({
                    'Team1_FGM': game['WFGM'],
                    'Team1_FGA': game['WFGA'],
                    'Team1_FGM3': game['WFGM3'],
                    'Team1_FGA3': game['WFGA3'],
                    'Team1_FTM': game['WFTM'],
                    'Team1_FTA': game['WFTA'],
                    'Team1_OR': game['WOR'],
                    'Team1_DR': game['WDR'],
                    'Team1_Ast': game['WAst'],
                    'Team1_TO': game['WTO'],
                    'Team1_Stl': game['WStl'],
                    'Team1_Blk': game['WBlk'],
                    'Team1_PF': game['WPF'],
                    
                    # Team2 offensive stats when it was the loser
                    'Team2_FGM': game['LFGM'],
                    'Team2_FGA': game['LFGA'],
                    'Team2_FGM3': game['LFGM3'],
                    'Team2_FGA3': game['LFGA3'],
                    'Team2_FTM': game['LFTM'],
                    'Team2_FTA': game['LFTA'],
                    'Team2_OR': game['LOR'],
                    'Team2_DR': game['LDR'],
                    'Team2_Ast': game['LAst'],
                    'Team2_TO': game['LTO'],
                    'Team2_Stl': game['LStl'],
                    'Team2_Blk': game['LBlk'],
                    'Team2_PF': game['LPF'],
                })
                
                # Team1 offensive stats when it was the loser
                features2.update({
                    'Team1_FGM': game['LFGM'],
                    'Team1_FGA': game['LFGA'],
                    'Team1_FGM3': game['LFGM3'],
                    'Team1_FGA3': game['LFGA3'],
                    'Team1_FTM': game['LFTM'],
                    'Team1_FTA': game['LFTA'],
                    'Team1_OR': game['LOR'],
                    'Team1_DR': game['LDR'],
                    'Team1_Ast': game['LAst'],
                    'Team1_TO': game['LTO'],
                    'Team1_Stl': game['LStl'],
                    'Team1_Blk': game['LBlk'],
                    'Team1_PF': game['LPF'],
                    
                    # Team2 offensive stats when it was the winner
                    'Team2_FGM': game['WFGM'],
                    'Team2_FGA': game['WFGA'],
                    'Team2_FGM3': game['WFGM3'],
                    'Team2_FGA3': game['WFGA3'],
                    'Team2_FTM': game['WFTM'],
                    'Team2_FTA': game['WFTA'],
                    'Team2_OR': game['WOR'],
                    'Team2_DR': game['WDR'],
                    'Team2_Ast': game['WAst'],
                    'Team2_TO': game['WTO'],
                    'Team2_Stl': game['WStl'],
                    'Team2_Blk': game['WBlk'],
                    'Team2_PF': game['WPF'],
                })
            
            game_pairs.append(features1)
            game_pairs.append(features2)
        
        # Convert to DataFrame
        games_df = pd.DataFrame(game_pairs)
        
        # Add team seed features if available
        if hasattr(self, 'seeds') and self.seeds is not None:
            # Join seed info for both teams
            games_df = pd.merge(
                games_df,
                self.seeds[['Season', 'TeamID', 'SeedValue']],
                left_on=['Season', 'Team1'],
                right_on=['Season', 'TeamID'],
                how='left'
            ).rename(columns={'SeedValue': 'Team1Seed'}).drop('TeamID', axis=1)
            
            games_df = pd.merge(
                games_df,
                self.seeds[['Season', 'TeamID', 'SeedValue']],
                left_on=['Season', 'Team2'],
                right_on=['Season', 'TeamID'],
                how='left'
            ).rename(columns={'SeedValue': 'Team2Seed'}).drop('TeamID', axis=1)
            
            # Create seed difference feature
            games_df['SeedDiff'] = games_df['Team1Seed'] - games_df['Team2Seed']
        
        # Add derived features
        if use_detailed:
            # Calculate shooting percentages
            for team in [1, 2]:
                prefix = f'Team{team}_'
                # Field goal percentage
                games_df[f'{prefix}FGPct'] = np.where(
                    games_df[f'{prefix}FGA'] > 0,
                    games_df[f'{prefix}FGM'] / games_df[f'{prefix}FGA'],
                    0
                )
                # 3-point percentage
                games_df[f'{prefix}FG3Pct'] = np.where(
                    games_df[f'{prefix}FGA3'] > 0,
                    games_df[f'{prefix}FGM3'] / games_df[f'{prefix}FGA3'],
                    0
                )
                # Free throw percentage
                games_df[f'{prefix}FTPct'] = np.where(
                    games_df[f'{prefix}FTA'] > 0,
                    games_df[f'{prefix}FTM'] / games_df[f'{prefix}FTA'],
                    0
                )
                # Total rebounds
                games_df[f'{prefix}TotalReb'] = games_df[f'{prefix}OR'] + games_df[f'{prefix}DR']
            
            # Calculate differentials between teams
            stat_pairs = [
                ('FGM', 'Field goals made'),
                ('FGA', 'Field goals attempted'),
                ('FGPct', 'Field goal percentage'),
                ('FGM3', '3-pointers made'),
                ('FGA3', '3-pointers attempted'),
                ('FG3Pct', '3-point percentage'),
                ('FTM', 'Free throws made'),
                ('FTA', 'Free throws attempted'),
                ('FTPct', 'Free throw percentage'),
                ('OR', 'Offensive rebounds'),
                ('DR', 'Defensive rebounds'),
                ('TotalReb', 'Total rebounds'),
                ('Ast', 'Assists'),
                ('TO', 'Turnovers'),
                ('Stl', 'Steals'),
                ('Blk', 'Blocks'),
                ('PF', 'Personal fouls')
            ]
            
            for stat, _ in stat_pairs:
                games_df[f'{stat}Diff'] = games_df[f'Team1_{stat}'] - games_df[f'Team2_{stat}']
        
        # Drop columns not needed for modeling
        drop_cols = ['Team1', 'Team2']  # We'll use team statistics instead of IDs
        
        # Keep Season and DayNum for evaluation and predictions
        X = games_df.drop(['Target'] + drop_cols, axis=1)
        y = games_df['Target']
        
        print(f"Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y
        
    def train_model(self, min_season=None, max_season=None, eval_season=None, use_detailed=True):
        """
        Train the XGBoost model using prepared features.
        
        Parameters:
        -----------
        min_season : int, optional
            Minimum season to include in training data
        max_season : int, optional
            Maximum season to include in training data
        eval_season : int, optional
            Season to use for evaluation (will be excluded from training)
        use_detailed : bool, default=True
            Whether to use detailed features
            
        Returns:
        --------
        model : XGBRegressor
            Trained XGBoost model
        eval_results : dict
            Evaluation results if eval_season is provided
        """
        print("Training XGBoost model...")
        
        # Prepare features
        X, y = self.prepare_features(min_season, max_season, use_detailed)
        
        # Split data into training and validation sets
        if eval_season is not None:
            # Use specific season for evaluation
            is_eval = X['Season'] == eval_season
            X_train, X_val = X[~is_eval].copy(), X[is_eval].copy()
            y_train, y_val = y[~is_eval], y[is_eval]
            
            # Remove Season from features
            X_train = X_train.drop(['Season', 'DayNum'], axis=1)
            X_val = X_val.drop(['Season', 'DayNum'], axis=1)
            
            print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        else:
            # Use random split
            X_train, X_val, y_train, y_val = train_test_split(
                X.drop(['Season', 'DayNum'], axis=1),
                y,
                test_size=0.2,
                random_state=42
            )
            print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        # Create model if not already done
        if not hasattr(self, 'model') or self.model is None:
            self.create_model()
        
        # Train the model without eval_metric parameter
        print("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        # Make predictions
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        # Calculate metrics
        train_brier = self.calculate_brier_score(y_train, train_preds)
        val_brier = self.calculate_brier_score(y_val, val_preds)
        
        # Calculate additional metrics
        train_log_loss = log_loss(y_train, np.clip(train_preds, 0.001, 0.999))
        val_log_loss = log_loss(y_val, np.clip(val_preds, 0.001, 0.999))
        
        train_accuracy = accuracy_score(y_train, np.round(train_preds))
        val_accuracy = accuracy_score(y_val, np.round(val_preds))
        
        # Store and print results
        results = {
            'train_brier': train_brier,
            'val_brier': val_brier,
            'train_log_loss': train_log_loss,
            'val_log_loss': val_log_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'feature_importance': self.model.feature_importances_,
            'feature_names': X_train.columns.tolist()
        }
        
        self.print_model_results(results)
        
        # Return trained model and results
        return self.model, results
    
    def calculate_brier_score(self, actual, predicted):
        """
        Calculate the Brier score between actual outcomes and predicted probabilities.
        
        Brier score = 1/N * sum((p_i - o_i)^2)
        where p_i is the predicted probability and o_i is the actual outcome (0 or 1)
        
        Parameters:
        -----------
        actual : array-like
            Actual outcomes (0 or 1)
        predicted : array-like
            Predicted probabilities
            
        Returns:
        --------
        float
            Brier score (lower is better)
        """
        # Ensure predictions are probabilities (between 0 and 1)
        predicted = np.clip(predicted, 0, 1)
        return np.mean((predicted - actual) ** 2)
    
    def print_model_results(self, results):
        """
        Print model evaluation results.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing evaluation metrics
        """
        print("\n===== MODEL RESULTS =====")
        print(f"Brier Score (Train): {results['train_brier']:.5f}")
        print(f"Brier Score (Validation): {results['val_brier']:.5f}")
        print(f"Log Loss (Train): {results['train_log_loss']:.5f}")
        print(f"Log Loss (Validation): {results['val_log_loss']:.5f}")
        print(f"Accuracy (Train): {results['train_accuracy']:.5f}")
        print(f"Accuracy (Validation): {results['val_accuracy']:.5f}")
        
        # Print feature importance
        print("\n===== FEATURE IMPORTANCE =====")
        importance_df = pd.DataFrame({
            'Feature': results['feature_names'],
            'Importance': results['feature_importance']
        }).sort_values('Importance', ascending=False)
        
        # Print top 15 features
        print(importance_df.head(15))
        
    def display_win_distribution(self, min_season=None, max_season=None, title=None):
        """
        Displays a heatmap of win distribution by seed matchup.
        
        Parameters:
        -----------
        min_season : int, optional
            Minimum season to include in the analysis
        max_season : int, optional
            Maximum season to include in the analysis
        title : str, optional
            Custom title for the plot
        
        Returns:
        --------
        None, displays the plot
        """
        # Filter tournament data by season if specified
        df = self.tourney_compact_results.copy()
        if min_season is not None:
            df = df[df['Season'] >= min_season]
        if max_season is not None:
            df = df[df['Season'] <= max_season]
        
        # Create a crosstab (matrix) of winner seed vs loser seed
        win_matrix = pd.crosstab(
            index=df['LSeedValue'],  # Y-axis: losing seeds
            columns=df['WSeedValue'],  # X-axis: winning seeds
            values=df['Season'],
            aggfunc='count'
        ).fillna(0)
        
        # Ensure all seeds from 1-16 are represented
        all_seeds = list(range(1, 17))
        for seed in all_seeds:
            if seed not in win_matrix.index:
                win_matrix.loc[seed] = 0
            if seed not in win_matrix.columns:
                win_matrix[seed] = 0
        
        # Sort the indices to ensure they're in order from 1-16
        win_matrix = win_matrix.reindex(index=all_seeds, columns=all_seeds)
        
        # Create a matrix to store text for each cell (number of wins + percentage)
        text_matrix = []
        for i, row in enumerate(win_matrix.values):
            text_row = []
            for j, val in enumerate(row):
                if val > 0:
                    # Calculate winning percentage when this seed matchup occurs
                    winning_seed = win_matrix.columns[j]
                    losing_seed = win_matrix.index[i]
                    # Look for the inverse matchup (same seeds but opposite outcome)
                    inverse_val = win_matrix.loc[winning_seed, losing_seed] if winning_seed in win_matrix.index and losing_seed in win_matrix.columns else 0
                    total_matchups = val + inverse_val
                    win_pct = val / total_matchups if total_matchups > 0 else 0
                    text_row.append(f"{int(val)}<br>({win_pct:.0%})")
                else:
                    text_row.append("")
            text_matrix.append(text_row)
        
        # Create the heatmap
        fig = sp.make_subplots(
            rows=1, cols=1,
            subplot_titles=[title or f"March Madness Win Distribution by Seed ({min_season or 'All'}-{max_season or 'Present'})"]
        )
            
        # Add the heatmap
        heatmap = go.Heatmap(
            z=win_matrix.values,
            x=win_matrix.columns,
            y=win_matrix.index,
            colorscale='Viridis',  # You can also try: 'Blues', 'YlOrRd', 'Plasma'
            showscale=True,
            colorbar=dict(title='Count'),
            text=text_matrix,  # Use our custom text matrix
            hoverinfo='text',
            hoverongaps=False,
            texttemplate="%{text}",  # Display our custom text
            textfont=dict(
                color='rgba(255,255,255,0.9)',  # White with slight transparency
                size=9,           # Slightly smaller to fit percentage
                family='Arial, sans-serif',
                weight='bold'     # Make text bold for better readability
            )
        )
        
        fig.add_trace(heatmap)
        
        # Update layout
        fig.update_layout(
            height=800,           # Increase height for better visibility
            width=900,            # Increase width for better visibility
            xaxis=dict(
                title=dict(
                    text='Winning Seed',
                    font=dict(size=14)
                ),
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis=dict(
                title=dict(
                    text='Losing Seed',
                    font=dict(size=14)
                ),
                tickmode='linear',
                tick0=1,
                dtick=1,
                autorange='reversed'  # Reverse y-axis to have 1 at the top
            ),
            title=dict(
                text=title or f"March Madness Win Distribution by Seed ({min_season or 'All'}-{max_season or 'Present'})",
                x=0.5,            # Center the title
                font=dict(size=18)
            )
        )
        
        # Add a diagonal line to indicate "expected" outcomes (lower seed beats higher seed)
        for i in range(1, 17):
            for j in range(1, 17):
                if i == j:
                    continue
                if i < j:  # "Expected" outcome: lower seed (i) beats higher seed (j)
                    # No need to highlight as this is expected
                    pass
                else:  # "Upset": higher seed (i) loses to lower seed (j)
                    if win_matrix.loc[i, j] > 0:
                        # Add a rectangle around upset cells
                        fig.add_shape(
                            type="rect",
                            x0=j-0.5, x1=j+0.5,
                            y0=i-0.5, y1=i+0.5,
                            line=dict(color="red", width=1),
                            fillcolor="rgba(0,0,0,0)"
                        )
        
        # Show the plot
        fig.show()
        
        # Calculate and display some statistics
        total_games = win_matrix.sum().sum()
        
        # Fix the calculation of expected wins to avoid out-of-bounds errors
        expected_wins = 0
        for i in range(1, 17):
            if i in win_matrix.columns:  # Check if the column exists
                # Get rows with index >= i
                higher_seeds = [idx for idx in win_matrix.index if idx >= i]
                if higher_seeds:  # If there are higher seeds
                    expected_wins += win_matrix.loc[higher_seeds, i].sum()
        
        upset_wins = total_games - expected_wins
        
        print(f"Total Tournament Games: {int(total_games)}")
        print(f"Expected Outcomes (lower seed beats higher seed): {int(expected_wins)} ({expected_wins/total_games:.1%})")
        print(f"Upset Outcomes (higher seed beats lower seed): {int(upset_wins)} ({upset_wins/total_games:.1%})")
        
        # Show the most common upsets
        upset_matrix = win_matrix.copy()
        for i in range(1, 17):
            for j in range(1, 17):
                if i <= j:  # Not an upset
                    upset_matrix.iloc[i-1, j-1] = 0
        
        if upset_matrix.sum().sum() > 0:
            top_upsets = []
            for i in range(1, 17):
                for j in range(1, 17):
                    if i > j and upset_matrix.iloc[i-1, j-1] > 0:
                        top_upsets.append((j, i, int(upset_matrix.iloc[i-1, j-1])))
            
            top_upsets.sort(key=lambda x: x[2], reverse=True)
            
            print("\nTop 5 Most Common Upsets:")
            for w_seed, l_seed, count in top_upsets[:5]:
                print(f"Seed #{w_seed} beating Seed #{l_seed}: {count} times")

    def display_score_distribution(self, use_plotly=True, min_season=None, max_season=None, game_type='all'):
        """
        Displays the distribution of scores for winning and losing teams.
        
        Parameters:
        -----------
        use_plotly : bool, optional (default=True)
            If True, uses plotly for interactive visualization. If False, uses seaborn.
        min_season : int, optional
            Minimum season to include in the analysis
        max_season : int, optional
            Maximum season to include in the analysis
        game_type : str, optional (default='all')
            Type of games to include: 'all', 'regular', or 'tournament'
        
        Returns:
        --------
        None, displays the plot
        """
        # Filter data based on parameters
        if game_type == 'regular':
            df = self.all_detailed_results[self.all_detailed_results['ST'] == 'S']
        elif game_type == 'tournament':
            df = self.all_detailed_results[self.all_detailed_results['ST'] == 'T']
        else:
            df = self.all_detailed_results

        if min_season is not None:
            df = df[df['Season'] >= min_season]
        if max_season is not None:
            df = df[df['Season'] <= max_season]

        if use_plotly:
            # Create figure with secondary y-axis
            fig = go.Figure()

            # Add traces for winning and losing scores
            fig.add_trace(go.Histogram(
                x=df['WScore'],
                name='Winning Score',
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            ))

            fig.add_trace(go.Histogram(
                x=df['LScore'],
                name='Losing Score',
                nbinsx=30,
                marker_color='red',
                opacity=0.7
            ))

            # Update layout
            title_text = "Score Distribution of Winning & Losing Teams"
            if min_season or max_season:
                title_text += f" ({min_season or 'All'}-{max_season or 'Present'})"
            if game_type != 'all':
                title_text += f" - {game_type.title()} Games"

            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis_title="Score",
                yaxis_title="Frequency",
                barmode='overlay',
                height=600,
                width=1000,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            # Add mean lines
            w_mean = df['WScore'].mean()
            l_mean = df['LScore'].mean()

            fig.add_vline(x=w_mean, line_dash="dash", line_color="blue",
                         annotation_text=f"Win Mean: {w_mean:.1f}")
            fig.add_vline(x=l_mean, line_dash="dash", line_color="red",
                         annotation_text=f"Loss Mean: {l_mean:.1f}")

            fig.show()

            # Create a statistics table using plotly
            stats_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Statistic', 'Winning Teams', 'Losing Teams'],
                    fill_color='royalblue',
                    align='center',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[
                        ['Mean Score', 'Median Score', 'Standard Deviation', 'Minimum Score', 'Maximum Score'],
                        [
                            f"{w_mean:.1f}",
                            f"{df['WScore'].median():.1f}",
                            f"{df['WScore'].std():.1f}",
                            f"{df['WScore'].min():.0f}",
                            f"{df['WScore'].max():.0f}"
                        ],
                        [
                            f"{l_mean:.1f}",
                            f"{df['LScore'].median():.1f}",
                            f"{df['LScore'].std():.1f}",
                            f"{df['LScore'].min():.0f}",
                            f"{df['LScore'].max():.0f}"
                        ]
                    ],
                    align='center',
                    fill_color=[['lightgrey', 'white'] * 3],
                    font=dict(color='black', size=11)
                )
            )])

            # Update table layout
            stats_fig.update_layout(
                title=dict(
                    text="Score Statistics Summary",
                    x=0.5,
                    font=dict(size=16)
                ),
                width=800,
                height=300,
                margin=dict(t=50, b=20)
            )

            # Add a row for margin of victory statistics
            margin_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Margin of Victory Statistics', 'Value'],
                    fill_color='royalblue',
                    align='center',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[
                        ['Average Margin', 'Median Margin', 'Maximum Margin', 'Minimum Margin', 'Std Dev of Margin'],
                        [
                            f"{(w_mean - l_mean):.1f}",
                            f"{(df['WScore'] - df['LScore']).median():.1f}",
                            f"{(df['WScore'] - df['LScore']).max():.1f}",
                            f"{(df['WScore'] - df['LScore']).min():.1f}",
                            f"{(df['WScore'] - df['LScore']).std():.1f}"
                        ]
                    ],
                    align='center',
                    fill_color=[['lightgrey', 'white'] * 3],
                    font=dict(color='black', size=11)
                )
            )])

            # Update margin table layout
            margin_fig.update_layout(
                title=dict(
                    text="Margin of Victory Statistics",
                    x=0.5,
                    font=dict(size=16)
                ),
                width=800,
                height=250,
                margin=dict(t=50, b=20)
            )

            # Display both tables
            stats_fig.show()
            margin_fig.show()

            # Print text version for non-interactive environments
            print(f"\nScore Statistics:")
            print(f"Winning Teams - Mean: {w_mean:.1f}, Median: {df['WScore'].median():.1f}, "
                  f"Std: {df['WScore'].std():.1f}")
            print(f"Losing Teams  - Mean: {l_mean:.1f}, Median: {df['LScore'].median():.1f}, "
                  f"Std: {df['LScore'].std():.1f}")
            print(f"Average Margin of Victory: {(w_mean - l_mean):.1f} points")

        else:
            # Create seaborn plot
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='WScore', bins=30, kde=True, color='blue', label='Winning Score', alpha=0.5)
            sns.histplot(data=df, x='LScore', bins=30, kde=True, color='red', label='Losing Score', alpha=0.5)
            
            # Add mean lines
            plt.axvline(df['WScore'].mean(), color='blue', linestyle='--', 
                       label=f'Win Mean: {df["WScore"].mean():.1f}')
            plt.axvline(df['LScore'].mean(), color='red', linestyle='--', 
                       label=f'Loss Mean: {df["LScore"].mean():.1f}')
            
            title = "Score Distribution of Winning & Losing Teams"
            if min_season or max_season:
                title += f"\n({min_season or 'All'}-{max_season or 'Present'})"
            if game_type != 'all':
                title += f" - {game_type.title()} Games"
                
            plt.title(title)
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

if __name__ == '__main__':
    data_dir = 'data/'
    predictor = MarchMadnessPredictor(data_dir)
    predictor.load_data()
    predictor.display_win_distribution()
    predictor.display_score_distribution()

    # Default usage (Plotly visualization of all games)
    predictor.display_score_distribution()

    # Use seaborn instead of Plotly
    predictor.display_score_distribution(use_plotly=False)

    # Show only tournament games from 2015-2020
    predictor.display_score_distribution(min_season=2015, max_season=2020, game_type='tournament')

    # Show only regular season games
    predictor.display_score_distribution(game_type='regular')