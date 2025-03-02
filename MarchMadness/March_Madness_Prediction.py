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
            objective='reg:squarederror',  # Optimizes for MSE which aligns with Brier score
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
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
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

        
if __name__ == '__main__':
    data_dir = 'data/'
    predictor = MarchMadnessPredictor(data_dir)
    predictor.load_data()
    predictor.display_win_distribution()