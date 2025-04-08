# Kaggle Competitions Repository

This repository contains machine learning projects for different Kaggle competitions.

## Projects

### March Madness Prediction

A comprehensive machine learning approach to predict NCAA basketball tournament outcomes, focusing on minimizing the Brier score.

#### Project Goal

The March Madness prediction project aims to achieve a low Brier score, which measures the accuracy of probabilistic predictions. The Brier score is the mean squared error between predicted probabilities and actual outcomes:

$$
Brier = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2
$$

Where:
- $p_i$ is the predicted probability of team 1 winning
- $o_i$ is the actual outcome (1 if team 1 won, 0 if team 2 won)
- Lower scores indicate better performance (perfect score = 0)

#### Methodology

1. **Data Preparation**
   - Combined regular season and tournament detailed results
   - Created differential features between teams (shooting percentages, rebounds, assists)
   - Engineered team performance metrics and seed-based features

2. **Model Selection**
   - Used XGBoost for its strong performance on binary classification problems
   - Optimized hyperparameters specifically for Brier score minimization
   - Feature importance analysis showed field goal percentage difference was the most significant predictor (40.5%)

3. **Evaluation**
   - Achieved a validation Brier score of 0.00349
   - Model accuracy of 99.8% on validation data

#### Project Structure

- `MarchMadness/` - Contains the March Madness prediction project
  - `data/` - Tournament and team statistics data
  - `march_madness_prediction.ipynb` - Main notebook with implementation
  - `submission.csv` - Kaggle submission file
  - `requirements.txt` - Required Python packages

### JaneStreetMarket (In Progress)

A new project focused on the Jane Street Market Prediction competition. More details will be added as the project develops.

