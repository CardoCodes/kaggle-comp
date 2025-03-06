import os
import pandas as pd
import numpy as np
from March_Madness_Prediction import MarchMadnessPredictor

def main():
    print("==== Training March Madness Prediction Model ====")
    
    # Initialize the predictor
    data_dir = './data'
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please check the path.")
        return
    
    predictor = MarchMadnessPredictor(data_dir)
    
    # Train model with hyperparameter tuning for low Brier score
    # Use more recent seasons for training to capture current trends
    # We'll use the most recent complete season for validation
    min_season = 2003  # Using data from 2003 onwards
    max_season = 2022  # Up to most recent complete data
    eval_season = 2022  # Use most recent season for validation
    
    print(f"Training on seasons {min_season}-{max_season}, evaluating on {eval_season}")
    
    # Initial training
    model, results = predictor.train_model(
        min_season=min_season, 
        max_season=max_season,
        eval_season=eval_season,
        use_detailed=True
    )
    
    # Fine-tune the model with optimized parameters
    print("\n==== Fine-tuning model for lower Brier score ====")
    
    # Create a new model with optimized parameters specifically for Brier score
    predictor.model = None  # Reset model
    predictor.create_model()  # This creates model with default parameters
    
    # Modified XGBoost parameters for better calibration (lower Brier score)
    predictor.model.n_estimators = 1000
    predictor.model.learning_rate = 0.01
    predictor.model.max_depth = 5
    predictor.model.min_child_weight = 4
    predictor.model.gamma = 0.1
    predictor.model.subsample = 0.8
    predictor.model.colsample_bytree = 0.8
    predictor.model.reg_alpha = 0.1
    predictor.model.reg_lambda = 1.0
    
    # Train again with optimized parameters
    model, results = predictor.train_model(
        min_season=min_season, 
        max_season=max_season,
        eval_season=eval_season,
        use_detailed=True
    )
    
    print("\n==== Final Model Results ====")
    predictor.print_model_results(results)
    
    print("\n==== Analysis of Prediction Distribution ====")
    # Analyze the distribution of predictions to check calibration
    X, y = predictor.prepare_features(min_season=min_season, max_season=max_season)
    X_model = X.drop(['Season', 'DayNum'], axis=1)
    
    # Get predictions
    predictions = predictor.model.predict(X_model)
    
    # Create a DataFrame for analysis
    results_df = pd.DataFrame({
        'Season': X['Season'],
        'Actual': y,
        'Predicted': predictions
    })
    
    # Print calibration analysis by binning predictions
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_df['PredictionBin'] = pd.cut(results_df['Predicted'], bins=bins)
    
    calibration = results_df.groupby('PredictionBin').agg({
        'Actual': 'mean',
        'Predicted': 'mean',
        'Season': 'count'
    }).rename(columns={'Season': 'Count'})
    
    calibration['Error'] = calibration['Predicted'] - calibration['Actual']
    print(calibration)
    
    # Save the model
    import pickle
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(predictor.model, f)
    
    print("\nModel saved as 'xgboost_model.pkl'")

if __name__ == "__main__":
    main() 