from src.features.build_features import create_dummy_vars, separate_features_target
from src.models.train_model import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model
from src.models.predict_model import predict_linear_regression, predict_decision_tree, predict_random_forest
from src.visualization.visualize import plot_feature_importance, plot_model_error, plot_predictions_vs_actuals
from sklearn.model_selection import train_test_split
import pandas as pd

def load_processed_data():
    """
    Load the processed dataset from 'data/processed/Processed_final.csv'.
    Returns:
        pd.DataFrame: Loaded processed dataset.
    """
    dataset_path = 'data/processed/Processed_final.csv'  # Path to the processed data
    df = pd.read_csv(dataset_path)
    return df

def main():
    # Load the processed dataset (which was saved in 'build_features.py')
    df = load_processed_data()  # Load the processed data from 'data/processed/Processed_final.csv'
    
    # Feature engineering: Create dummy variables and separate features and target
    X, y = separate_features_target(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate models on training data
    lr_train_mae = evaluate_model(lr_model, X_train, y_train)
    dt_train_mae = evaluate_model(dt_model, X_train, y_train)
    rf_train_mae = evaluate_model(rf_model, X_train, y_train)
    
    # Evaluate models on testing data
    lr_test_mae = evaluate_model(lr_model, X_test, y_test)
    dt_test_mae = evaluate_model(dt_model, X_test, y_test)
    rf_test_mae = evaluate_model(rf_model, X_test, y_test)
    
    # Print model evaluation results
    print(f"Train MAE (Linear Regression): {lr_train_mae}")
    print(f"Test MAE (Linear Regression): {lr_test_mae}")
    print(f"Train MAE (Decision Tree): {dt_train_mae}")
    print(f"Test MAE (Decision Tree): {dt_test_mae}")
    print(f"Train MAE (Random Forest): {rf_train_mae}")
    print(f"Test MAE (Random Forest): {rf_test_mae}")
    
    # Make predictions with each model
    lr_predictions = predict_linear_regression(lr_model, X_test)
    dt_predictions = predict_decision_tree(dt_model, X_test)
    rf_predictions = predict_random_forest(rf_model, X_test)
    
    # Visualize feature importance for tree-based models
    plot_feature_importance(dt_model, X.columns)
    plot_feature_importance(rf_model, X.columns)
    
    # Visualize model error
    plot_model_error(lr_train_mae, lr_test_mae)
    plot_model_error(dt_train_mae, dt_test_mae)
    plot_model_error(rf_train_mae, rf_test_mae)
    
    # Visualize predictions vs actuals
    plot_predictions_vs_actuals(y_test, lr_predictions)
    plot_predictions_vs_actuals(y_test, dt_predictions)
    plot_predictions_vs_actuals(y_test, rf_predictions)

if __name__ == "__main__":
    main()
