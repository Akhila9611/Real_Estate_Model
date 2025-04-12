import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create a directory to save the visualizations if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def plot_feature_importance(model, feature_names, save_path='results/feature_importance.png'):
    """
    Plot and save the feature importance of a model (e.g., DecisionTree, RandomForest).
    Args:
        model: Trained model (DecisionTree, RandomForest).
        feature_names: List of feature names.
        save_path: Path where the plot will be saved.
    """
    # Get feature importance from the model
    importance = model.feature_importances_

    # Create a DataFrame to hold feature names and their corresponding importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_model_error(train_error, test_error, save_path='results/model_error.png'):
    """
    Plot and save the model error (Training vs Testing error).
    Args:
        train_error: Training error value (MAE).
        test_error: Testing error value (MAE).
        save_path: Path where the plot will be saved.
    """
    errors = {'Train Error': train_error, 'Test Error': test_error}
    
    plt.figure(figsize=(6, 6))
    sns.barplot(x=list(errors.keys()), y=list(errors.values()), palette='Blues')
    plt.title('Model Error (Train vs Test)')
    plt.xlabel('Error Type')
    plt.ylabel('Mean Absolute Error (MAE)')
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_predictions_vs_actuals(y_true, y_pred, save_path='results/predictions_vs_actuals.png'):
    """
    Plot and save predictions vs actual values.
    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        save_path: Path where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color='blue', s=80, edgecolor='black')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title('Predictions vs Actuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Plot and save the confusion matrix to a file.
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        save_path: Path where the plot will be saved.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Admitted', 'Admitted'], yticklabels=['Not Admitted', 'Admitted'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(model, save_path='results/loss_curve.png'):
    """
    Plot the loss curve of the trained model and save it as a PNG file.
    Args:
        model: The trained model that has a `loss_curve_` attribute.
        save_path: Path where the plot will be saved.
    """
    loss_values = model.loss_curve_
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
