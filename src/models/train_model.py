import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_linear_regression(X_train, y_train):
    """
    Train and save a Linear Regression model.
    """
    model = LinearRegression()
    lr_model = model.fit(X_train, y_train)
    
    # Save the trained model using pickle
    with open('src/models/lr_model.pkl', 'wb') as file:
        pickle.dump(lr_model, file)
    
    return lr_model

def train_decision_tree(X_train, y_train, max_depth=5, min_samples_leaf=5):
    """
    Train and save a Decision Tree model with regularization parameters to prevent overfitting.
    """
    dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    dt_model = dt.fit(X_train, y_train)
    
    # Save the trained model using pickle
    with open('src/models/dt_model.pkl', 'wb') as file:
        pickle.dump(dt_model, file)
    
    return dt_model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train and save a Random Forest model.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model = rf.fit(X_train, y_train)
    
    # Save the trained model using pickle
    with open('src/models/rf_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)
    
    return rf_model

def evaluate_model(model, X, y):
    """
    Evaluate a model using Mean Absolute Error (MAE).
    """
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    return mae
