# 🏘️ Real Estate Price Prediction Model

## Project Overview

This project focuses on building a **Real Estate Price Prediction Model** using machine learning techniques. The model estimates housing prices based on various input features such as location, area, and number of rooms. It uses a structured dataset and provides both a visual and an interactive interface for predictions.

### **Goal**:
- Develop a regression model to predict house prices accurately.
- Achieve reliable performance and interpretability through visualization and evaluation metrics.

### **Machine Learning Task**:
- **Task Type**: Regression  
- **Target Variable**: `House_Price` (continuous numerical value)  
- **Success Criteria**: Low RMSE, good R² score, and interpretable results.

---

## Files in the Project

1. **`data/`**
   - `raw/`: Contains raw input data.
   - `processed/`: Cleaned and transformed data used for model training.

2. **`results/`**
   - Contains key visual outputs:
     - `feature_importance.png`: Feature contribution to predictions.
     - `model_error.png`: Visualization of model prediction errors.
     - `predictions_vs_actual.png`: Predicted vs actual house prices.

3. **`src/`**
   - `data/`: Scripts for loading and preprocessing data.
   - `features/`: Feature selection and transformation code.
   - `models/`: Model training, validation, and evaluation functions.
   - `visualization/`: Scripts to generate charts and plots.

4. **`Real_Estate_Solution.ipynb`**
   - A Jupyter notebook containing exploratory data analysis (EDA), model prototyping, and evaluation.

5. **`main.py`**
   - Main training script to execute the full pipeline.

6. **`app.py`**
   - A **Streamlit app** that allows users to input housing features and receive a real-time price prediction.

7. **`confusion_matrix.png` / `loss_curve.png`**
   - Visual diagnostics of classification confusion and model loss over training (if applicable).

8. **`requirements.txt`**
   - Python dependencies for the entire project.

---

## Dataset

The dataset used for this project typically includes the following columns:

- **Location**: Area or locality of the house
- **Size**: Size of the house (e.g., in square feet)
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Year_Built**: Construction year
- **Lot_Size**: Size of the property lot
- **Garage**: Presence/size of garage
- **Amenities**: Any additional features (e.g., pool, garden)
- **Price**: Target variable — the actual house price

---

