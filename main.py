# House Prices: Advanced Regression Techniques
# Full Python Script with Detailed Comments

# Author: ChatGPT
# Date: 10-11-2024

# ===============================
# Import Necessary Libraries
# ===============================

import warnings  # To suppress warnings for cleaner output

import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import seaborn as sns  # For data visualization
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV

# ===============================
# Initial Setup
# ===============================

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style for better aesthetics
sns.set_style('darkgrid')

# ===============================
# Step 1: Load the Data
# ===============================

# Read the training and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Display the shape of the datasets
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

# ===============================
# Step 2: Exploratory Data Analysis (EDA)
# ===============================

# Uncomment the following lines if you want to perform EDA

# # Check for missing values in the training data
# missing = train.isnull().sum()
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# print("Missing values in train data:\n", missing)

# # Visualize the distribution of the target variable 'SalePrice'
# plt.figure(figsize=(10,6))
# sns.histplot(train['SalePrice'], kde=True)
# plt.title('SalePrice Distribution')
# plt.xlabel('SalePrice')
# plt.ylabel('Frequency')
# plt.show()

# ===============================
# Step 3: Log Transformation of the Target Variable
# ===============================

# Apply log transformation to 'SalePrice' to normalize its distribution
train['SalePrice'] = np.log1p(train['SalePrice'])

# ===============================
# Step 4: Data Preprocessing
# ===============================

# Combine train and test data for consistent preprocessing
train['Train'] = True  # Mark training data
test['Train'] = False  # Mark test data
test['SalePrice'] = np.nan  # Add 'SalePrice' column to test data for consistency
full_data = pd.concat([train, test], sort=False)  # Combine datasets

# Handle Missing Values

# A. Numerical Features
# Exclude 'SalePrice' from numerical features
num_features = full_data.select_dtypes(include=[np.number]).columns.drop('SalePrice')

# Fill missing values in numerical features with the median
for feature in num_features:
    full_data[feature].fillna(full_data[feature].median(), inplace=True)

# B. Categorical Features
# Fill missing values in categorical features with the string 'Missing'
cat_features = full_data.select_dtypes(include=['object']).columns
for feature in cat_features:
    full_data[feature].fillna('Missing', inplace=True)

# Encode Categorical Variables
# Convert categorical variables into dummy/indicator variables (one-hot encoding)
full_data = pd.get_dummies(full_data, columns=cat_features, drop_first=True)

# Feature Engineering
# Create new feature 'TotalSF' as the sum of 'TotalBsmtSF', '1stFlrSF', and '2ndFlrSF'
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']

# Drop Unnecessary Columns
# Remove 'Id' and 'Train' columns as they are no longer needed
full_data.drop(['Id', 'Train'], axis=1, inplace=True)

# ===============================
# Step 5: Splitting Data Back into Train and Test Sets
# ===============================

# Separate the data back into training and test sets
train_processed = full_data[~full_data['SalePrice'].isnull()]  # Training data
test_processed = full_data[full_data['SalePrice'].isnull()].drop(['SalePrice'], axis=1)  # Test data


# Define Features and Target Variable
X = train_processed.drop(['SalePrice'], axis=1)  # Features
y = train_processed['SalePrice']  # Target variable (log-transformed 'SalePrice')

# ===============================
# Step 6: Model Building and Evaluation
# ===============================

# Split the training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Function to evaluate model performance
def evaluate_model(model, X_valid, y_valid):
    """
    Evaluate the model using Root Mean Squared Log Error (RMSLE).

    Parameters:
    - model: Trained model
    - X_valid: Validation features
    - y_valid: Actual target values (log-transformed)

    Returns:
    - rmsle: Calculated RMSLE on the validation set
    """
    y_pred = model.predict(X_valid)
    # Exponentiate the predictions and actual values to get back to original scale
    y_pred_exp = np.expm1(y_pred)
    y_valid_exp = np.expm1(y_valid)
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_valid_exp, y_pred_exp))
    return rmsle


# A. Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
rmsle_lr = evaluate_model(lr, X_valid, y_valid)
print(f'Linear Regression RMSLE: {rmsle_lr}')

# B. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rmsle_rf = evaluate_model(rf, X_valid, y_valid)
print(f'Random Forest RMSLE: {rmsle_rf}')

# C. Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.05, random_state=42
)
gbr.fit(X_train, y_train)
rmsle_gbr = evaluate_model(gbr, X_valid, y_valid)
print(f'Gradient Boosting RMSLE: {rmsle_gbr}')

# ===============================
# Step 7: Hyperparameter Tuning for Gradient Boosting Regressor
# ===============================

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4, 5]
}

# Initialize Grid Search with Cross-Validation
grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',  # Use negative MSE as scoring metric
    n_jobs=-1
)

# Fit the Grid Search to find the best parameters
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Evaluate the Best Model on Validation Set
rmsle_best = evaluate_model(best_model, X_valid, y_valid)
print(f'Best Model RMSLE: {rmsle_best}')

# ===============================
# Step 8: Residual Analysis (Optional)
# ===============================

# Exponentiate predictions and actual values
y_valid_exp = np.expm1(y_valid)
y_pred_best_exp = np.expm1(best_model.predict(X_valid))

# A. Plot Actual vs Predicted SalePrice
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_valid_exp, y=y_pred_best_exp)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.plot([min(y_valid_exp), max(y_valid_exp)], [min(y_valid_exp), max(y_valid_exp)], 'r--')  # Diagonal line
plt.show()

# B. Plot Residuals Distribution
residuals = y_valid_exp - y_pred_best_exp
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# ===============================
# Step 9: Retrain Best Model on Full Training Data
# ===============================

# Retrain the best model on the entire training data
best_model.fit(X, y)

# ===============================
# Step 10: Predict on Test Data
# ===============================

# Check the test_processed DataFrame before prediction
print("Test Processed Shape:", test_processed.shape)
print("Test Processed Head:\n", test_processed.head())

# Ensure there are samples in the test DataFrame
if test_processed.empty:
    raise ValueError("Test data is empty!")

# Ensure that the features are consistent with the training data
missing_columns = set(X.columns) - set(test_processed.columns)
if missing_columns:
    raise ValueError(f"Test data is missing the following columns: {missing_columns}")

# Predict 'SalePrice' for the test data
test_preds = best_model.predict(test_processed)

# Exponentiate the predictions to get back to original scale
test_preds = np.expm1(test_preds)

# ===============================
# Step 11: Prepare Submission File
# ===============================

# Create a DataFrame for submission
submission = pd.DataFrame({
    'Id': test['Id'],  # Use the 'Id' from the test data
    'SalePrice': test_preds  # Predicted 'SalePrice'
})

# Save the submission file in CSV format
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")
