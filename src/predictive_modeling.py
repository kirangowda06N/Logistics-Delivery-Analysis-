# ============================================================
# WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS
# ============================================================
#
# Objective:
# Predict delivery time using machine learning.
#
# Target variable:
# delivery_time_hours
#
# This program:
# 1. Loads the cleaned logistics dataset
# 2. Checks the dataset
# 3. Selects prediction features
# 4. Preprocesses numerical and categorical data
# 5. Splits data into training and testing sets
# 6. Creates a baseline model
# 7. Trains Linear Regression
# 8. Trains Ridge Regression
# 9. Trains Gradient Boosting
# 10. Trains Random Forest
# 11. Compares model performance
# 12. Creates visualizations
# 13. Performs high-risk prediction analysis
# 14. Performs delivery-mode what-if analysis
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning: splitting data
from sklearn.model_selection import train_test_split

# Machine learning: preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Machine learning models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Model evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders if they do not already exist
import os


# ------------------------------------------------------------
# STEP 2: CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

# These folders will contain Week 4 charts and result files.

os.makedirs("outputs/week4_charts", exist_ok=True)
os.makedirs("outputs/week4_results", exist_ok=True)

print("Output folders are ready.")


# ------------------------------------------------------------
# STEP 3: LOAD THE CLEANED LOGISTICS DATASET
# ------------------------------------------------------------

# The dataset comes from the Week 2 preprocessing stage.

df = pd.read_csv("data/Delivery_Logistics_cleaned.csv")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# STEP 4: CHECK MISSING VALUES
# ------------------------------------------------------------

# This confirms whether any missing values remain
# in the cleaned dataset.

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 5: CHECK DATA TYPES
# ------------------------------------------------------------

# This helps us understand which columns are numerical
# and which columns are categorical.

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# STEP 6: DISPLAY SUMMARY STATISTICS
# ------------------------------------------------------------

# describe() provides basic statistical information
# such as mean, standard deviation, minimum and maximum.

print("\nSummary statistics:")
print(df.describe())


# ------------------------------------------------------------
# STEP 7: DEFINE THE TARGET VARIABLE
# ------------------------------------------------------------

# This is the variable that our machine learning models
# will attempt to predict.

target = "delivery_time_hours"

print("\nTarget variable:")
print(target)


# ------------------------------------------------------------
# STEP 8: SELECT INPUT FEATURES
# ------------------------------------------------------------

# These variables will be used to predict delivery time.

features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
    "distance_km",
    "package_weight_kg",
    "expected_time_hours"
]

# X contains the input variables.
X = df[features]

# y contains the target variable.
y = df[target]

print("\nFeature shape:")
print(X.shape)

print("Target shape:")
print(y.shape)


# ------------------------------------------------------------
# STEP 9: DEFINE NUMERICAL FEATURES
# ------------------------------------------------------------

# Numerical variables contain numbers and can be scaled.

numeric_features = [
    "distance_km",
    "package_weight_kg",
    "expected_time_hours"
]


# ------------------------------------------------------------
# STEP 10: DEFINE CATEGORICAL FEATURES
# ------------------------------------------------------------

# Categorical variables contain labels or categories.

categorical_features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition"
]


# ------------------------------------------------------------
# STEP 11: PREPROCESS NUMERICAL FEATURES
# ------------------------------------------------------------

# Median imputation handles missing numerical values.
# StandardScaler puts numerical variables on a similar scale.

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ------------------------------------------------------------
# STEP 12: PREPROCESS CATEGORICAL FEATURES
# ------------------------------------------------------------

# Most-frequent imputation handles missing categorical values.
# OneHotEncoder converts categories into numerical columns.

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ------------------------------------------------------------
# STEP 13: COMBINE BOTH PREPROCESSING METHODS
# ------------------------------------------------------------

# ColumnTransformer applies the correct preprocessing
# to numerical and categorical columns.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ------------------------------------------------------------
# STEP 14: SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

# 80% of the data is used for training.
# 20% is reserved for testing.
#
# random_state=42 makes the result reproducible.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:")
print(len(X_train))

print("Testing records:")
print(len(X_test))


# ------------------------------------------------------------
# STEP 15: BASELINE MODEL
# ------------------------------------------------------------

# The baseline always predicts the average delivery time
# from the training data.
#
# This gives us a simple reference point against which
# machine learning models can be compared.

baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)

print("\n==============================")
print("BASELINE MODEL")
print("==============================")

print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)


# ------------------------------------------------------------
# STEP 16: LINEAR REGRESSION
# ------------------------------------------------------------

# Linear Regression is used as a simple machine learning model
# for predicting delivery time.

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# IMPORTANT:
# First train the model using the training data.

linear_model.fit(
    X_train,
    y_train
)

# Then make predictions on the test data.

linear_predictions = linear_model.predict(
    X_test
)

# Calculate evaluation metrics.

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2:", linear_r2)


# ------------------------------------------------------------
# STEP 17: RIDGE REGRESSION
# ------------------------------------------------------------

# Ridge Regression is a regularized form of Linear Regression.
# alpha=10 controls the strength of regularization.

ridge_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=10))
    ]
)

# Train the Ridge model.

ridge_model.fit(
    X_train,
    y_train
)

# Generate predictions.

ridge_predictions = ridge_model.predict(
    X_test
)

# Calculate evaluation metrics.

ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ridge_predictions
    )
)

ridge_r2 = r2_score(
    y_test,
    ridge_predictions
)

print("\n==============================")
print("RIDGE REGRESSION")
print("==============================")

print("MAE:", ridge_mae)
print("RMSE:", ridge_rmse)
print("R2:", ridge_r2)


# ------------------------------------------------------------
# STEP 18: GRADIENT BOOSTING
# ------------------------------------------------------------

# Gradient Boosting is an ensemble machine learning method.
# It combines multiple weak models to improve prediction.

gradient_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                random_state=42
            )
        )
    ]
)

# Train the Gradient Boosting model.

gradient_model.fit(
    X_train,
    y_train
)

# Generate predictions.

gradient_predictions = gradient_model.predict(
    X_test
)

# Calculate evaluation metrics.

gradient_mae = mean_absolute_error(
    y_test,
    gradient_predictions
)

gradient_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gradient_predictions
    )
)

gradient_r2 = r2_score(
    y_test,
    gradient_predictions
)

print("\n==============================")
print("GRADIENT BOOSTING")
print("==============================")

print("MAE:", gradient_mae)
print("RMSE:", gradient_rmse)
print("R2:", gradient_r2)


# ------------------------------------------------------------
# STEP 19: RANDOM FOREST
# ------------------------------------------------------------

# Random Forest uses many decision trees to make predictions.

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train the Random Forest model.

random_forest_model.fit(
    X_train,
    y_train
)

# Generate predictions.

rf_predictions = random_forest_model.predict(
    X_test
)

# Calculate evaluation metrics.

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)


# ------------------------------------------------------------
# STEP 20: COMPARE ALL MODELS
# ------------------------------------------------------------

# Create a DataFrame containing the performance
# of every model.

model_results = pd.DataFrame({
    "Model": [
        "Mean Baseline",
        "Linear Regression",
        "Ridge Regression",
        "Gradient Boosting",
        "Random Forest"
    ],
    "MAE": [
        baseline_mae,
        linear_mae,
        ridge_mae,
        gradient_mae,
        rf_mae
    ],
    "RMSE": [
        baseline_rmse,
        linear_rmse,
        ridge_rmse,
        gradient_rmse,
        rf_rmse
    ],
    "R2": [
        baseline_r2,
        linear_r2,
        ridge_r2,
        gradient_r2,
        rf_r2
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(model_results)


# ------------------------------------------------------------
# STEP 21: SAVE MODEL COMPARISON
# ------------------------------------------------------------

# Save the model results as a CSV file.

model_results.to_csv(
    "outputs/week4_results/model_comparison.csv",
    index=False
)

print("\nModel comparison saved.")


# ------------------------------------------------------------
# STEP 22: ACTUAL VS PREDICTED VISUALIZATION
# ------------------------------------------------------------

# This chart compares actual delivery times with
# Ridge Regression predictions.

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    ridge_predictions,
    alpha=0.5
)

plt.xlabel("Actual Delivery Time (hours)")
plt.ylabel("Predicted Delivery Time (hours)")
plt.title("Actual vs Predicted Delivery Time")

# The diagonal line represents perfect predictions.

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "outputs/week4_charts/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 23: RESIDUAL ANALYSIS
# ------------------------------------------------------------

# A residual is the difference between:
#
# Actual value - Predicted value
#
# Residual analysis helps us understand prediction errors.

residuals = y_test - ridge_predictions

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=30
)

plt.xlabel("Prediction Error (hours)")
plt.ylabel("Frequency")
plt.title("Residual Distribution - Ridge Regression")

plt.tight_layout()

plt.savefig(
    "outputs/week4_charts/residual_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 24: MODEL COMPARISON VISUALIZATION
# ------------------------------------------------------------

# RMSE is used here to visually compare model errors.
# Lower RMSE means smaller prediction error.

plt.figure(figsize=(9, 6))

plt.bar(
    model_results["Model"],
    model_results["RMSE"]
)

plt.xlabel("Model")
plt.ylabel("RMSE (hours)")
plt.title("Model Comparison by RMSE")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "outputs/week4_charts/model_comparison_rmse.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 25: HIGH-RISK DELIVERY ANALYSIS
# ------------------------------------------------------------

# The 90th percentile is used as a threshold.
# Deliveries predicted above this threshold are treated
# as high-risk for long delivery time.

prediction_threshold = np.percentile(
    ridge_predictions,
    90
)

high_risk = ridge_predictions >= prediction_threshold

risk_summary = pd.DataFrame({
    "Predicted_Time": ridge_predictions,
    "Actual_Time": y_test.values,
    "High_Risk": high_risk
})

print("\n==============================")
print("HIGH-RISK DELIVERY ANALYSIS")
print("==============================")

print("90th percentile prediction threshold:")
print(prediction_threshold)

print("\nHigh-risk deliveries:")
print(high_risk.sum())


# Save risk predictions.

risk_summary.to_csv(
    "outputs/week4_results/risk_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# STEP 26: PREDICTED DELIVERY TIME VS DISTANCE
# ------------------------------------------------------------

# This visualization examines how predicted delivery time
# changes with delivery distance.

plt.figure(figsize=(8, 6))

plt.scatter(
    X_test["distance_km"],
    ridge_predictions,
    alpha=0.5
)

plt.xlabel("Distance (km)")
plt.ylabel("Predicted Delivery Time (hours)")
plt.title("Predicted Delivery Time vs Distance")

plt.tight_layout()

plt.savefig(
    "outputs/week4_charts/predicted_time_vs_distance.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 27: DELIVERY MODE WHAT-IF ANALYSIS
# ------------------------------------------------------------

# We take 1,000 test records and change ONLY the
# delivery mode.
#
# This is a model-based scenario analysis.
# It should NOT be interpreted as proof of causal impact.

scenario_data = X_test.head(1000).copy()

scenario_results = []

for mode in df["delivery_mode"].dropna().unique():

    # Create a copy of the test data.

    scenario_copy = scenario_data.copy()

    # Change delivery mode for the scenario.

    scenario_copy["delivery_mode"] = mode

    # Predict delivery time under this scenario.

    predictions = ridge_model.predict(
        scenario_copy
    )

    # Store the average predicted time.

    scenario_results.append({
        "delivery_mode": mode,
        "average_predicted_time_hours": predictions.mean()
    })


# Convert scenario results into a DataFrame.

scenario_df = pd.DataFrame(
    scenario_results
)

print("\n==============================")
print("DELIVERY MODE SCENARIO ANALYSIS")
print("==============================")

print(scenario_df)


# ------------------------------------------------------------
# STEP 28: SAVE DELIVERY MODE SCENARIO RESULTS
# ------------------------------------------------------------

scenario_df.to_csv(
    "outputs/week4_results/delivery_mode_scenario_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# STEP 29: DELIVERY MODE SCENARIO VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    scenario_df["delivery_mode"],
    scenario_df["average_predicted_time_hours"]
)

plt.xlabel("Delivery Mode")
plt.ylabel("Average Predicted Delivery Time (hours)")
plt.title("Delivery Mode What-If Scenario Analysis")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "outputs/week4_charts/mode_scenario_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 30: FINAL MESSAGE
# ------------------------------------------------------------

print("\n========================================")
print("WEEK 4 ANALYSIS COMPLETED SUCCESSFULLY")
print("========================================")

print("\nGenerated charts:")
print("- actual_vs_predicted.png")
print("- residual_distribution.png")
print("- model_comparison_rmse.png")
print("- predicted_time_vs_distance.png")
print("- mode_scenario_comparison.png")

print("\nGenerated result files:")
print("- model_comparison.csv")
print("- risk_predictions.csv")
print("- delivery_mode_scenario_analysis.csv")