
"""
Customer Lifetime Value Prediction for E-Commerce
Target: Customer spending during the next 90-day period.
Dataset: Online Retail.xlsx
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "Online Retail.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs_advanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Future90DaySpend"
TEST_SIZE = 0.20
RANDOM_STATE = 42

print("=" * 65)
print("CUSTOMER LIFETIME VALUE PREDICTION FOR E-COMMERCE")
print("=" * 65)

# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

print("\n[1/7] Loading and cleaning dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place Online Retail.xlsx inside the dataset folder."
    )

df = pd.read_excel(DATASET_PATH)

print("Original dataset shape:", df.shape)

required_columns = [
    "InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = df["CustomerID"].astype("string").str.strip()
df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

df = df.dropna(
    subset=[
        "CustomerID", "InvoiceDate", "InvoiceNo",
        "Quantity", "UnitPrice", "StockCode", "Country"
    ]
).copy()

# Remove cancelled invoices
df = df[~df["InvoiceNo"].str.startswith("C", na=False)].copy()

# Remove invalid transactions
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

# Calculate transaction revenue
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

df = df.sort_values("InvoiceDate").reset_index(drop=True)

if df["InvoiceDate"].nunique() < 100:
    raise ValueError("Insufficient dates for a 90-day target period.")

df.to_csv(OUTPUT_DIR / "cleaned_online_retail.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("Unique customers:", df["CustomerID"].nunique())
print("Unique invoices:", df["InvoiceNo"].nunique())
print("Total revenue:", round(df["Revenue"].sum(), 2))
print("Cleaned data saved.")

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n[2/7] Creating EDA graphs...")

# Graph 1: Monthly revenue
monthly_revenue = (
    df.set_index("InvoiceDate")["Revenue"].resample("MS").sum()
)

plt.figure(figsize=(10, 5))
monthly_revenue.plot(marker="o")
plt.title("Monthly Revenue Analysis")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_monthly_revenue.png", dpi=300)
plt.close()

# Graph 2: Top countries by revenue
country_revenue = (
    df.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
)

plt.figure(figsize=(10, 6))
country_revenue.plot(kind="barh")
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_country_revenue.png", dpi=300)
plt.close()

# Graph 3: Top customers by historical spending
customer_spend = (
    df.groupby("CustomerID")["Revenue"].sum().nlargest(10).sort_values()
)

plt.figure(figsize=(10, 6))
customer_spend.plot(kind="barh")
plt.title("Top 10 Customers by Historical Spending")
plt.xlabel("Historical Revenue")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_top_customers.png", dpi=300)
plt.close()

# Graph 4: Orders vs spending
orders_spend = df.groupby("CustomerID").agg(
    Orders=("InvoiceNo", "nunique"),
    Spending=("Revenue", "sum")
)

plt.figure(figsize=(8, 5))
plt.scatter(orders_spend["Orders"], orders_spend["Spending"], alpha=0.5)
plt.title("Number of Orders vs Customer Spending")
plt.xlabel("Number of Orders")
plt.ylabel("Historical Spending")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_orders_vs_spending.png", dpi=300)
plt.close()

# ============================================================
# 4. CREATE HISTORICAL FEATURES AND FUTURE TARGET
# ============================================================

print("\n[3/7] Creating customer features and 90-day target...")

# Reserve the final 90 days as the target period.
# All input features are calculated before this period.
dataset_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
cutoff_date = dataset_end - pd.Timedelta(days=90)

history = df[df["InvoiceDate"] < cutoff_date].copy()
future = df[
    (df["InvoiceDate"] >= cutoff_date) &
    (df["InvoiceDate"] < dataset_end)
].copy()

if history.empty or future.empty:
    raise ValueError("Historical or future target period is empty.")

features = history.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate", "max"),
    FirstPurchase=("InvoiceDate", "min"),
    Frequency=("InvoiceNo", "nunique"),
    HistoricalRevenue=("Revenue", "sum"),
    TotalQuantity=("Quantity", "sum"),
    AverageUnitPrice=("UnitPrice", "mean"),
    UniqueProducts=("StockCode", "nunique"),
    Country=(
        "Country",
        lambda values: (
            values.mode().iloc[0] if not values.mode().empty else "Unknown"
        )
    )
)

features["RecencyDays"] = (
    cutoff_date - features["LastPurchase"]
).dt.days

features["CustomerTenureDays"] = (
    cutoff_date - features["FirstPurchase"]
).dt.days

features["AverageOrderValue"] = (
    features["HistoricalRevenue"] / features["Frequency"].clip(lower=1)
)

features["PurchaseRate"] = (
    features["Frequency"] / (features["CustomerTenureDays"] + 1)
)

features = features.drop(columns=["LastPurchase", "FirstPurchase"])

# Future target includes zero for customers who do not return.
future_spending = future.groupby("CustomerID")["Revenue"].sum()

features[TARGET_COLUMN] = (
    future_spending.reindex(features.index).fillna(0.0)
)

model_data = (
    features.replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index()
)

model_data.to_csv(OUTPUT_DIR / "clv_model_dataset.csv", index=False)

print("Historical period ends before:", cutoff_date)
print("Target period:", cutoff_date, "to", dataset_end)
print("Customers with historical data:", len(model_data))
print(
    "Customers with future spending:",
    int((model_data[TARGET_COLUMN] > 0).sum())
)
print(
    "Customers with zero future spending:",
    int((model_data[TARGET_COLUMN] == 0).sum())
)
print("Customer modelling dataset saved.")

# ============================================================
# 5. TRAIN AND EVALUATE REGRESSION MODELS
# ============================================================

print("\n[4/7] Training regression models...")

X = model_data.drop(columns=["CustomerID", TARGET_COLUMN])
y = model_data[TARGET_COLUMN]

categorical_features = ["Country"]
numeric_features = [
    column for column in X.columns
    if column not in categorical_features
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

results = []
fitted_models = {}
prediction_columns = {}

for name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    # Predicted spending cannot be negative.
    predictions = np.maximum(pipeline.predict(X_test), 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    fitted_models[name] = pipeline
    prediction_columns[name.replace(" ", "_") + "_Prediction"] = predictions

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

results_df = (
    pd.DataFrame(results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# Select best model using the lowest RMSE
best_name = results_df.loc[0, "Model"]
best_model = fitted_models[best_name]
best_predictions = prediction_columns[
    best_name.replace(" ", "_") + "_Prediction"
]

joblib.dump(best_model, OUTPUT_DIR / "best_clv_model.joblib")

# Save best model predictions
prediction_results = pd.DataFrame({
    "CustomerID": model_data.loc[X_test.index, "CustomerID"].values,
    "Actual_Future90DaySpend": y_test.values,
    "Predicted_Future90DaySpend": best_predictions
})

prediction_results.to_csv(
    OUTPUT_DIR / "best_model_predictions.csv",
    index=False
)

# Save all models' predictions
all_predictions = pd.DataFrame({
    "Actual_Future90DaySpend": y_test.reset_index(drop=True)
})

for column, values in prediction_columns.items():
    all_predictions[column] = values

all_predictions.to_csv(
    OUTPUT_DIR / "model_predictions.csv",
    index=False
)

print("\nModel comparison:")
print(results_df.to_string(index=False))
print("\nBest model by RMSE:", best_name)
print("Saved best_clv_model.joblib")

# ============================================================
# 6. MODEL COMPARISON AND EVALUATION GRAPHS
# ============================================================

print("\n[5/7] Creating evaluation graphs...")

# RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Regression Model Comparison - RMSE")
plt.xlabel("Regression Model")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_model_comparison_rmse.png", dpi=300)
plt.close()

# MAE comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Regression Model Comparison - MAE")
plt.xlabel("Regression Model")
plt.ylabel("MAE (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_model_comparison_mae.png", dpi=300)
plt.close()

# Actual vs predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.55)

maximum = max(
    float(y_test.max()),
    float(np.max(best_predictions)),
    1.0
)

plt.plot([0, maximum], [0, maximum], linestyle="--")
plt.title(f"Actual vs Predicted 90-Day Spending ({best_name})")
plt.xlabel("Actual 90-Day Spending")
plt.ylabel("Predicted 90-Day Spending")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_actual_vs_predicted.png", dpi=300)
plt.close()

# Residual distribution
residuals = y_test.to_numpy() - best_predictions

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=40)
plt.title("Prediction Residual Distribution")
plt.xlabel("Actual - Predicted Spending")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_residual_distribution.png", dpi=300)
plt.close()

# ============================================================
# 7. FEATURE IMPORTANCE AND PROJECT SUMMARY
# ============================================================

print("\n[6/7] Calculating feature importance...")

importance_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance_result.importances_mean
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance_df.head(10).sort_values("Importance")

plt.figure(figsize=(9, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top Features for 90-Day Spending Prediction")
plt.xlabel("Permutation Importance")
plt.ylabel("Customer Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_feature_importance.png", dpi=300)
plt.close()

# Write project summary
summary_path = OUTPUT_DIR / "project_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("CUSTOMER LIFETIME VALUE PREDICTION FOR E-COMMERCE\n")
    file.write("=" * 55 + "\n")
    file.write("Dataset: Online Retail\n")
    file.write("Target: Customer spending during the 90-day target window\n")
    file.write(
        "Note: This is an estimated CLV proxy, not true lifetime spending.\n\n"
    )
    file.write(f"Historical period ends before: {cutoff_date}\n")
    file.write(f"Target window: {cutoff_date} to {dataset_end}\n")
    file.write(f"Customers in modelling dataset: {len(model_data)}\n")
    file.write(f"Best model by RMSE: {best_name}\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nTop features by permutation importance:\n")
    file.write(importance_df.head(10).to_string(index=False))
    file.write("\n")

print("\n[7/7] PROJECT COMPLETED SUCCESSFULLY!")
print("Output folder:", OUTPUT_DIR.resolve())
print("\nImportant output files:")
print("- cleaned_online_retail.csv")
print("- clv_model_dataset.csv")
print("- model_comparison.csv")
print("- model_predictions.csv")
print("- best_model_predictions.csv")
print("- best_clv_model.joblib")
print("- feature_importance.csv")
print("- project_summary.txt")
print("\nNote: The target is 90-day future spending, used as a CLV proxy.")
