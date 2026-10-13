
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_advanced"

MODEL_PATH = OUTPUT_DIR / "best_clv_model.joblib"
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_comparison.csv"
CUSTOMER_DATA_PATH = OUTPUT_DIR / "clv_model_dataset.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "best_model_predictions.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
HISTORY_PATH = OUTPUT_DIR / "prediction_history.csv"

st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


model = load_model()
model_results = load_csv(str(MODEL_RESULTS_PATH))
customer_data = load_csv(str(CUSTOMER_DATA_PATH))
predictions = load_csv(str(PREDICTIONS_PATH))
importance = load_csv(str(IMPORTANCE_PATH))


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def find_target_column(df):
    """Find the future spending target column."""
    if df.empty:
        return None

    preferred_names = [
        "TargetFutureSpending",
        "FutureSpending",
        "TargetRevenue",
        "CLV"
    ]

    for name in preferred_names:
        if name in df.columns:
            return name

    candidates = [
        col for col in df.columns
        if "future" in col.lower()
        and ("spend" in col.lower() or "revenue" in col.lower())
    ]

    return candidates[0] if candidates else None


def save_prediction(record):
    """Append a prediction to local CSV history."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if HISTORY_PATH.exists():
        old_history = pd.read_csv(HISTORY_PATH)
        updated_history = pd.concat(
            [old_history, record],
            ignore_index=True
        )
    else:
        updated_history = record.copy()

    updated_history.to_csv(HISTORY_PATH, index=False)


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🛍️ CLV Analytics")
st.sidebar.caption("E-Commerce Customer Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Customer Spending",
        "Model Evaluation",
        "Data Explorer",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Prediction target: spending during the next 90 days."
)
st.sidebar.caption("M.Tech Machine Learning Project")


# =========================================================
# 5. MODEL VALIDATION
# =========================================================

if model is None:
    st.error(
        "Saved model not found. Please check this file:\n\n"
        "outputs_advanced/best_clv_model.joblib"
    )
    st.stop()


# =========================================================
# 6. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Customer Lifetime Value Prediction")
    st.write(
        "Analyze customer behavior, compare regression models, "
        "and estimate future customer spending."
    )

    if customer_data.empty:
        st.warning(
            "Customer dataset not found. Please run main.py first."
        )
        st.stop()

    target_col = find_target_column(customer_data)

    total_customers = len(customer_data)

    if target_col:
        average_spending = customer_data[target_col].mean()
        total_spending = customer_data[target_col].sum()
    else:
        average_spending = 0.0
        total_spending = 0.0
        st.warning(
            "Future spending target column was not detected. "
            "Please check clv_model_dataset.csv."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Historical Customers",
        f"{total_customers:,}"
    )
    col2.metric(
        "Average 90-Day Spending",
        f"{average_spending:,.2f}"
    )
    col3.metric(
        "Total 90-Day Spending",
        f"{total_spending:,.2f}"
    )

    st.divider()
    st.subheader("Customer Analytics")

    col1, col2 = st.columns(2)

    with col1:
        if "HistoricalRevenue" in customer_data.columns:
            fig = px.histogram(
                customer_data,
                x="HistoricalRevenue",
                nbins=35,
                title="Distribution of Historical Customer Revenue",
                labels={
                    "HistoricalRevenue": "Historical Revenue"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if {
            "Frequency",
            "HistoricalRevenue"
        }.issubset(customer_data.columns):

            fig = px.scatter(
                customer_data,
                x="Frequency",
                y="HistoricalRevenue",
                title="Purchase Frequency vs Historical Revenue",
                labels={
                    "Frequency": "Number of Orders",
                    "HistoricalRevenue": "Historical Revenue"
                },
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("EDA Visualizations")

    image_files = [
        ("01_monthly_revenue.png", "Monthly Revenue Trend"),
        ("02_country_revenue.png", "Revenue by Country"),
        ("03_top_customers.png", "Top Customers by Revenue"),
        ("04_orders_vs_spending.png", "Orders vs Spending")
    ]

    for image_name, image_title in image_files:
        image_path = OUTPUT_DIR / image_name

        if image_path.exists():
            with st.expander(image_title):
                st.image(
                    str(image_path),
                    caption=image_title,
                    use_container_width=True
                )

    st.caption(
        "Note: Future spending over 90 days is used as a proxy "
        "for Customer Lifetime Value, not actual lifetime spending."
    )


# =========================================================
# 7. CUSTOMER SPENDING PREDICTION
# =========================================================

elif page == "Predict Customer Spending":

    st.title("Predict Customer Spending")
    st.write(
        "Enter customer purchase information to estimate "
        "spending over the next 90 days."
    )

    if customer_data.empty:
        st.error("Customer dataset not found.")
        st.stop()

    required_features = [
        "Frequency",
        "HistoricalRevenue",
        "TotalQuantity",
        "AverageUnitPrice",
        "UniqueProducts",
        "Country",
        "RecencyDays",
        "CustomerTenureDays",
        "AverageOrderValue",
        "PurchaseRate"
    ]

    missing_features = [
        col for col in required_features
        if col not in customer_data.columns
    ]

    if missing_features:
        st.error(
            "Expected training features are missing: "
            + ", ".join(missing_features)
        )
        st.stop()

    countries = sorted(
        customer_data["Country"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not countries:
        countries = ["United Kingdom"]

    st.info(
        "Use values in the same units as the training dataset. "
        "The sample values below are only for testing."
    )

    with st.form("prediction_form"):

        st.subheader("Customer Purchase Details")

        col1, col2 = st.columns(2)

        with col1:
            frequency = st.number_input(
                "Number of Historical Orders",
                min_value=1,
                value=5,
                step=1
            )

            historical_revenue = st.number_input(
                "Historical Revenue",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )

            total_quantity = st.number_input(
                "Total Quantity Purchased",
                min_value=1,
                value=20,
                step=1
            )

            average_unit_price = st.number_input(
                "Average Unit Price",
                min_value=0.01,
                value=20.0,
                step=1.0
            )

            unique_products = st.number_input(
                "Unique Products Purchased",
                min_value=1,
                value=10,
                step=1
            )

        with col2:
            recency_days = st.number_input(
                "Days Since Last Purchase",
                min_value=0,
                value=30,
                step=1
            )

            tenure_days = st.number_input(
                "Customer Tenure (Days)",
                min_value=0,
                value=180,
                step=1
            )

            country = st.selectbox(
                "Customer Country",
                countries
            )

        submitted = st.form_submit_button(
            "Predict 90-Day Spending",
            type="primary",
            use_container_width=True
        )

    if submitted:

        # Keep feature names identical to training columns.
        avg_order_value = (
            historical_revenue / max(frequency, 1)
        )
        purchase_rate = frequency / (tenure_days + 1)

        input_data = pd.DataFrame([{
            "Frequency": frequency,
            "HistoricalRevenue": historical_revenue,
            "TotalQuantity": total_quantity,
            "AverageUnitPrice": average_unit_price,
            "UniqueProducts": unique_products,
            "Country": country,
            "RecencyDays": recency_days,
            "CustomerTenureDays": tenure_days,
            "AverageOrderValue": avg_order_value,
            "PurchaseRate": purchase_rate
        }])

        try:
            raw_prediction = float(
                np.asarray(
                    model.predict(input_data)
                ).reshape(-1)[0]
            )

            displayed_prediction = max(0.0, raw_prediction)

            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                "Estimated Spending in Next 90 Days",
                f"{displayed_prediction:,.2f}"
            )

            if raw_prediction < 0:
                st.caption(
                    f"Raw model output: {raw_prediction:,.2f}. "
                    "The displayed estimate is limited to zero "
                    "because spending cannot be negative."
                )

            # Create a timestamped record.
            history_record = input_data.copy()
            history_record["PredictionDate"] = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            history_record["RawPredictedSpending"] = raw_prediction
            history_record["PredictedSpending"] = displayed_prediction

            # Save prediction to CSV.
            save_prediction(history_record)

            st.success("Prediction saved to history successfully!")

            st.subheader("Input Details")
            st.dataframe(input_data, use_container_width=True)

            st.download_button(
                "Download This Prediction",
                data=history_record.to_csv(index=False).encode("utf-8"),
                file_name="customer_spending_prediction.csv",
                mime="text/csv"
            )

            st.info(
                "This is an estimate, not a guarantee of future spending. "
                "The model uses historical data and a 90-day spending target."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.caption(
                "Check the saved model, training feature names, "
                "and scikit-learn compatibility."
            )


# =========================================================
# 8. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")
    st.write(
        "Compare regression models using MAE, RMSE, and R²."
    )

    if model_results.empty:
        st.warning("model_comparison.csv was not found.")
        st.stop()

    st.dataframe(model_results, use_container_width=True)

    # Support common R-squared column name variants.
    r2_col = next(
        (
            col for col in model_results.columns
            if col.strip().lower() in {"r2", "r²", "r2_score"}
        ),
        None
    )

    required = {"Model", "MAE", "RMSE"}

    if required.issubset(model_results.columns):

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                model_results,
                x="Model",
                y="RMSE",
                title="Model Comparison: RMSE",
                text_auto=".2f"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                model_results,
                x="Model",
                y="MAE",
                title="Model Comparison: MAE",
                text_auto=".2f"
            )
            st.plotly_chart(fig, use_container_width=True)

        if r2_col:
            fig = px.bar(
                model_results,
                x="Model",
                y=r2_col,
                title="Model Comparison: R²",
                text_auto=".3f"
            )
            st.plotly_chart(fig, use_container_width=True)

        best_row = model_results.sort_values("RMSE").iloc[0]

        st.success(
            f"Best model by RMSE: {best_row['Model']} "
            f"(RMSE = {best_row['RMSE']:,.2f})"
        )

        st.caption(
            "Lower MAE and RMSE are better. Higher R² is generally better."
        )

    evaluation_images = [
        ("05_model_comparison_rmse.png", "RMSE Comparison"),
        ("06_model_comparison_mae.png", "MAE Comparison"),
        ("07_actual_vs_predicted.png", "Actual vs Predicted"),
        ("08_residual_distribution.png", "Residual Distribution"),
        ("09_feature_importance.png", "Feature Importance")
    ]

    for image_name, image_title in evaluation_images:
        image_path = OUTPUT_DIR / image_name

        if image_path.exists():
            with st.expander(image_title):
                st.image(
                    str(image_path),
                    caption=image_title,
                    use_container_width=True
                )

    if not importance.empty:
        st.subheader("Feature Importance Data")
        st.dataframe(importance, use_container_width=True)


# =========================================================
# 9. DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.title("Customer Data Explorer")

    if customer_data.empty:
        st.warning("Customer dataset is missing.")
        st.stop()

    st.write("Prepared customer-level dataset")
    st.dataframe(
        customer_data.head(100),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{customer_data.shape[0]:,}")
    col2.metric("Columns", f"{customer_data.shape[1]:,}")

    st.download_button(
        "Download Customer Dataset",
        data=customer_data.to_csv(index=False).encode("utf-8"),
        file_name="clv_model_dataset.csv",
        mime="text/csv"
    )

    st.subheader("Missing Values")

    missing = customer_data.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(missing, use_container_width=True)


# =========================================================
# 10. PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.title("Prediction History")
    st.write(
        "View, review, and download predictions saved from this app."
    )

    if HISTORY_PATH.exists():

        history = pd.read_csv(HISTORY_PATH)

        if history.empty:
            st.info("No predictions saved yet.")
        else:
            col1, col2 = st.columns(2)

            col1.metric(
                "Total Predictions",
                f"{len(history):,}"
            )

            if "PredictedSpending" in history.columns:
                col2.metric(
                    "Average Predicted Spending",
                    f"{history['PredictedSpending'].mean():,.2f}"
                )

            st.dataframe(history, use_container_width=True)

            st.download_button(
                "Download Prediction History",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

            if st.button("Clear Prediction History"):
                HISTORY_PATH.unlink(missing_ok=True)
                st.success("Prediction history cleared.")
                st.rerun()

    else:
        st.info(
            "No prediction history exists yet. "
            "Go to Predict Customer Spending and make a prediction."
        )


# =========================================================
# 11. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("About the Project")

    st.markdown("""
    ## Customer Lifetime Value Prediction for E-Commerce

    **Objective:** Estimate future customer spending using historical
    purchase behavior and regression algorithms.

    **Dataset:** Online Retail transaction dataset.

    **Machine Learning Algorithms:**
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor

    **Evaluation Metrics:**
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of Determination

    **Technology Stack:**
    Python, Pandas, NumPy, scikit-learn, Joblib,
    Plotly, and Streamlit.

    **Application Features:**
    - Interactive customer analytics dashboard
    - Future spending prediction
    - Model evaluation and visualizations
    - Customer dataset exploration
    - Prediction history and CSV downloads
    """)

    st.warning(
        "The dataset does not contain actual lifetime spending. "
        "This project uses spending during the next 90 days as a "
        "proxy for CLV. Predictions should be treated as estimates."
    )

    st.caption(
        "Developed as an M.Tech machine learning project."
    )
