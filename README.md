
# Customer Lifetime Value Prediction for E-Commerce

## Project Overview
This project uses machine learning regression algorithms to estimate customer spending over the next 90 days using historical purchasing behavior.

## Objectives
- Analyze e-commerce transaction data.
- Clean and preprocess customer records.
- Explore revenue and purchasing patterns.
- Train and compare regression models.
- Predict future customer spending.
- Provide an interactive dashboard using Streamlit.

## Dataset
- Dataset: Online Retail
- File: `Online Retail.xlsx`
- Source: UCI Machine Learning Repository
- Target: Future spending during the next 90 days, used as a proxy for Customer Lifetime Value.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Joblib

## Machine Learning Models
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## Application Features
- Customer analytics dashboard
- Interactive 90-day spending prediction
- Model comparison and evaluation graphs
- Customer data explorer
- Prediction history
- CSV download options

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

First, generate the trained model and outputs:

```bash
python main.py
```

Then launch the web application:

```bash
python -m streamlit run streamlit_app.py
```

## Project Structure

```text
Customer_Lifetime_Value_Prediction/
├── main.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── Online Retail.xlsx
└── outputs_advanced/
    ├── best_clv_model.joblib
    ├── model_comparison.csv
    ├── clv_model_dataset.csv
    ├── best_model_predictions.csv
    ├── feature_importance.csv
    └── prediction_history.csv
```

## Results
Linear Regression achieved the best results among the three tested models:
- MAE: 908.52
- RMSE: 6871.76
- R²: 0.2744

These results are from the current test split and may vary if the dataset or training process changes.

## Limitation
The dataset does not contain actual lifetime spending. Therefore, spending during the next 90 days is used as a proxy for Customer Lifetime Value. Predictions are estimates, not guaranteed future revenue.

## Future Scope
- Evaluate additional regression algorithms.
- Improve feature engineering and validation.
- Integrate a database for storing prediction records.
- Deploy the application online.
