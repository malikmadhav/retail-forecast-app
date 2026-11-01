# Retail Demand Forecasting (CSE3141 mini project)

Streamlit app that forecasts retail sales. Steps inside the app: Data -> Clean -> Explore -> Models -> Forecast (+ Viva Help page).

## Files (only these are needed)
- `app.py`            - the whole project (one file)
- `requirements.txt`  - libraries
- `data/us_retail_sales_monthly.csv` - real monthly US retail sales
- `data/large_store_sales.csv`       - large daily store dataset (87,660 rows)

## Run on your laptop
    pip install -r requirements.txt
    streamlit run app.py

## Deploy (free)
1. GitHub -> New repository -> name `retail-forecast-app` -> Public.
2. Upload `app.py`, `requirements.txt`, `README.md` and the `data` folder (drag them into the browser). Commit.
3. share.streamlit.io -> Create app -> pick the repo -> branch `main` -> main file `app.py` -> Deploy.

## Use a real Kaggle file
Open the app -> Step 1 -> "Upload my own CSV" -> choose the date column and the sales column -> Load.
Works with Kaggle "Walmart Sales Forecast", "Store Sales - Time Series Forecasting", etc.
