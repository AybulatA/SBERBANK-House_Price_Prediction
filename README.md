# Sberbank Russian Housing Market — Price Prediction Project


## Overview

This project tackles the Sberbank Russian Housing Market Kaggle competition, where the goal is to predict real estate prices in Moscow using a mix of property characteristics and macroeconomic indicators.

**Workflow includes:**
- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Statistical testing (ANOVA)
- Correlation & multicollinearity checks
- Model development & tuning
- Deployment with FastAPI + Docker
- Local prediction API via `predict.py`


## Dataset Summary

- **Training period:** Aug 2011 → Jun 2015  
- **Test period:** Jun 2015 → Jun 2016  
- **Target:** `price_doc` (apartment sale price in rubles)  
- **Features:**  
  - 275 numerical features  
  - 15 categorical features  
  - Additional macroeconomic dataset (2010–2016)  

**After cleaning:**
- Removed unrealistic price outliers  
- Applied log-transform to normalize target distribution  


## Key Findings From EDA

### Temporal Insights
Training and test sets cover different periods. We applied a temporal train/validation split at 2014-12-01.

### Numerical Features
Many features are highly correlated or near-duplicates (e.g., `metro_km_walk` vs `metro_min_walk`).

### Categorical Features
Mostly low cardinality except for `sub_area`. Target encoding used for categorical variables with many categories.

### Macroeconomic Data
Selected features include CPI, USD/RUB rate, etc.


## Feature Engineering

**New features created:**
- `season` → spring, summer etc.  
- `floorbin` → categorical grouping of floors  
- `days_since_start` → time progression indicator  

**Final Feature Selection Pipeline:**
- ANOVA testing for categorical influence  
- Target encoding  
- Correlation & multicollinearity checks  
- Selection of the 11 strongest features for modeling


## Modeling & Results

### Baseline
- Mean price by area → **RMSLE: 0.47**

### Ridge Regression
- Handles linear effects and multicollinearity  
- RMSLE: 0.42  
- **Key positive drivers:** `sub_area_encoded`, `full_sq`, `cpi`  
- **Negative drivers:** `kremlin_km`, `kindergarten_km`, `product_type_encoded`  

### Decision Tree Regressor
- Tuned: `max_depth=9`, `min_samples_leaf=5`  
- RMSLE: 0.352  
- Most important features: `full_sq`, `sub_area_encoded`  

### Random Forest
- Tuned: `max_depth=10`, `min_samples_leaf=5`, `n_estimators=300`  
- RMSLE: 0.336  

### XGBoost (Final Model)
- Tuned: `eta=0.05`, `max_depth=4`, `min_child_weight=10`  
- ~465 boosting rounds  
- Best RMSLE: 0.329 on validation  
- Kaggle private leaderboard: 0.343 → final selected model  


## Deployment

The final model is deployed locally using:
- FastAPI  
- Uvicorn  
- Docker  
- `uv` (environment and dependency management)


**Prediction endpoint:**  
```http
POST /predict
```
 **Local test script:**  
```bash
python app/request.py
```


## Example Request Body:

``` json
{
"sub_area": "juzhnoe_butovo",
"ecology": "satisfactory",
"product_type": "investment",
"full_sq": 39.0,
"kremlin_km": 24.7,
"metro_km_walk": 0.73,
"school_km": 0.74,
"kindergarten_km": 0.07,
"railroad_station_avto_min": 6.27,
"usdrub": 55.6,
"cpi": 490.5
}
```

## Example Response:

``` json
{
"predicted_price_rub": 6500000.0
}
```


## Repository structure
```
.
├── Dockerfile
├── app/
│ ├── predict.py
│ ├── request.py
│ └── train.py
├── models/
│ └── model_artifacts.joblib
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── macro.csv
├── notebooks/
│ └── Sberbank_house_price_prediction.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

## Running the Project Locally


Install dependencies using uv

``` bash
uv sync
```

Start the FastAPI service

``` bash
uvicorn app.predict:app --reload
```

Send a test request

``` bash
python app/request.py
```

## Running with Docker


Build the Docker image

``` bash
docker build -t sberbank-app .
```

Run the container

``` bash
docker run -p 9696:9696 sberbank-app
```

The API will now be available at

```
http://localhost:9696/predict
```


## Notebook Reproducibility


The notebook in `notebooks/` can be rerun in the same environment:

```bash
uv sync
jupyter notebook
```
The notebook is NOT required for deployment. It documents the analysis and modeling process only.

## Summary

This project provides a full ML pipeline:
- Data cleaning and feature engineering
- Statistical testing & feature selection
- Model comparison & hyperparameter tuning
- FastAPI inference service
- Dockerized deployment
- Reproducible environment via uv