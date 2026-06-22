# =============================================================================
# train.py
# Bengaluru House Price Prediction — Comprehensive Modeling & Export Pipeline
# Employs: Linear Regression, Lasso, Decision Trees, Random Forests, XGBoost
# =============================================================================

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import kagglehub
import shutil

print("=" * 60)
print("DOWNLOADING LATEST DATASET VIA KAGGLEHUB...")
print("=" * 60)

# Download to KaggleHub cache
# Download dataset
download_path = kagglehub.dataset_download(
    "sumanbera19/bengaluru-house-price-dataset"
)

# Find the CSV file
csv_file = next(
    f for f in os.listdir(download_path)
    if f.endswith(".csv")
)

# Source and destination paths
src = os.path.join(download_path, csv_file)
dst = os.path.join(os.getcwd(), "bengaluru_house_prices.csv")

# Copy to current working directory
shutil.copy2(src, dst)

print(f"Copied {csv_file} -> {dst}")

# --- STEP 0: INITIAL PREPROCESSING & OUTLIER LOGIC ---
# Pulling the raw historical dataset
df = pd.read_csv('bengaluru_house_prices.csv')

# Drop trivial metrics to isolate core features
df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
df = df.dropna().reset_index(drop=True)

# Feature Extraction: Structural numeric conversion of categorical structural strings
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df.drop(['size'], axis=1, inplace=True)

# Structural numeric parser for 'total_sqft' string fields (Handling space-hyphen formatting)
def convert_sqft_to_num(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        try:
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
        except:
            return np.nan
    try:
        return float(str(x).strip())
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft']).reset_index(drop=True)

# Local Domain-Specific Outlier Removal (Business Logic)
# 1. Cap dimension variations: Property floor plans should logically span at least 300 sqft per room
df = df[~(df.total_sqft / df.bhk < 300)]

# 2. Add derived tracker metric for price grouping normalization
df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

# 3. Micro-market spatial analysis: Filter property entries based on neighborhood mean variance (1 Std Dev)
def remove_pps_outliers(df_in):
    df_out = pd.DataFrame()
    for key, subdf in df_in.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_subdf = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_subdf], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)

# 4. Filter out pricing inversion flaws where properties with less bedrooms outprice superior units locally
def remove_bhk_outliers(df_in):
    exclude_indices = np.array([])
    for location, location_subdf in df_in.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_subdf in location_subdf.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_subdf.price_per_sqft),
                'std': np.std(bhk_subdf.price_per_sqft),
                'count': bhk_subdf.shape[0]
            }
        for bhk, bhk_subdf in location_subdf.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_subdf[bhk_subdf.price_per_sqft < stats['mean']].index.values)
    return df_in.drop(exclude_indices, axis=0)

df = remove_bhk_outliers(df)

# 5. Plumbing structure sanity validation
df = df[df.bath < df.bhk + 2]

# --- STEP 1: CARDINALITY MANIPULATION & MATRIC EXPANSION ---
# Group micro-neighborhood variables with fewer than 10 global records into a general index category
location_stats = df['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Manual structural assignment to pre-encoded variable state matrix prior to partitioning
dummies = pd.get_dummies(df.location, prefix='location', dtype=int)
df_model = pd.concat([df, dummies], axis=1)

# Eliminate original non-numeric target elements safely
df_model.drop(['location', 'price_per_sqft'], axis=1, inplace=True)

# Build features X and target matrix y
X = df_model.drop(columns=["price"])
y = df_model["price"]

print("=" * 60)
print("  DATASET MATRIX PROFILE OVERVIEW")
print("=" * 60)
print(f"  Total Clean Rows  : {len(df_model)}")
print(f"  Total Data Columns: {X.shape[1]}")
print(f"  Price Limits (INR): Min=₹{y.min():.1f}L | Max=₹{y.max():.1f}L | Average=₹{y.mean():.1f}L")

# Partition the arrays into a 80/20 train-test split configuration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- STEP 2: MULTI-MODEL AUTOMATED CROSS-VALIDATION LOOP ---
def evaluate(name, pipeline, X_tr, y_tr, X_te, y_te, cv_scores):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    
    test_r2 = r2_score(y_te, y_pred)
    test_mae = mean_absolute_error(y_te, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    
    print(f"\nEvaluating: {name} completed.")
    return pipeline, {
        "model": name,
        "cv_mean_r2": round(cv_scores.mean(), 4),
        "cv_std_r2": round(cv_scores.std(), 4),
        "test_r2": round(test_r2, 4),
        "test_mae": round(test_mae, 2),
        "test_rmse": round(test_rmse, 2)
    }

# Instantiating custom models matching different geometric algorithms
# 1. Linear Regression Baseline
linear_pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
cv_linear = cross_val_score(linear_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
linear_pipeline, linear_res = evaluate("Linear Regression", linear_pipeline, X_train, y_train, X_test, y_test, cv_linear)

# 2. Lasso L1 Model
lasso_pipeline = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000))])
cv_lasso = cross_val_score(lasso_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
lasso_pipeline, lasso_res = evaluate("Lasso Regression", lasso_pipeline, X_train, y_train, X_test, y_test, cv_lasso)

# 3. Decision Tree Regressor
dt_pipeline = Pipeline([("model", DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42))])
cv_dt = cross_val_score(dt_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
dt_pipeline, dt_res = evaluate("Decision Tree Regressor", dt_pipeline, X_train, y_train, X_test, y_test, cv_dt)

# 4. Parallel Ensemble: Random Forest
rf_pipeline = Pipeline([("model", RandomForestRegressor(n_estimators=100, max_depth=12, max_features="sqrt", random_state=42, n_jobs=-1))])
cv_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
rf_pipeline, rf_res = evaluate("Random Forest Regressor", rf_pipeline, X_train, y_train, X_test, y_test, cv_rf)

# 5. Sequential Ensemble: XGBoost
xgb_pipeline = Pipeline([("model", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42, n_jobs=-1))])
cv_xgb = cross_val_score(xgb_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
xgb_pipeline, xgb_res = evaluate("XGBoost Regressor", xgb_pipeline, X_train, y_train, X_test, y_test, cv_xgb)

# --- STEP 3: SUMMARY GENERATION & LEADERBOARD COMPILATION ---
all_results = [linear_res, lasso_res, dt_res, rf_res, xgb_res]
summary = pd.DataFrame(all_results).set_index("model")

print("\n" + "=" * 80)
print("  FINAL COMPARISON MATRIX LEADERBOARD")
print("=" * 80)
print(summary.to_string())
print("=" * 80)

# Programmatically query the index of the highest test-performing array score
best_model_name = summary["test_r2"].idxmax()
print(f"  🏆 Selected Production Model: {best_model_name}")

pipeline_mapping = {
    "Linear Regression": linear_pipeline,
    "Lasso Regression": lasso_pipeline,
    "Decision Tree Regressor": dt_pipeline,
    "Random Forest Regressor": rf_pipeline,
    "XGBoost Regressor": xgb_pipeline,
}
best_pipeline = pipeline_mapping[best_model_name]

# --- STEP 4: MLOPS ARTIFACT PRODUCTION EXPORT ---
# Compile unique location drop-down labels directly from structural matrix features
unique_locations = sorted([c.replace("location_", "") for c in X.columns if str(c).startswith("location_")])

artifact = {
    "pipeline": best_pipeline,
    "locations": unique_locations,
    "model_name": best_model_name,
    "test_r2": float(summary.loc[best_model_name, "test_r2"]),
    "test_mae": float(summary.loc[best_model_name, "test_mae"]),
    "feature_columns": list(X.columns)
}

with open("bengaluru_price_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print(f"\n  ✅ Successfully exported → bengaluru_price_model.pkl ready for app.py!")