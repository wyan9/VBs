import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump


### Paths (customized for supplementary materials)
data_path = 'data/factors.csv'           # Input feature table
model_save_path = 'results/models'         # Folder to store trained models
output_summary_xlsx = 'results/VB_model_performance.xlsx'
output_cv_csv = 'results/VB_all_CV_results.csv'

os.makedirs(model_save_path, exist_ok=True)
os.makedirs('results', exist_ok=True)


### Load dataset
data = pd.read_csv(data_path)


### Define feature matrix and target variable
# Exclude the first column (Sample ID) and the last column (VB_number)
X = data.iloc[:, 1:-1]
y = data['VB_number']


### Define machine learning models
models = {
    'XGBoost': XGBRegressor(objective='reg:squarederror', verbosity=0),
    'Random Forest': RandomForestRegressor(),
    'Linear Regression': LinearRegression(),
    'K Nearest Neighbors': KNeighborsRegressor()
}


### Hyperparameter search space
param_grids = {
    'XGBoost': {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2]
    },
    'Random Forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    },
    'Linear Regression': {},  # No hyperparameters to tune
    'K Nearest Neighbors': {
        'model__n_neighbors': [3, 5, 7, 10],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    }
}



### Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []
all_cv_results = []


### Train and evaluate each model

for model_name, base_model in models.items():
    print(f"\nTraining model: {model_name} ...")

    # Create a pipeline with scaling + model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])

    # Hyperparameter search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=10,
        scoring='r2',
        refit=True,
        n_jobs=-1
    )

    # Train model
    grid_search.fit(X_train, y_train)

    # Store CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['model_name'] = model_name
    all_cv_results.append(cv_results)

    # Best parameters
    best_params = grid_search.best_params_

    # Evaluate on test set
    y_pred_test = grid_search.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

    # Evaluate on full dataset
    y_pred_full = grid_search.predict(X)
    r2_full = r2_score(y, y_pred_full)

    # Save best model
    safe_name = model_name.replace(" ", "_")
    model_filename = os.path.join(model_save_path, f"{safe_name}_best_model.pkl")
    dump(grid_search.best_estimator_, model_filename)

    # Summarize results
    results.append({
        'Model': model_name,
        'Best Parameters': str(best_params),
        'R2_overall': r2_full,
        'R2_test': r2_test,
        'MAE_test': mae_test,
        'RMSE_test': rmse_test,
        'Model File': model_filename
    })

    print(f"  Finished. R2_test = {r2_test:.4f},  R2_overall = {r2_full:.4f}")


### Save summary results (Excel)
results_df = pd.DataFrame(results)
with pd.ExcelWriter(output_summary_xlsx) as writer:
    results_df.to_excel(writer, index=False)


### Save all CV results (CSV)
big_cv = pd.concat(all_cv_results, ignore_index=True)
big_cv.to_csv(output_cv_csv, index=False)

print("\nAll models have been trained. Results have been saved.")
