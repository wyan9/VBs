import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import numpy as np

### 1.Load data
random_seed = 5052

# Input data file (professionalized path for supplementary material)
file_path = "./data/factors.csv"
data = pd.read_csv(file_path)

# Split into features (X) and target variable (y)
X = data.iloc[:, 1:-1]    # All columns except sample name and target
y = data.iloc[:, -1]      # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Standard deviation of target (used for normalized RMSE)
y_std = y_train.std()


### 2.10-fold cross-validation for model training

print("----- 10-fold Cross-Validation Started -----")

kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)

best_model = None
best_r2 = -np.inf
best_fold = None
cv_results = []

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_seed,
        n_jobs=-1
    )
    rf.fit(X_train_fold, y_train_fold)

    # Validation predictions
    y_val_pred = rf.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_val_pred)

    # Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred)) / y_std

    cv_results.append({"fold": i + 1, "model": rf, "r2": r2, "rmse": rmse})
    print(f"Fold {i + 1}: R² = {r2:.4f}, Normalized RMSE = {rmse:.4f}")

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = rf
        best_fold = i + 1

# Cross-validation summary
mean_r2 = np.mean([res["r2"] for res in cv_results])
mean_rmse = np.mean([res["rmse"] for res in cv_results])
print(f"\nMean R² (CV): {mean_r2:.4f}")
print(f"Mean Normalized RMSE (CV): {mean_rmse:.4f}")
print(f"\nBest Model from Fold {best_fold}, R² = {best_r2:.4f}")


### 3.Evaluate the best model on the test set

y_test_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) / y_std

print(f"\nTest Set R²: {test_r2:.4f}")
print(f"Test Set Normalized RMSE: {test_rmse:.4f}")

# Save model
model_output_path = "./models/optimal model_RF.pkl"
os.makedirs("./models", exist_ok=True)
joblib.dump(best_model, model_output_path)
print(f"\nBest model saved to: {model_output_path}")


### 4.Feature importance ranking

importances = best_model.feature_importances_
features = X.columns

feature_importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n----- Feature Importance Ranking -----")
print(feature_importances_df)

# Select top 11 features for PDP
top_features = feature_importances_df.head(11)['Feature'].tolist()


### 5.Partial Dependence Plot (PDP) generation

pdf_output_path = "./results/partial_dependence_plots.pdf"
os.makedirs("./results", exist_ok=True)

with PdfPages(pdf_output_path) as pdf:
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        pd_results = partial_dependence(best_model, X_train, [feature], grid_resolution=50)
        x_vals = pd_results["grid_values"][0]
        y_vals = pd_results["average"][0]

        ax = axes[i]
        ax.plot(x_vals, y_vals, color='blue', label=f"PDP: {feature}")
        ax.plot(x_vals, y_vals, 'r--', lw=2, label="Fitted Curve")
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Partial Dependence", fontsize=10)
        ax.set_title(f"PDP for {feature}", fontsize=10)
        ax.legend(fontsize=8)

    # Remove empty axes
    for j in range(len(top_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    pdf.savefig(fig)
    plt.show()

print(f"\nPDP PDF saved to: {pdf_output_path}")
