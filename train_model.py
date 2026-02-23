import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🏍️  XGBoost Motorcycle Price Prediction")
print("=" * 60)


# 1. DATA LOADING & FEATURE ENGINEERING
print("\n📂 Loading cleaned dataset...")
df = pd.read_csv('ikman_bikes_cleaned.csv')
print(f"   Loaded {len(df)} rows × {len(df.columns)} columns")
print(f"   Columns: {df.columns.tolist()}")

# Create 'age' feature from year of manufacture
current_year = 2026
df['age'] = current_year - df['yom']

# Drop columns not useful for training
df = df.drop(['title', 'yom'], axis=1)

# Label Encode categorical columns (better than one-hot for small datasets + XGBoost)
label_encoders = {}
categorical_cols = ['make', 'model', 'location']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   Encoded '{col}': {len(le.classes_)} unique values → {list(le.classes_)}")

print(f"\n   Final features: {[c for c in df.columns if c != 'price']}")
print(f"   Target: price")


# 2. TRAIN / TEST SPLIT
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set:     {X_test.shape[0]} samples")


# 3. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
print("\n🔧 Hyperparameter Tuning with GridSearchCV (5-Fold CV)...")

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\n   ✅ Best Parameters Found:")
for param, value in best_params.items():
    print(f"      {param}: {value}")
print(f"   Best CV MAE: Rs. {-grid_search.best_score_:,.2f}")


# 4. TRAIN FINAL MODEL WITH BEST PARAMETERS
print("\n🚀 Training Final Model with Best Parameters...")

model = xgb.XGBRegressor(
    **best_params,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# Cross-validation score on full training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"   5-Fold CV MAE: Rs. {-cv_scores.mean():,.2f} (± Rs. {cv_scores.std():,.2f})")


# 5. EVALUATION METRICS
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("📊 MODEL PERFORMANCE METRICS (Test Set)")
print("=" * 50)
print(f"   Mean Absolute Error (MAE):  Rs. {mae:,.2f}")
print(f"   Root Mean Sq Error (RMSE):  Rs. {rmse:,.2f}")
print(f"   R-squared (R² Score):       {r2:.4f}")
print("=" * 50)

# 6. ACTUAL vs PREDICTED SCATTER PLOT
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='#2196F3', edgecolors='k', s=80)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Price (Rs.)', fontsize=12)
plt.ylabel('Predicted Price (Rs.)', fontsize=12)
plt.title('XGBoost: Actual vs Predicted Motorcycle Prices', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
print("\n✅ Saved: actual_vs_predicted.png")
plt.close()

# 7. FEATURE IMPORTANCE 
plt.figure(figsize=(8, 5))
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='#4CAF50')
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx], fontsize=11)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('XGBoost Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300)
print("✅ Saved: xgb_feature_importance.png")
plt.close()


# 8. SHAP EXPLAINABILITY
print("\n🧠 Generating SHAP Explainability Plots...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
print("✅ Saved: shap_summary.png")
plt.close()

# SHAP Bar Plot (Mean |SHAP|)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_feature_importance.png', bbox_inches='tight', dpi=300)
print("✅ Saved: shap_feature_importance.png")
plt.close()

# SHAP Waterfall Plot (Explain first test prediction)
shap_explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist()
)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_explanation, show=False)
plt.tight_layout()
plt.savefig('shap_waterfall.png', bbox_inches='tight', dpi=300)
print("✅ Saved: shap_waterfall.png")
plt.close()


# 9. SAVE MODEL & ENCODERS
joblib.dump(model, 'xgb_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')

print("\n💾 Model saved: xgb_model.joblib")
print("💾 Encoders saved: label_encoders.joblib")
print("💾 Feature names saved: feature_names.joblib")

# 10. SUMMARY
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"   Model:          XGBoost Regressor")
print(f"   Best Params:    {best_params}")
print(f"   Test MAE:       Rs. {mae:,.2f}")
print(f"   Test R²:        {r2:.4f}")
print(f"   Plots saved:    actual_vs_predicted.png")
print(f"                   xgb_feature_importance.png")
print(f"                   shap_summary.png")
print(f"                   shap_feature_importance.png")
print(f"                   shap_waterfall.png")
print("=" * 60)