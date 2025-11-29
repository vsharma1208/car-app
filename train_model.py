import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("\n=== LOADING DATA ===")
df = pd.read_csv("car_price_prediction.csv")

print("Raw shape:", df.shape)
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())

# --- NUMERIC CLEANUP ---
numeric_cols = [
    "Price", "Levy", "Prod. year", "Engine volume",
    "Mileage", "Cylinders", "Airbags"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"⚠️ Missing column: {col}")

print("\nAfter numeric cleanup:", df.shape)

# Drop rows with missing price or mileage
df = df.dropna(subset=["Price", "Mileage"])
print("After dropping missing price/mileage:", df.shape)

# --------- FEATURES ---------
target = "Price"
X = df.drop(columns=[target])
y = df[target]

# Identify categoricals
cat_cols = X.select_dtypes(include="object").columns.tolist()

# Fill only categorical values
X[cat_cols] = X[cat_cols].fillna("Unknown")

print("\nFinal shape before training:", X.shape)
print("Categorical columns:", cat_cols)

if X.shape[0] == 0:
    raise ValueError("❌ Dataset empty after cleaning — check CSV format!")

# --- TRAIN TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- MODEL ---
model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=800,
    loss_function="MAE",
    verbose=False,
    random_seed=42
)

print("\n=== TRAINING MODEL ===")
model.fit(X_train, y_train, cat_features=cat_cols)

# --- EVAL ---
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"\n🔥 MAE: ${mae:,.2f}")

# --- SAVE ---
model.save_model("car_price_model.cbm")
print("💾 Saved model: car_price_model.cbm")