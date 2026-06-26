import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Settings
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20  
DATA_PATH = "train.csv"
# -------------------------

# Loading data (Silent)
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

# Drop leakage columns
if "casual" in df.columns and "registered" in df.columns:
    df = df.drop(columns=["casual", "registered"])

# Feature engineering
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

feature_cols = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "dayofweek", "month", "year"
]

X_all = df[feature_cols].copy()
y_all = df["count"].values

# Random train/test split
n = len(df)
np.random.seed(RANDOM_STATE)
perm = np.random.permutation(n)
split_idx = int((1 - TEST_SIZE) * n)
train_idx = perm[:split_idx]
test_idx = perm[split_idx:]

X_train_df = X_all.iloc[train_idx].reset_index(drop=True)
X_test_df = X_all.iloc[test_idx].reset_index(drop=True)
y_train = y_all[train_idx]
y_test = y_all[test_idx]

# Preprocessing
num_feats = ["temp", "atemp", "humidity", "windspeed"]
cat_feats = ["season", "holiday", "workingday", "weather", "hour", "dayofweek", "month", "year"]

# Numeric scaling
num_means = X_train_df[num_feats].mean(axis=0).values
num_stds = X_train_df[num_feats].std(axis=0).values
num_stds[num_stds == 0] = 1.0

X_train_num = (X_train_df[num_feats].values - num_means) / num_stds
X_test_num = (X_test_df[num_feats].values - num_means) / num_stds

# One-hot encoding
train_cat = pd.get_dummies(X_train_df[cat_feats], columns=cat_feats)
test_cat = pd.get_dummies(X_test_df[cat_feats], columns=cat_feats)
test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

X_train = np.concatenate([X_train_num, train_cat.values], axis=1)
X_test = np.concatenate([X_test_num, test_cat.values], axis=1)

# Helper functions
def add_intercept(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

def solve_least_squares(X_design, y):
    try:
        w = np.linalg.lstsq(X_design, y, rcond=None)[0]
        return w
    except:
        return np.linalg.pinv(X_design) @ y

def predict(X_design, w):
    return X_design @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def evaluate(y_true, y_pred):
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return np.inf, -np.inf
    m = mse(y_true, y_pred)
    r = r2_score(y_true, y_pred)
    return m, r

results = {}

# MODEL 1: Linear Regression
X_train_design = add_intercept(X_train)
X_test_design = add_intercept(X_test)
w_linear = solve_least_squares(X_train_design, y_train)
pred_linear = predict(X_test_design, w_linear)
results["Linear Regression (baseline)"] = evaluate(y_test, pred_linear)

# Polynomial features WITHOUT interactions
def add_polynomial_features_numeric(X_num, degree):
    parts = [X_num]
    for d in range(2, degree + 1):
        parts.append(X_num ** d)
    return np.concatenate(parts, axis=1)

# MODELS 2-4: Polynomial (no interactions)
for deg in [2, 3, 4]:
    poly_train_num = add_polynomial_features_numeric(X_train_num, deg)
    poly_test_num  = add_polynomial_features_numeric(X_test_num,  deg)
    
    poly_mean = poly_train_num.mean(axis=0)
    poly_std  = poly_train_num.std(axis=0)
    poly_std[poly_std == 0] = 1.0
    
    poly_train_num_scaled = (poly_train_num - poly_mean) / poly_std
    poly_test_num_scaled  = (poly_test_num  - poly_mean) / poly_std
    
    X_train_poly = np.concatenate([poly_train_num_scaled, train_cat.values], axis=1)
    X_test_poly  = np.concatenate([poly_test_num_scaled,  test_cat.values],  axis=1)
    
    X_train_poly_design = add_intercept(X_train_poly)
    X_test_poly_design  = add_intercept(X_test_poly)
    
    w_poly = solve_least_squares(X_train_poly_design, y_train)
    pred_poly = predict(X_test_poly_design, w_poly)
    
    results[f"Polynomial d={deg}"] = evaluate(y_test, pred_poly)


# MODEL 5: Quadratic WITH interactions
def add_quadratic_interactions_numeric(X_num):
    n_samples, n_features = X_num.shape
    parts = []
    for i in range(n_features):
        for j in range(i, n_features):
            parts.append((X_num[:, i] * X_num[:, j]).reshape(-1, 1))
    if len(parts) == 0:
        return np.zeros((n_samples, 0))
    return np.concatenate(parts, axis=1)

X_train_inter = add_quadratic_interactions_numeric(X_train_num)
X_test_inter  = add_quadratic_interactions_numeric(X_test_num)

inter_mean = X_train_inter.mean(axis=0) if X_train_inter.shape[1] > 0 else np.array([])
inter_std  = X_train_inter.std(axis=0)  if X_train_inter.shape[1] > 0 else np.array([])
if X_train_inter.shape[1] > 0:
    inter_std[inter_std == 0] = 1.0
    X_train_inter_scaled = (X_train_inter - inter_mean) / inter_std
    X_test_inter_scaled  = (X_test_inter  - inter_mean) / inter_std
else:
    X_train_inter_scaled = X_train_inter
    X_test_inter_scaled  = X_test_inter

X_train_quad = np.concatenate([X_train_num, X_train_inter_scaled, train_cat.values], axis=1)
X_test_quad  = np.concatenate([X_test_num,  X_test_inter_scaled,  test_cat.values],  axis=1)

X_train_quad_design = add_intercept(X_train_quad)
X_test_quad_design  = add_intercept(X_test_quad)

w_quad = solve_least_squares(X_train_quad_design, y_train)
pred_quad = predict(X_test_quad_design, w_quad)

results["Quadratic with Interactions"] = evaluate(y_test, pred_quad)


# -------------------------
# FINAL OUTPUT
# -------------------------
print("\nFinal Results:")

# Create DataFrame
summary_df = pd.DataFrame.from_dict(
    results, orient="index", columns=["TestMSE", "TestR2"]
)
summary_df.index.name = 'Model'
summary_df = summary_df.reset_index()

# Map to display names
display_map = {
    "Linear Regression (baseline)": "1. Linear Regression (baseline)",
    "Polynomial d=2": "2. Polynomial d=2",
    "Polynomial d=3": "2. Polynomial d=3",
    "Polynomial d=4": "2. Polynomial d=4",
    "Quadratic with Interactions": "3. Quadratic with Interactions",
}
summary_df['Model'] = summary_df['Model'].apply(lambda x: display_map.get(x, x))

# Enforce order
order = [
    "1. Linear Regression (baseline)", 
    "2. Polynomial d=2", 
    "2. Polynomial d=3", 
    "2. Polynomial d=4", 
    "3. Quadratic with Interactions"
]
summary_df = summary_df.set_index('Model').reindex(order).reset_index()

# Format strings
summary_df.index = summary_df.index.map(str)
summary_df['TestMSE'] = summary_df['TestMSE'].apply(lambda x: f"{x:.6f}")
summary_df['TestR2'] = summary_df['TestR2'].apply(lambda x: f"{x:.6f}")

# Print table
print(f"{'':<3} {'Model':<40} {'TestMSE':<16} {'TestR2':<8}")
for idx, row in summary_df.iterrows():
    print(f"{idx:<3} {row['Model']:<40} {row['TestMSE']:<16} {row['TestR2']:<8}")
