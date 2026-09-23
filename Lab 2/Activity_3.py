import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("Car Price Prediction.csv")  
print(df.head())
print(df.info())

# 2. Select features (adjust names to match your dataset's actual columns)
features = ["enginesize", "horsepower", "curbweight", "citympg"]
target = "price"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_pred_lin = lin_model.predict(X_test_scaled)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)
print(f"Linear Regression -> RMSE: {rmse_lin:.2f}, R²: {r2_lin:.3f}")

# 4. Polynomial Regression (degree 2, 3, 4)
results = {"Linear": (rmse_lin, r2_lin)}

for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2_poly = r2_score(y_test, y_pred_poly)
    results[f"Degree {degree}"] = (rmse_poly, r2_poly)
    print(f"Polynomial (deg={degree}) -> RMSE: {rmse_poly:.2f}, R²: {r2_poly:.3f}")

# 5. Compare results
results_df = pd.DataFrame(results, index=["RMSE", "R2"]).T
print("\nComparison:\n", results_df)

# 6. Plot fitted curves (using a single feature, e.g. Year, for visualization)
plot_feature = "enginesize"
X_single = df[[plot_feature]].values
y_single = df[target].values

X_single_train, X_single_test, y_single_train, y_single_test = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)

x_range = np.linspace(X_single.min(), X_single.max(), 200).reshape(-1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X_single_train, y_single_train, alpha=0.4, label="Training Data")

for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_single_train)
    model = LinearRegression()
    model.fit(X_poly_train, y_single_train)

    x_range_poly = poly.transform(x_range)
    y_range_pred = model.predict(x_range_poly)

    label = "Linear" if degree == 1 else f"Degree {degree}"
    plt.plot(x_range, y_range_pred, label=label)

plt.xlabel(plot_feature)
plt.ylabel("Price")
plt.title(f"Fitted Curves: Price vs {plot_feature}")
plt.legend()
plt.tight_layout()
plt.savefig("fitted_curves.png")
plt.show()