import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


# 1. Load dataset
data = pd.read_csv("Car Price Prediction.csv")

print(data.head())
print(data.columns)


# 2. Select features and target
X = data[["horsepower", "enginesize", "citympg", "highwaympg"]]
y = data["price"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Linear Regression
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

r2_linear = r2_score(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

print("\nLinear Regression")
print("R2 Score:", r2_linear)
print("RMSE:", rmse_linear)


# 5. Polynomial Regression
degrees = [2, 3, 4]

for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nPolynomial Regression Degree", degree)
    print("R2 Score:", r2)
    print("RMSE:", rmse)


# 6. Plot fitted curves
# We use Horsepower as the x-axis for visualization.

X_single = data[["horsepower"]]
y_single = data["price"]

X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    X_single,
    y_single,
    test_size=0.2,
    random_state=42
)

plt.figure(figsize=(10, 6))

# Scatter plot of actual data
plt.scatter(
    X_test_single,
    y_test_single,
    label="Actual Data"
)


# Linear Regression curve
linear_single = LinearRegression()
linear_single.fit(X_train_single, y_train_single)

x_range = np.linspace(
    X_single["horsepower"].min(),
    X_single["horsepower"].max(),
    200
).reshape(-1, 1)

y_linear_curve = linear_single.predict(x_range)

plt.plot(
    x_range,
    y_linear_curve,
    label="Linear Regression"
)


# Polynomial curves
for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train_single)

    model = LinearRegression()

    model.fit(X_train_poly, y_train_single)

    x_poly = poly.transform(x_range)

    y_curve = model.predict(x_poly)

    plt.plot(
        x_range,
        y_curve,
        label=f"Polynomial Degree {degree}"
    )


plt.xlabel("Horsepower")
plt.ylabel("Car Price")
plt.title("Linear and Polynomial Regression")
plt.legend()
plt.show()