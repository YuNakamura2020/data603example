import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def main():
    X = np.random.uniform(size=50)
    y = X + np.random.normal(0, 0.05, size=50)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    joblib.dump(model, "fake.joblib")


if __name__ == "__main__":
    main()
