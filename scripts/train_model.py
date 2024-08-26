from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return model

if __name__ == "__main__":
    # Assuming you have a processed dataset
    data = pd.read_csv("../data/processed/cleaned_dataset.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
    model = train_model(X, y)
    joblib.dump(model, "../models/random_forest_model.pkl")
