import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Example: Fill missing values
    data.fillna(method='ffill', inplace=True)
    return data

if __name__ == "__main__":
    raw_data = load_data("../data/raw/dataset.csv")
    processed_data = preprocess_data(raw_data)
    processed_data.to_csv("../data/processed/cleaned_dataset.csv", index=False)
