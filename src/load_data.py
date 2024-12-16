import pandas as pd
from sklearn.datasets import fetch_california_housing

def fetch_data():
    """Fetch California Housing dataset and return as a Pandas DataFrame."""
    dataset = fetch_california_housing(as_frame=True)
    data = dataset.data
    target = dataset.target
    data['Target'] = target
    return data

if __name__ == "__main__":
    # Load dataset and save to CSV for versioning
    df = fetch_data()
    df.to_csv('data/california_housing.csv', index=False)
    print("Dataset saved to data/california_housing.csv")

