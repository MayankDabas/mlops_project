import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_train, output_test, test_size=0.2, random_state=42):
    """Load, preprocess, and split the dataset."""
    # Load the dataset
    data = pd.read_csv(input_path)
    
    # Separate features and target
    features = data.drop(columns=['Target'])
    target = data['Target']
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=test_size, random_state=random_state
    )
    
    # Save to CSV
    pd.DataFrame(X_train).to_csv(output_train, index=False)
    pd.DataFrame(X_test).to_csv(output_test, index=False)
    
    print(f"Preprocessed data saved to {output_train} and {output_test}")

if __name__ == "__main__":
    preprocess_data(
        input_path="data/california_housing.csv",
        output_train="data/train.csv",
        output_test="data/test.csv"
    )
