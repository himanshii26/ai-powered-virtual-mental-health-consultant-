import pandas as pd
from sklearn.ensemble import IsolationForest

def train_model(input_file="data/features.parquet", output_file="data/predictions.parquet"):
    # Load feature data
    df = pd.read_parquet(input_file)

    # Select features for the model
    feature_cols = ["cpu_percent", "cpu_ma_5", "cpu_std_5", "mem_rss", "mem_diff"]
    X = df[feature_cols].fillna(0)

    # Train anomaly detection model
    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,   # 5% anomalies
        random_state=42
    )

    df["anomaly"] = model.fit_predict(X)
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    # Save predictions
    df.to_parquet(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    train_model()
