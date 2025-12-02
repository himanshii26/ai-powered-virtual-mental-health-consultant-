# Feature engineering for ML model
import pandas as pd

def build_features(input_file="data/sample.parquet", output_file="data/features.parquet"):
    # Load data
    df = pd.read_parquet(input_file)

    # Sort by timestamp (important)
    df = df.sort_values("timestamp")

    # Rolling CPU mean (smooth signal)
    df["cpu_ma_5"] = (
        df.groupby("pid")["cpu_percent"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Rolling CPU standard deviation (measures instability)
    df["cpu_std_5"] = (
        df.groupby("pid")["cpu_percent"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Memory difference (check leak trends)
    df["mem_diff"] = df.groupby("pid")["mem_rss"].diff().fillna(0)

    # Save output
    df.to_parquet(output_file, index=False)
    print(f"Features saved to {output_file}")


if __name__ == "__main__":
    build_features()
