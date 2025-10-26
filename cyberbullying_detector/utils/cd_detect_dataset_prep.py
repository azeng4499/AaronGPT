import pandas as pd

def prepare_cyberbullying_data(
        agg_path, 
        non_agg_path,
        train_ratio=0.80, 
        val_ratio=0.10, 
        test_ratio=0.10
    ):

    df_agg = pd.read_csv(agg_path)
    df_non = pd.read_csv(non_agg_path)

    if "No." in df_agg.columns:
        df_agg = df_agg.drop(columns=["No."])
    if "No." in df_non.columns:
        df_non = df_non.drop(columns=["No."])

    df_agg = df_agg.rename(columns={"Message": "Text"})
    df_non = df_non.rename(columns={"Message": "Text"})

    df_agg["Label"] = 1
    df_non["Label"] = 0

    df = pd.concat([df_agg, df_non], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"Total: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    train_df.to_csv("cyberbullying_detector/data/cyberbullying_train.csv", index=False)
    val_df.to_csv("cyberbullying_detector/data/cyberbullying_validation.csv", index=False)
    test_df.to_csv("cyberbullying_detector/data/cyberbullying_test.csv", index=False)

    print(f"Total: {len(df)} rows")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("Files written to cyberbullying_detector/data/")


# Example Use
prepare_cyberbullying_data(
    agg_path="cyberbullying_detector/data/aggressive_all.csv",
    non_agg_path="cyberbullying_detector/data/non_aggressive_all.csv"
)
