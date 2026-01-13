import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance_dataset.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/validation.csv", index=False)
