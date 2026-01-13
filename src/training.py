import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from preprocessing import preprocess

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/validation.csv")

y_train = train_df["charges"]
y_val = val_df["charges"]

X_train, encoder = preprocess(train_df.drop("charges", axis=1), training=True)
X_val, _ = preprocess(val_df.drop("charges", axis=1), encoder=encoder, training=False)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = mean_squared_error(y_val, preds, squared=False)

print(f"Validation RMSE: {rmse}")

