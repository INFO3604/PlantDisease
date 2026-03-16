import pandas as pd

df = pd.read_csv("data/demo_output/features.csv")

print(df.shape)
print(df.head())
print(df.isna().sum().sum())

constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
print(constant_cols)

print(df.describe())