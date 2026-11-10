import pandas as pd

df = pd.read_csv("resume_dataset.csv")
print(df.shape)
print(df["Category"].value_counts())
print(df["Resume"][0][:500])