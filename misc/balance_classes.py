import numpy as np
import pandas as pd

df = pd.read_csv('data/task_1_classification.csv')

min_size = df["Y"].value_counts().min()

balanced_df = (
    df.groupby("Y", group_keys=False)
      .sample(n=min_size, random_state=42)
)

print(balanced_df["Y"].value_counts())

balanced_df.to_csv("task_1_classification_balanced.csv", index=False)