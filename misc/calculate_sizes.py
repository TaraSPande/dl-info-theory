import numpy as np
import pandas as pd

# df = pd.read_csv('data/task_1_classification.csv')
# print(len(df))

def calculate_tsizes(n_row, reps=10, min_val=6, max_val=None):
    if max_val is None:
        max_val = 0.9 * n_row

    up = np.log(max_val) - np.log(min_val)

    vals = np.arange(reps) / (reps - 1) * up + np.log(min_val)
    vals = np.exp(vals)
    vals = np.round(vals)

    return vals.astype(int)

tsizes = sorted(set(calculate_tsizes(50318)))

task = "classification"
model = "transformer"
attention = "vanilla"

for ts in tsizes:
    print(f"python main.py train \
  --task {task} \
  --model {model} \
  --csv data/task_1_{task}.csv \
  --features V1,V2,V3,V4,V5 \
  --label Y \
  --max-train-samples {ts} \
  --attn-self-enc {attention}")

    print(f"python main.py eval \
  --run ./runs/{task}-{model}-task_1_{task}-enc6d512h8-{attention}_n{ts} \
  --max-eval-samples 100")
