import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trans_type = "_dense"
model_task = f"transformer{trans_type}_regression"

# load json
with open("../results/"+model_task+".json", "r") as f:
    data = json.load(f)

rows = []




#REGRESSION
for entry in data:
    run_dir = entry["run_dir"]
    rmse = entry["metrics"]["rmse"]
    model_family = entry["model_family"]

    # extract the number after n
    n = int(re.search(r'n(\d+)', run_dir).group(1))

    rows.append({"n_train": n, "rmse": rmse, "model": f"{model_family}{trans_type}"})

#CLASSIFICATION
# for entry in data:
#     run_dir = entry["run_dir"]
#     acc = entry["metrics"]["acc"]
#     model_family = entry["model_family"]

#     # extract the number after n
#     n = int(re.search(r'n(\d+)', run_dir).group(1))

#     rows.append({"n_train": n, "accuracy": acc, "model": f"{model_family}{trans_type}"})




df = pd.DataFrame(rows)

# sort so the line graph is ordered
df = df.sort_values("n_train")


df.to_csv("../results/"+model_task+".csv", index=False)