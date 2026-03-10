import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

model_task = "transformer_vanilla_classification"

# load json
with open("results/"+model_task+".json", "r") as f:
    data = json.load(f)

rows = []

#REGRESSION
# for entry in data:
#     run_dir = entry["run_dir"]
#     rmse = entry["metrics"]["rmse"]
#     model_family = entry["model_family"]

#     # extract the number after n
#     n = int(re.search(r'n(\d+)', run_dir).group(1))

#     rows.append({"n_train": n, "rmse": rmse, "model": model_family})

#CLASSIFICATION
for entry in data:
    run_dir = entry["run_dir"]
    acc = entry["metrics"]["acc"]
    model_family = entry["model_family"]

    # extract the number after n
    n = int(re.search(r'n(\d+)', run_dir).group(1))

    rows.append({"n_train": n, "accuracy": acc, "model": model_family+"_vanilla"})

df = pd.DataFrame(rows)

# sort so the line graph is ordered
df = df.sort_values("n_train")


df.to_csv("results/"+model_task+".csv", index=False)


# x = df["n_train"]
# y = df["rmse"]

# # exponential decay function
# def exp_decay(x, a, b, c):
#     return a * np.exp(-b * x) + c

# # fit parameters
# params, _ = curve_fit(exp_decay, x, y, p0=(max(y), 0.1, min(y)))

# a, b, c = params

# # smooth curve for plotting
# x_fit = np.linspace(min(x), max(x), 200)
# y_fit = exp_decay(x_fit, a, b, c)

# # plot
# plt.scatter(x, y, marker="o")
# plt.plot(x_fit, y_fit)
# plt.xscale('log') # Set the x-axis to a logarithmic scale
# plt.xlabel("Number of Samples")
# plt.ylabel("RMSE")
# plt.title("MLP_Small_Regression: RMSE vs Number of Samples")
# plt.show()