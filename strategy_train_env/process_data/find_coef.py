import pandas as pd
data = pd.read_csv("./data/adjust_best_alpha/training_data_feature_cpa.csv")
result = data.copy()
for day in range(7):
    for ad in range(48):
        for time in range(10):
            for timestep in range(48):
                if timestep == 0:
                    continue
                index = day * 48 * 10 * 48 + ad * 10 * 48 + time * 48 + timestep
                result.loc[index, "action"] = (data.loc[index , "action"] - data.loc[index - 1, "action"]) / data.loc[index - 1, "action"]
rows = [i for i in range(len(data)) if i%48 == 0]
result = result.drop(rows)
result = result.reset_index(drop=True)
result.to_csv("./data/adjust_best_alpha/training_data_feature_cpa_coef.csv", index=False)


