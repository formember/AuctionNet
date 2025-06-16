import pandas as pd
data = pd.read_csv("./data/adjust_best_alpha/training_data_feature_cpa.csv")
print(data.loc[0,"action"])
result = []
for day in range(7):
    day_result = []
    for ad in range(48):
        action = data.loc[day*10*48*48+ad*10*48,"action"]
        day_result.append(action)
    result.append(day_result)
with open("./data/adjust_best_alpha/alpha_cpa.txt", "wb") as f:
    f.write(str(result).encode('utf-8'))


