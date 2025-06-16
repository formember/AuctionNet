import pandas as pd

data = pd.read_csv("./data/adjust_best_alpha/training_data_dt_origin_feature.csv")
data = data[data["deliveryPeriodIndex"] != 7]
data.to_csv("./data/adjust_best_alpha/training_data_dt_origin_feature_without7.csv", index=False)



data = pd.read_csv("./data/adjust_best_alpha/training_data_feature.csv")
data = data.drop(range(0,48*10*48))
data.to_csv("./data/adjust_best_alpha/training_data_feature_without7.csv", index=False)