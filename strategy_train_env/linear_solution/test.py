import pandas as pd 
data = pd.read_csv("./data/traffic/training_data_rlData_folder/period-7-rlData.csv")

data["real_cpa"] = data["realAllCost"] / data["realAllConversion"]
import numpy as np
data["score"] = data.apply(
    lambda row: row["realAllConversion"] * pow(min(1, row["CPAConstraint"] / row["real_cpa"]), 2), axis=1
)
print(data["score"].mean())