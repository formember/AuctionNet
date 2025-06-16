import pandas as pd 
data = pd.read_csv("./data/adjust_best_alpha/training_data_dt_feature_without7.csv")

# Check for anomalies in the 'action' column
if 'action' in data.columns:
	anomalies = data[(data['action'] > 200) | (data['action'] < 0)]
	if not anomalies.empty:
		print("Found anomalies in 'action':")
		print(anomalies["action"])
else:
	print("'action' column is not present in the dataset.")
