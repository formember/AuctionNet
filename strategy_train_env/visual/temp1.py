import pandas as pd


data = pd.read_csv('/Users/pusen.dong/Desktop/python/AuctionNet/strategy_train_env/visual/temp1.csv')

# Calculate the mean of the 'Score' column
score_mean = data['Score'].mean()

print(f"The mean of the 'Score' column is: {score_mean}")