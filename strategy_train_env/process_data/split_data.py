import os
import pandas as pd
import warnings
import glob
import pickle as pk
from tqdm import tqdm
from multiprocessing import Pool
file_folder_path = "./data/traffic"
training_data_path = "./data/splited_data"
os.makedirs(training_data_path, exist_ok=True)
# csv_files = glob.glob(os.path.join(file_folder_path, '*.csv'))
csv_files = []
for day in range(7,14):
    csv_files.append(os.path.join(file_folder_path, f"period-{day}.csv"))

training_data_list = []
def find_winning_bids(group):
    # 计算每个广告位的最小竞价
    result = {}
    for ad_slot in [1, 2, 3]:
        result[f'{ad_slot}WinningBid'] = group.loc[group['adSlot'] == ad_slot, 'bid'].min()
    return pd.Series(result)

def process_file(csv_path):
    print("开始处理文件：", csv_path)
    df = pd.read_csv(csv_path)
    df_winning_bids = df.groupby(['timeStepIndex', 'pvIndex']).apply(find_winning_bids)
    
    # 将计算的 winning bids 合并回原始数据
    df = df.merge(df_winning_bids, on=['timeStepIndex', 'pvIndex'])
    df.drop(columns=['bid'], inplace=True)
    for i in range(10):
        filtered_df = df[df['pvIndex'] % 10 == i]
        filtered_df.to_csv(os.path.join(training_data_path, os.path.basename(csv_path).rsplit('.', 1)[0] + f'_{i}.csv'))

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(process_file, csv_files)
    