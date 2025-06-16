import numpy as np 
import pandas as pd
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from linear_solution.linear_best import get_best_alpha
from bidding_train_env.offline_eval.offline_env import OfflineEnv
import math
import multiprocessing

def process_day(day):
        day_training_data_rows = []
        data = pd.read_csv(f"./data/traffic/period-{day}.csv")
        for advertiserNumber in range(0, 48):
            print(f"Processing advertiser {advertiserNumber} on day {day}")
            data_ad = data[(data["advertiserNumber"] == advertiserNumber)]
            cpa = data_ad["CPAConstraint"].values[0]
            historyLeastWinningCost = []
            historyPValueInfo = []
            for timeStepIndex in range(0, 48):
                data_t = data_ad[(data_ad["timeStepIndex"] >= timeStepIndex)]
                pValues = data_t["pValue"].values
                now_pValues = data_ad[data_ad["timeStepIndex"] == timeStepIndex]["pValue"].values
                now_leastWinningCosts = data_ad[data_ad["timeStepIndex"] == timeStepIndex]["leastWinningCost"].values
                leastWinningCosts = data_t["leastWinningCost"].values
                budget_coef = []
                real_cpa = leastWinningCosts / np.where(pValues == 0, 1e-10, pValues)
                sorted_indices = np.argsort(real_cpa)
                sorted_cpa = real_cpa[sorted_indices]  # Get sorted values for direct comparison
                # Find the FIRST position in sorted array where CPA > 300
                split_index = 0
                for i in range(len(sorted_cpa)):
                    if sorted_cpa[i] > 300:
                        split_index = i  # Position where sorted array crosses 300
                        break
                else:
                    split_index = len(sorted_cpa)  # All values <=300
                cumulative_cost = np.cumsum(leastWinningCosts[sorted_indices])
                cumulative_orders = np.cumsum(pValues[sorted_indices])
                max_budget = min(20000,cumulative_cost[split_index-1])
                if max_budget >= 15000:
                    min_budget = 1000
                elif max_budget >= 10000:
                    min_budget = 200
                else:
                    min_budget = 10
                budget_range = np.linspace(min_budget, max_budget, 10)
                for budget in budget_range:
                    best_alpha,_,_ = get_best_alpha(cpa=cpa,budget=budget,leastWinningCosts=leastWinningCosts, pValues=pValues, cumulative_cost=cumulative_cost, cumulative_orders=cumulative_orders, sorted_indices=sorted_indices)
                    budget_coef.append([budget,best_alpha])
                print(budget_coef)
                quantiles = np.percentile(now_pValues, [10, 25, 40, 55, 70, 85])
                now_mean_pValue = np.mean(now_pValues) if now_pValues.size > 0 else 0
                q10, q25, q40, q55, q70, q85 = quantiles
                history_pValue = [result[:, 0] for result in historyPValueInfo]
                def mean_of_last_n_elements(history, n):
                    # 获取最后n个元素，n为0或负数时返回空列表
                    last_n_data = history[-n:] if n > 0 else []
                    if not last_n_data:
                        return 0
                    # 计算每个子列表均值，再整体求平均
                    return np.mean([np.mean(data) for data in last_n_data])
                
                def percentile_of_last_n_elements(history, n, percentile):
                    # 获取最后n个元素，n为0或负数时返回空列表
                    last_n_data = history[-n:] if n > 0 else []
                    if not last_n_data:
                        return 0
                    # 计算每个子列表的百分位数，再整体求平均
                    flattened_data = [item for sublist in last_n_data for item in sublist]
                    return np.percentile(flattened_data, percentile) if flattened_data else 0
                
                historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
                last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
                last_one_pValues_mean = mean_of_last_n_elements(history_pValue, 1)  

                last_three_pecentile10_pvalues = percentile_of_last_n_elements(history_pValue, 3, 10)
                last_one_pecentile10_pvalues = percentile_of_last_n_elements(history_pValue, 1, 10)
                last_all_pecentile10_pvalues = percentile_of_last_n_elements(history_pValue, 0, 10)
                last_three_pecentile90_pvalues = percentile_of_last_n_elements(history_pValue, 3, 90)
                last_one_pecentile90_pvalues = percentile_of_last_n_elements(history_pValue, 1, 90)
                last_all_pecentile90_pvalues = percentile_of_last_n_elements(history_pValue, 0, 90)
                last_three_pecentile50_pvalues = percentile_of_last_n_elements(history_pValue, 3, 50)
                last_one_pecentile50_pvalues = percentile_of_last_n_elements(history_pValue, 1, 50)
                last_all_pecentile50_pvalues = percentile_of_last_n_elements(history_pValue, 0, 50)

                historical_LeastWinningCost_mean = np.mean([np.mean(result) for result in historyLeastWinningCost]) if historyLeastWinningCost else 0
                last_one_leastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 1)
                last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)

                last_one_percentile10_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 1, 10)
                last_three_percentile10_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 3, 10)
                last_all_percentile10_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 0, 10)
                last_one_percentile90_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 1, 90)
                last_three_percentile90_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 3, 90)
                last_all_percentile90_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 0, 90)
                last_one_percentile50_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 1, 50)
                last_three_percentile50_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 3, 50)
                last_all_percentile50_leastWinningCost = percentile_of_last_n_elements(historyLeastWinningCost, 0, 50)
                
                current_pv_num = len(now_pValues)
                historical_pv_num_total = sum(len(value) for value in history_pValue) if history_pValue else 0
                last_three_pv_num_total = sum(
                    [len(history_pValue[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if history_pValue else 0
                last_one_pv_num_total = sum(
                    [len(history_pValue[i]) for i in range(max(0, timeStepIndex - 1), timeStepIndex)]) if history_pValue else 0
                state = np.array([
                    day % 7,  # 0. Day of the week (0-6)
                    advertiserNumber,
                    timeStepIndex,  # 1. Time left in the campaign
                    data_ad["advertiserCategoryIndex"].values[0] / 6.0 if "advertiserCategoryIndex" in data_ad.columns else 0,  # 5. Category of the campaign
                    q10,  # 6. 10th percentile of pValues
                    q25,  # 7. 25th percentile of pValues
                    q40,  # 8. 40th percentile of pValues
                    q55,  # 9. 55th percentile of pValues
                    q70,  # 10. 70th percentile of pValues
                    q85,  # 11. 85th percentile of pValues
                    now_mean_pValue,  # 18. Mean of current pValues
                    last_three_pecentile10_pvalues,
                    last_one_pecentile10_pvalues,
                    last_all_pecentile10_pvalues,
                    last_three_pecentile90_pvalues,
                    last_one_pecentile90_pvalues,
                    last_all_pecentile90_pvalues,
                    last_three_pecentile50_pvalues,
                    last_one_pecentile50_pvalues,
                    last_all_pecentile50_pvalues,
                    historical_pValues_mean,
                    last_three_pValues_mean,
                    last_one_pValues_mean,
                    last_one_percentile10_leastWinningCost,
                    last_three_percentile10_leastWinningCost,
                    last_all_percentile10_leastWinningCost,
                    last_one_percentile90_leastWinningCost,
                    last_three_percentile90_leastWinningCost,
                    last_all_percentile90_leastWinningCost,
                    last_one_percentile50_leastWinningCost,
                    last_three_percentile50_leastWinningCost,
                    last_all_percentile50_leastWinningCost, 
                    historical_LeastWinningCost_mean,  # 27. Mean of historical least winning costs
                    last_one_leastWinningCost_mean,  # 28. Mean of least winning costs from the last step
                    last_three_LeastWinningCost_mean,  # 29. Mean of least winning costs from the last three steps
                    current_pv_num,  # 46. Number of current pvalues
                    historical_pv_num_total,  # 47. Total number of historical pvalues
                    last_three_pv_num_total,  # 48. Total number of pvalues from the last three steps
                    last_one_pv_num_total,  # 49. Total number of pvalues from the last step
                ])
                day_training_data_rows.append({
                    "period": day,
                    "advertiserNumber": advertiserNumber,
                    "timeStepIndex": timeStepIndex,
                    "state": tuple(state),
                    "budget_coef": tuple(budget_coef),
                })
                historyLeastWinningCost.append(now_leastWinningCosts)
                temHistoryPValueInfo = [(now_pValues[i], 0) for i in range(now_pValues.shape[0])]
                historyPValueInfo.append(np.array(temHistoryPValueInfo))
        return day_training_data_rows

def main():
    training_data_rows = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_day, range(7, 14))

    for result in results:
        training_data_rows.extend(result)

    training_data = pd.DataFrame(training_data_rows)
    training_data.to_csv("./data/bspline_data/training_data_bspline_10.csv", index=False)
    print("Data generation completed.")

if __name__ == "__main__":
    main()