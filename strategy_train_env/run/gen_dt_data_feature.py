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
    data = pd.read_csv(f"./data/splited_data/period-{day}_0.csv")
    training_data_rows = []
    num_timeStepIndex = 48
    for advertiserNumber in range(0, 48):
        print(f"Processing advertiser {advertiserNumber} on day {day}")
        for repeat in range(10):
            env = OfflineEnv()
            data_ad = data[(data["advertiserNumber"] == advertiserNumber)]
            budget = data_ad["budget"].values[0] / 10.0
            cpa = data_ad["CPAConstraint"].values[0]
            catgory = data_ad["advertiserCategoryIndex"].values[0]
            remaining_budget = budget
            historyBid = []
            historyAuctionResult = []
            historyImpressionResult = []
            historyLeastWinningCost = []
            historyPValueInfo = []
            states = []
            actions = []
            rewards = np.zeros(num_timeStepIndex)
            reward_continuous = np.zeros(num_timeStepIndex)
            dones = np.zeros(num_timeStepIndex)
            for timeStepIndex in range(0, 48):
                data_t = data_ad[(data_ad["timeStepIndex"] >= timeStepIndex)]
                pValues = data_t["pValue"].values
                leastWinningCosts = data_t["leastWinningCost"].values
                best_alpha = get_best_alpha(cpa = cpa, budget=remaining_budget, pValues=pValues, leastWinningCosts=leastWinningCosts)
                now_pValues = data_ad[data_ad["timeStepIndex"] == timeStepIndex]["pValue"].values
                now_leastWinningCosts = data_ad[data_ad["timeStepIndex"] == timeStepIndex]["leastWinningCost"].values
                quantiles = np.percentile(now_pValues, [10, 25, 40, 55, 70, 85])
                q10, q25, q40, q55, q70, q85 = quantiles
                history_xi = [result[:, 0] for result in historyAuctionResult]
                history_pValue = [result[:, 0] for result in historyPValueInfo]
                history_conversion = [result[:, 1] for result in historyImpressionResult]

                historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
                historical_conversion_mean = np.mean([np.mean(reward) for reward in history_conversion]) if history_conversion else 0
                historical_LeastWinningCost_mean = np.mean([np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
                historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
                historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0


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
                
                last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
                last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
                last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
                last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
                last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

                last_one_pValues_mean = mean_of_last_n_elements(history_pValue, 1)  
                last_one_convension_mean = mean_of_last_n_elements(history_conversion, 1)
                last_one_xi_mean = mean_of_last_n_elements(history_xi, 1)

                
                last_one_bid_mean = mean_of_last_n_elements(historyBid, 1)
                last_one_leastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 1)

                LWC_all_10 = percentile_of_last_n_elements(historyLeastWinningCost, timeStepIndex, 10)
                LWC_last_one_10 = percentile_of_last_n_elements(historyLeastWinningCost, 1, 10)
                LWC_last_three_10 = percentile_of_last_n_elements(historyLeastWinningCost, 3, 10)

                LWC_all_1 = percentile_of_last_n_elements(historyLeastWinningCost, timeStepIndex, 1)
                LWC_last_one_1 = percentile_of_last_n_elements(historyLeastWinningCost, 1, 1)
                LWC_last_three_1 = percentile_of_last_n_elements(historyLeastWinningCost, 3, 1)
                #Mean historical bid over LWC ratio
                

                def mean_over_LWC_ratio(history, historyLeastWinningCost, n):
                    if n <= 0 or not history or not historyLeastWinningCost:
                        return 0
                    
                    # 转换为NumPy数组并截取最后n个元素
                    bids = history[-n:]
                    costs = historyLeastWinningCost[-n:]

                    bids = np.array([item for sublist in bids for item in sublist], dtype=np.float32)
                    costs = np.array([item for sublist in costs for item in sublist], dtype=np.float32)
                    
                    # 向量化过滤和计算
                    valid_mask = (costs > 0) & (bids > 0)
                    ratios = np.divide(bids, costs, where=valid_mask, out=np.zeros_like(bids))
                    
                    return ratios[valid_mask].mean() if np.any(valid_mask) else 0
                
                def percentile_over_LWC_ratio(history, historyLeastWinningCost,n,percentile):
                    if n <= 0 or not history or not historyLeastWinningCost:
                        return 0
                    
                    # 转换为NumPy数组并截取最后n个元素

                    bids = history[-n:]
                    costs = historyLeastWinningCost[-n:]

                    bids = np.array([item for sublist in bids for item in sublist], dtype=np.float32)
                    costs = np.array([item for sublist in costs for item in sublist], dtype=np.float32)

                    
                    # 向量化过滤和计算
                    valid_mask = (costs > 0) & (bids > 0)
                    ratios = np.divide(bids, costs, where=valid_mask, out=np.zeros_like(bids))
                    ratios = ratios.reshape(-1)
                    # print(ratios.shape)
                    return np.percentile(ratios, percentile) if np.any(ratios) else 0

                last_all_bid_over_LWC_ratio = mean_over_LWC_ratio(historyBid, historyLeastWinningCost, timeStepIndex)
                last_one_bid_over_LWC_ratio = mean_over_LWC_ratio(historyBid, historyLeastWinningCost, 1)
                last_three_bid_over_LWC_ratio = mean_over_LWC_ratio(historyBid, historyLeastWinningCost, 3)

                last_all_pValues_over_LWC_ratio = mean_over_LWC_ratio(history_pValue, historyLeastWinningCost, timeStepIndex)
                last_one_pValues_over_LWC_ratio = mean_over_LWC_ratio(history_pValue, historyLeastWinningCost, 1)
                last_three_pValues_over_LWC_ratio = mean_over_LWC_ratio(history_pValue, historyLeastWinningCost, 3)

                last_all_ratio_90 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, timeStepIndex, 90)
                last_one_ratio_90 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 1, 90)
                last_three_ratio_90 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 3, 90)

                last_all_ratio_99 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, timeStepIndex, 99)
                last_one_ratio_99 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 1, 99)
                last_three_ratio_99 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 3, 99)

                current_pv_num = len(now_pValues)*10

                historical_pv_num_total = sum(len(bids) for bids in historyBid)*10 if historyBid else 0
                last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
                last_three_pv_num_total = sum(
                    [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)])*10 if historyBid else 0
                last_one_pv_num_total = sum(
                    [len(historyBid[i]) for i in range(max(0, timeStepIndex - 1), timeStepIndex)])*10 if historyBid else 0
                state = np.array([
                    timeStepIndex / 48.0,  # 1. Time left in the campaign
                    remaining_budget / budget,  # 2. Budget remaining for the campaign
                    budget / 485.0,  # 3. Total campaign budget
                    cpa / 130.0,  # 4. Current cost-per-acquisition (CPA)
                    data_ad["advertiserCategoryIndex"].values[0] / 6.0 if "advertiserCategoryIndex" in data_ad.columns else 0,  # 5. Category of the campaign
                    historical_bid_mean,  # 6. Mean of historical bids
                    last_one_bid_mean,  # 7. Mean of bids from the last step
                    last_three_bid_mean,  # 8. Mean of bids from the last three steps
                    historical_LeastWinningCost_mean,  # 9. Mean least winning cost (LWC)
                    last_one_leastWinningCost_mean,  # 10. Mean LWC from the last step
                    last_three_LeastWinningCost_mean,  # 11. Mean LWC from the last three steps
                    LWC_all_10,  # 12. 10th percentile of LWC
                    LWC_last_one_10,  # 13. 10th percentile of LWC from the last step
                    LWC_last_three_10,  # 14. 10th percentile of LWC from the last three steps
                    LWC_all_1,  # 15. 1st percentile of LWC
                    LWC_last_one_1,  # 16. 1st percentile of LWC from the last step
                    LWC_last_three_1,  # 17. 1st percentile of LWC from the last three steps
                    historical_pValues_mean,  # 18. Mean conversion probability (pvalue)
                    historical_conversion_mean,  # 19. Mean historical conversion rate
                    historical_xi_mean,  # 20. Mean historical bid success rate
                    last_one_pValues_mean,  # 21. Mean bid success rate from the last three steps
                    last_three_pValues_mean,  # 22. Mean bid success rate from the last step
                    last_one_convension_mean,  # 23. Mean conversion rate from the last step
                    last_three_conversion_mean,  # 24. Mean conversion rate from the last three steps
                    last_one_xi_mean,  # 25. Mean bid success rate from the last step
                    last_three_xi_mean,  # 26. Mean bid success rate from the last three steps
                    0,
                    0,
                    0,
                    (budget - remaining_budget) / timeStepIndex if timeStepIndex !=0 else 0,  # 30. Total cost incurred so far
                    last_all_bid_over_LWC_ratio,  # 31. Mean bid over LWC ratio
                    last_one_bid_over_LWC_ratio,  # 32. Mean bid over LWC ratio from the last step
                    last_three_bid_over_LWC_ratio,  # 33. Mean bid over LWC ratio from the last three steps
                    last_all_pValues_over_LWC_ratio,  # 34. Mean pvalue over LWC ratio
                    last_one_pValues_over_LWC_ratio,  # 35. Mean pvalue over LWC ratio from the last step
                    last_three_pValues_over_LWC_ratio,  # 36. Mean pvalue over LWC ratio from the last three steps
                    last_all_ratio_90,  # 37. 90th percentile of bid over LWC ratio
                    last_one_ratio_90,  # 38. 90th percentile of bid over LWC ratio from the last step
                    last_three_ratio_90,  # 39. 90th percentile of bid over LWC ratio from the last three steps
                    last_all_ratio_99,  # 40. 99th percentile of bid over LWC ratio
                    last_one_ratio_99,  # 41. 99th percentile of bid over LWC ratio from the last step
                    last_three_ratio_99,  # 42. 99th percentile of bid over LWC ratio from the last three steps
                    now_pValues.mean() if len(now_pValues) > 0 else 0,  # 43. Mean of current pvalues
                    np.percentile(now_pValues, 90) if len(now_pValues) > 0 else 0,  # 44. 90th percentile of current pvalues
                    np.percentile(now_pValues, 99) if len(now_pValues) > 0 else 0,  # 45. 99th percentile of current pvalues
                    current_pv_num,  # 46. Number of current pvalues
                    historical_pv_num_total,  # 47. Total number of historical pvalues
                    last_three_pv_num_total,  # 48. Total number of pvalues from the last three steps
                    last_one_pv_num_total,  # 49. Total number of pvalues from the last step
                ])
                states.append(state)
                actions.append(best_alpha)
                adjust_alpha = best_alpha + 60 * (np.random.rand() - 0.5)
                bids = adjust_alpha * now_pValues
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(now_pValues, np.zeros(len(now_pValues)), bids, now_leastWinningCosts)
                over_cost_ratio = max((np.sum(tick_cost) - remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
                while over_cost_ratio > 0:
                    pv_index = np.where(tick_status == 1)[0]
                    dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                        replace=False)
                    bids[dropped_pv_index] = 0
                    tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(now_pValues, np.zeros(len(now_pValues)), bids, now_leastWinningCosts)
                    over_cost_ratio = max((np.sum(tick_cost) - remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
                remaining_budget -= np.sum(tick_cost)
                rewards[timeStepIndex] = np.sum(tick_conversion)
                reward_continuous[timeStepIndex] = np.sum(tick_value)
                historyBid.append(bids)
                historyAuctionResult.append(np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])]))
                historyImpressionResult.append(np.array([(tick_conversion[i], tick_conversion[i]) for i in range(tick_conversion.shape[0])]))
                historyLeastWinningCost.append(now_leastWinningCosts)
                temHistoryPValueInfo = [(now_pValues[i], 0) for i in range(now_pValues.shape[0])]
                historyPValueInfo.append(np.array(temHistoryPValueInfo))
            dones[-1] = 1
            for timeStepIndex in range(num_timeStepIndex):
                    training_data_rows.append({
                        'deliveryPeriodIndex': day,
                        'advertiserNumber': advertiserNumber,
                        'advertiserCategoryIndex': catgory,
                        'budget': budget * 10,
                        'CPAConstraint': cpa,
                        'realAllCost': (budget - remaining_budget) * 10,
                        'realAllConversion': np.sum(rewards) * 10,
                        'timeStepIndex': timeStepIndex,
                        'state': tuple(states[timeStepIndex]),
                        'action': actions[timeStepIndex],
                        'reward': rewards[timeStepIndex]*10,
                        'reward_continuous': reward_continuous[timeStepIndex]*10,
                        'done': dones[timeStepIndex],
                        'next_state': tuple(states[timeStepIndex + 1]) if dones[timeStepIndex] == 0 else np.nan
                    })
    return training_data_rows
def main():
    training_data_rows = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_day, range(7, 14))

    for result in results:
        training_data_rows.extend(result)

    training_data = pd.DataFrame(training_data_rows)
    training_data.to_csv("./data/adjust_best_alpha/training_data_dt_feature.csv", index=False)
    print("Data generation completed.")

if __name__ == "__main__":
    main()