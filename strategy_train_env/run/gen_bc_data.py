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
        data = pd.read_csv(f"./data/splited_data/period-{day}_0.csv")
        for advertiserNumber in range(0, 48):
            print(f"Processing advertiser {advertiserNumber} on day {day}")
            for repeat in range(50):
                env = OfflineEnv()
                data_ad = data[(data["advertiserNumber"] == advertiserNumber)]
                budget = data_ad["budget"].values[0] / 10.0
                cpa = data_ad["CPAConstraint"].values[0]
                remaining_budget = budget
                historyBid = []
                historyAuctionResult = []
                historyImpressionResult = []
                historyLeastWinningCost = []
                historyPValueInfo = []
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
                        last_three_data = history[max(0, n - 3):n]
                        if len(last_three_data) == 0:
                            return 0
                        else:
                            return np.mean([np.mean(data) for data in last_three_data])

                    last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
                    last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
                    last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
                    last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
                    last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

                    current_pv_num = len(now_pValues)*10

                    historical_pv_num_total = sum(len(bids) for bids in historyBid)*10 if historyBid else 0
                    last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
                    last_three_pv_num_total = sum(
                        [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)])*10 if historyBid else 0
                    state = np.array([
                        timeStepIndex / 48.0,
                        budget / 485.0,
                        remaining_budget / budget,
                        np.mean(now_pValues),
                        q10,
                        q25,
                        q40,
                        q55,
                        q70,
                        q85,
                        historical_bid_mean,
                        last_three_bid_mean,
                        historical_LeastWinningCost_mean,
                        historical_pValues_mean,
                        historical_conversion_mean,
                        historical_xi_mean,
                        last_three_LeastWinningCost_mean,
                        last_three_pValues_mean,
                        last_three_conversion_mean,
                        last_three_xi_mean,
                        current_pv_num,
                        last_three_pv_num_total,
                        historical_pv_num_total
                    ])
                    day_training_data_rows.append({
                        "state": tuple(state),
                        "action": best_alpha,
                    })
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
                    historyBid.append(bids)
                    historyAuctionResult.append(np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])]))
                    historyImpressionResult.append(np.array([(tick_conversion[i], tick_conversion[i]) for i in range(tick_conversion.shape[0])]))
                    historyLeastWinningCost.append(np.array([(now_leastWinningCosts[i], now_leastWinningCosts[i]) for i in range(now_leastWinningCosts.shape[0])]))
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
    training_data.to_csv("./data/adjust_best_alpha/training_data_v2.csv", index=False)
    print("Data generation completed.")

if __name__ == "__main__":
    main()