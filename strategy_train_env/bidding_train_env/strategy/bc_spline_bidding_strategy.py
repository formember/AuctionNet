import numpy as np
import torch
import pickle
import os

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

from bidding_train_env.baseline.bc.bc_spline import BC_SPLINE

class BcBiddingStrategy(BaseBiddingStrategy):
    """
    Behavioral Cloning (bc) Strategy
    """

    def __init__(self, budget=100, name="Bc-Spline-k3-PlayerStrategy", cpa=2, category=1,day=7, id=0):
        super().__init__(budget, name, cpa, category)
        self.day = day
        self.advertiserNumber = id
        self.Category = category
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "bc_bspline_k3", "bc.pkl")
        dict_path = os.path.join(dir_name, "saved_model", "bc_bspline_k3", "normalize_dict.pkl")
        self.model = BC_SPLINE()
        self.model.load_state_dict(torch.load(model_path))

        with open(dict_path, 'rb') as f:
            self.normalize_dict = pickle.load(f)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        quantiles = np.percentile(pValues, [10, 25, 40, 55, 70, 85])
        now_mean_pValue = np.mean(pValues) if pValues.size > 0 else 0
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
        
        current_pv_num = len(pValues)
        historical_pv_num_total = sum(len(value) for value in history_pValue) if history_pValue else 0
        last_three_pv_num_total = sum(
            [len(history_pValue[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if history_pValue else 0
        last_one_pv_num_total = sum(
            [len(history_pValue[i]) for i in range(max(0, timeStepIndex - 1), timeStepIndex)]) if history_pValue else 0
        test_state = np.array([
            self.day % 7,  # 0. Day of the week (0-6)
            self.advertiserNumber,
            timeStepIndex,  # 1. Time left in the campaign
            self.Category,  # 2. Category of the advertiser
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

        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        for key, value in self.normalize_dict.items():
            test_state[key] = normalize(test_state[key], value["min"], value["max"])

        test_state = torch.tensor(test_state, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        
        budget =torch.tensor([self.remaining_budget / 20000.0], dtype=torch.float).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            self.model.eval()
            alpha = self.model(test_state,budget)
        alpha = alpha.cpu().numpy() * 300.0
        print(alpha)
        bids = alpha[0][0] * pValues

        return bids