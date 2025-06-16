import numpy as np
import torch
import pickle
import os

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class BcBiddingStrategy(BaseBiddingStrategy):
    """
    Behavioral Cloning (bc) Strategy
    """

    def __init__(self, day=7,id=0,budget=100, name="Bcaa-feature--coef-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        self.day = day
        self.id = id
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "BC_adjust_alpha_feature_coef", "bc_model.pth")
        dict_path = os.path.join(dir_name, "saved_model", "BC_adjust_alpha_feature_coef", "normalize_dict.pkl")

        self.model = torch.jit.load(model_path)

        with open(dict_path, 'rb') as f:
            self.normalize_dict = pickle.load(f)
        alphas_path = os.path.join(dir_name, "data", "adjust_best_alpha", "alpha_cpa.txt")
        with open(alphas_path, 'r') as f:
            alphas = f.readlines()
            alphas = eval(alphas[0])
            self.alpha = alphas[int(self.day%7)][int(self.id)]

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
        # time = timeStepIndex / 48
        # budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        # history_xi = [result[:, 0] for result in historyAuctionResult]
        # history_pValue = [result[:, 0] for result in historyPValueInfo]
        # history_conversion = [result[:, 1] for result in historyImpressionResult]

        # historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
        # historical_conversion_mean = np.mean([np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        # historical_LeastWinningCost_mean = np.mean([np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        # historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
        # historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0


        # def mean_of_last_n_elements(history, n):
        #     # 获取最后n个元素，n为0或负数时返回空列表
        #     last_n_data = history[-n:] if n > 0 else []
        #     if not last_n_data:
        #         return 0
        #     # 计算每个子列表均值，再整体求平均
        #     return np.mean([np.mean(data) for data in last_n_data])
        
        # def percentile_of_last_n_elements(history, n, percentile):
        #     # 获取最后n个元素，n为0或负数时返回空列表
        #     last_n_data = history[-n:] if n > 0 else []
        #     if not last_n_data:
        #         return 0
        #     # 计算每个子列表的百分位数，再整体求平均
        #     flattened_data = [item for sublist in last_n_data for item in sublist]
        #     return np.percentile(flattened_data, percentile) if flattened_data else 0
        
        # last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        # last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        # last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        # last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        # last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        # last_one_pValues_mean = mean_of_last_n_elements(history_pValue, 1)  
        # last_one_convension_mean = mean_of_last_n_elements(history_conversion, 1)
        # last_one_xi_mean = mean_of_last_n_elements(history_xi, 1)

        
        # last_one_bid_mean = mean_of_last_n_elements(historyBid, 1)
        # last_one_leastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 1)

        # LWC_all_10 = percentile_of_last_n_elements(historyLeastWinningCost, timeStepIndex, 10)
        # LWC_last_one_10 = percentile_of_last_n_elements(historyLeastWinningCost, 1, 10)
        # LWC_last_three_10 = percentile_of_last_n_elements(historyLeastWinningCost, 3, 10)

        # LWC_all_1 = percentile_of_last_n_elements(historyLeastWinningCost, timeStepIndex, 1)
        # LWC_last_one_1 = percentile_of_last_n_elements(historyLeastWinningCost, 1, 1)
        # LWC_last_three_1 = percentile_of_last_n_elements(historyLeastWinningCost, 3, 1)
        # #Mean historical bid over LWC ratio
        

        # def mean_over_LWC_ratio(history, historyLeastWinningCost, n):
        #     if n <= 0 or not history or not historyLeastWinningCost:
        #         return 0
            
        #     # 转换为NumPy数组并截取最后n个元素
        #     bids = history[-n:]
        #     costs = historyLeastWinningCost[-n:]

        #     bids = np.array([item for sublist in bids for item in sublist], dtype=np.float32)
        #     costs = np.array([item for sublist in costs for item in sublist], dtype=np.float32)
            
        #     # 向量化过滤和计算
        #     valid_mask = (costs > 0) & (bids > 0)
        #     ratios = np.divide(bids, costs, where=valid_mask, out=np.zeros_like(bids))
            
        #     return ratios[valid_mask].mean() if np.any(valid_mask) else 0
        
        # def percentile_over_LWC_ratio(history, historyLeastWinningCost,n,percentile):
        #     if n <= 0 or not history or not historyLeastWinningCost:
        #         return 0
            
        #     # 转换为NumPy数组并截取最后n个元素

        #     bids = history[-n:]
        #     costs = historyLeastWinningCost[-n:]

        #     bids = np.array([item for sublist in bids for item in sublist], dtype=np.float32)
        #     costs = np.array([item for sublist in costs for item in sublist], dtype=np.float32)

            
        #     # 向量化过滤和计算
        #     valid_mask = (costs > 0) & (bids > 0)
        #     ratios = np.divide(bids, costs, where=valid_mask, out=np.zeros_like(bids))
        #     ratios = ratios.reshape(-1)
        #     # print(ratios.shape)
        #     return np.percentile(ratios, percentile) if np.any(ratios) else 0

        # last_all_bid_over_LWC_ratio = mean_over_LWC_ratio(historyBid, historyLeastWinningCost, timeStepIndex)
        # last_one_bid_over_LWC_ratio = mean_over_LWC_ratio(historyBid, historyLeastWinningCost, 1)
        # last_three_bid_over_LWC_ratio = mean_over_LWC_ratio(historyBid, historyLeastWinningCost, 3)

        # last_all_pValues_over_LWC_ratio = mean_over_LWC_ratio(history_pValue, historyLeastWinningCost, timeStepIndex)
        # last_one_pValues_over_LWC_ratio = mean_over_LWC_ratio(history_pValue, historyLeastWinningCost, 1)
        # last_three_pValues_over_LWC_ratio = mean_over_LWC_ratio(history_pValue, historyLeastWinningCost, 3)

        # last_all_ratio_90 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, timeStepIndex, 90)
        # last_one_ratio_90 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 1, 90)
        # last_three_ratio_90 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 3, 90)

        # last_all_ratio_99 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, timeStepIndex, 99)
        # last_one_ratio_99 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 1, 99)
        # last_three_ratio_99 = percentile_over_LWC_ratio(history_pValue, historyLeastWinningCost, 3, 99)

        # current_pv_num = len(pValues)

        # historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        # last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        # last_three_pv_num_total = sum(
        #     [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0
        # last_one_pv_num_total = sum(
        #     [len(historyBid[i]) for i in range(max(0, timeStepIndex - 1), timeStepIndex)]) if historyBid else 0
        
        # test_state = np.array([
        #     timeStepIndex / 48.0,  # 1. Time left in the campaign
        #     self.remaining_budget / self.budget,  # 2. Budget remaining for the campaign
        #     self.budget / 4850.0,  # 3. Total campaign budget
        #     self.cpa / 130.0,  # 4. Current cost-per-acquisition (CPA)
        #     self.category / 6.0,  # 5. Category of the campaign
        #     historical_bid_mean,  # 6. Mean of historical bids
        #     last_one_bid_mean,  # 7. Mean of bids from the last step
        #     last_three_bid_mean,  # 8. Mean of bids from the last three steps
        #     historical_LeastWinningCost_mean,  # 9. Mean least winning cost (LWC)
        #     last_one_leastWinningCost_mean,  # 10. Mean LWC from the last step
        #     last_three_LeastWinningCost_mean,  # 11. Mean LWC from the last three steps
        #     LWC_all_10,  # 12. 10th percentile of LWC
        #     LWC_last_one_10,  # 13. 10th percentile of LWC from the last step
        #     LWC_last_three_10,  # 14. 10th percentile of LWC from the last three steps
        #     LWC_all_1,  # 15. 1st percentile of LWC
        #     LWC_last_one_1,  # 16. 1st percentile of LWC from the last step
        #     LWC_last_three_1,  # 17. 1st percentile of LWC from the last three steps
        #     historical_pValues_mean,  # 18. Mean conversion probability (pvalue)
        #     historical_conversion_mean,  # 19. Mean historical conversion rate
        #     historical_xi_mean,  # 20. Mean historical bid success rate
        #     last_one_pValues_mean,  # 21. Mean bid success rate from the last three steps
        #     last_three_pValues_mean,  # 22. Mean bid success rate from the last step
        #     last_one_convension_mean,  # 23. Mean conversion rate from the last step
        #     last_three_conversion_mean,  # 24. Mean conversion rate from the last three steps
        #     last_one_xi_mean,  # 25. Mean bid success rate from the last step
        #     last_three_xi_mean,  # 26. Mean bid success rate from the last three steps
        #     0,
        #     0,
        #     0,
        #     (self.budget - self.remaining_budget) / (timeStepIndex*10) if timeStepIndex !=0 else 0,  # 30. Total cost incurred so far
        #     last_all_bid_over_LWC_ratio,  # 31. Mean bid over LWC ratio
        #     last_one_bid_over_LWC_ratio,  # 32. Mean bid over LWC ratio from the last step
        #     last_three_bid_over_LWC_ratio,  # 33. Mean bid over LWC ratio from the last three steps
        #     last_all_pValues_over_LWC_ratio,  # 34. Mean pvalue over LWC ratio
        #     last_one_pValues_over_LWC_ratio,  # 35. Mean pvalue over LWC ratio from the last step
        #     last_three_pValues_over_LWC_ratio,  # 36. Mean pvalue over LWC ratio from the last three steps
        #     last_all_ratio_90,  # 37. 90th percentile of bid over LWC ratio
        #     last_one_ratio_90,  # 38. 90th percentile of bid over LWC ratio from the last step
        #     last_three_ratio_90,  # 39. 90th percentile of bid over LWC ratio from the last three steps
        #     last_all_ratio_99,  # 40. 99th percentile of bid over LWC ratio
        #     last_one_ratio_99,  # 41. 99th percentile of bid over LWC ratio from the last step
        #     last_three_ratio_99,  # 42. 99th percentile of bid over LWC ratio from the last three steps
        #     pValues.mean() if len(pValues) > 0 else 0,  # 43. Mean of current pvalues
        #     np.percentile(pValues, 90) if len(pValues) > 0 else 0,  # 44. 90th percentile of current pvalues
        #     np.percentile(pValues, 99) if len(pValues) > 0 else 0,  # 45. 99th percentile of current pvalues
        #     current_pv_num,  # 46. Number of current pvalues
        #     historical_pv_num_total,  # 47. Total number of historical pvalues
        #     last_three_pv_num_total,  # 48. Total number of pvalues from the last three steps
        #     last_one_pv_num_total,  # 49. Total number of pvalues from the last step
        # ])
        # # print(test_state)
        # def normalize(value, min_value, max_value):
        #     return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        # for key, value in self.normalize_dict.items():
        #     test_state[key] = normalize(test_state[key], value["min"], value["max"])

        # test_state = torch.tensor(test_state, dtype=torch.float)
        # alpha = self.model(test_state)
        # alpha = alpha.cpu().numpy()
        # if timeStepIndex != 0:
        #     self.alpha = self.alpha * (1+alpha)
        bids = self.alpha * pValues
        print(self.alpha)

        return bids
    

    