
import numpy as np
import pandas as pd
import os
import glob
import random
import math
import torch
expose_rate = {
    0:0.0,
    1:1.0,
    2:0.799860,
    3:0.482163
}
budget_set={0.0: (2900.0, 100.0), 1.0: (4350.0, 70.0), 2.0: (3000.0, 90.0), 3.0: (2400.0, 110.0), 4.0: (4800.0, 60.0), 5.0: (2000.0, 130.0), 6.0: (2050.0, 120.0), 7.0: (3500.0, 80.0), 8.0: (4600.0, 70.0), 9.0: (2000.0, 130.0), 10.0: (2800.0, 100.0), 11.0: (2350.0, 110.0), 12.0: (2050.0, 120.0), 13.0: (2900.0, 90.0), 14.0: (4750.0, 60.0), 15.0: (3450.0, 80.0), 16.0: (2000.0, 130.0), 17.0: (3500.0, 80.0), 18.0: (2200.0, 110.0), 19.0: (2700.0, 100.0), 20.0: (3100.0, 90.0), 21.0: (2100.0, 120.0), 22.0: (4850.0, 60.0), 23.0: (4100.0, 70.0), 24.0: (2000.0, 120.0), 25.0: (4800.0, 60.0), 26.0: (3050.0, 90.0), 27.0: (4250.0, 70.0), 28.0: (2850.0, 100.0), 29.0: (2250.0, 110.0), 30.0: (2000.0, 130.0), 31.0: (3900.0, 80.0), 32.0: (2000.0, 120.0), 33.0: (3250.0, 90.0), 34.0: (4450.0, 70.0), 35.0: (3550.0, 80.0), 36.0: (2700.0, 100.0), 37.0: (2100.0, 110.0), 38.0: (4650.0, 60.0), 39.0: (2000.0, 130.0), 40.0: (3400.0, 90.0), 41.0: (2650.0, 100.0), 42.0: (2300.0, 110.0), 43.0: (4100.0, 80.0), 44.0: (4800.0, 60.0), 45.0: (4450.0, 70.0), 46.0: (2000.0, 130.0), 47.0: (2050.0, 120.0)}
class OnlineEnv:
    
    def reset(self,data,i):
        self.advertiserNumber = i % 48
        self.category = self.advertiserNumber // 8 / 6
        self.budget, self.cpa =  budget_set[float(self.advertiserNumber)]
        self.budget = self.budget / 10
        self.remaining_budget=self.budget
        self.total_reward = 0
        self.timeStepIndex = 0
        self.data = data
        self.row = self.data[(self.data['advertiserNumber'] == self.advertiserNumber) & 
                                 (self.data['timeStepIndex'] == self.timeStepIndex)]
        self.pValue = self.row['pValue'].values
        self.pValueSigma = self.row['pValueSigma'].values
        self.total_Conversion = 0
        self.coef = 0
        self.BCR = 0
        self.CPM = 0
        self.WR = 0
        self.avg_convention = 0
        self.historyPValueInfo, self.historyBid,self.historyAuctionResult, self.historyImpressionResult, self.historyLeastWinningCost= [],[],[],[],[]
        
        return self.trans_observation()
        
    def trans_observation(self):
        time_left = (48 - self.timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
#         history_xi = [result[:, 0] for result in self.historyAuctionResult]
#         history_pValue = [result[:, 0] for result in self.historyPValueInfo]
#         history_conversion = [result[:, 1] for result in self.historyImpressionResult]

#         historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

#         historical_conversion_mean = np.mean(
#             [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

#         historical_LeastWinningCost_mean = np.mean(
#             [np.mean(price) for price in self.historyLeastWinningCost]) if self.historyLeastWinningCost else 0

#         historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

#         historical_bid_mean = np.mean([np.mean(bid) for bid in self.historyBid]) if self.historyBid else 0

#         def mean_of_last_n_elements(history, n):
#             last_three_data = history[max(0, n - 3):n]
#             if len(last_three_data) == 0:
#                 return 0
#             else:
#                 return np.mean([np.mean(data) for data in last_three_data])

#         last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
#         last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
#         last_three_LeastWinningCost_mean = mean_of_last_n_elements(self.historyLeastWinningCost, 3)
#         last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
#         last_three_bid_mean = mean_of_last_n_elements(self.historyBid, 3)
        
        current_pValues_mean = np.mean(self.pValue)
        
        test_state = np.array([
            self.category, time_left, budget_left, self.coef, current_pValues_mean,
            # historical_bid_mean, last_three_bid_mean,
            # historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            # historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            # last_three_conversion_mean, last_three_xi_mean,
            # current_pValues_mean
        ])
        test_state = torch.tensor(test_state, dtype=torch.float)
        
        return test_state
        
        
    def simulate_ad_bidding(self, bids,WinningBid1,WinningBid2,WinningBid3,leastWinningCosts):
        tick_status = bids >= WinningBid3
        tick_slot_1 = bids >= WinningBid1
        tick_slot_2 = bids >= WinningBid2
        tick_slot = tick_slot_1 + tick_slot_2 + tick_status
        tick_slot = (4 - tick_slot) * tick_status
        WinningBid = [WinningBid1,WinningBid2,WinningBid3]
        # 计算获胜成本
        tick_cost = [WinningBid[slot-1][index] if slot!=0 else 0 for index,slot in enumerate(tick_slot)]
        
        # 是否暴露给用户
        tick_exposed = [expose_rate[slot] for slot in tick_slot]
        tick_exposed = np.random.binomial(n=1, p=tick_exposed)
        
        values = np.random.normal(loc=self.pValue, scale=self.pValueSigma)
        values = values*tick_exposed
        tick_value = np.clip(values,0,1)
        
        tick_conversion = np.random.binomial(n=1, p=tick_value)
        
        return tick_exposed,tick_status, tick_cost, tick_slot,tick_conversion
    
    def simulate_ad_bidding1(self, bids: np.ndarray, leastWinningCosts: np.ndarray):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        """
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=self.pValue, scale=self.pValueSigma)
        values = values*tick_status
        tick_value = np.clip(values,0,1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status,tick_conversion
    
    
    
    # def step1(self,alpha):
        
    #     bids = alpha * (self.pValue - self.pValueSigma)
    #     termination = 0
    #     leastWinningCosts = self.row['leastWinningCost'].values
        
    #     tick_value, tick_cost, tick_status,tick_conversion = self.simulate_ad_bidding1(bids,leastWinningCosts)
    #     real_cost = np.sum(tick_cost)
    #     over_cost_ratio = max((real_cost - self.remaining_budget) / (real_cost + 1e-4), 0)
    #     while over_cost_ratio > 0:
    #         termination = 1
    #         pv_index = np.where(tick_status == 1)[0]
    #         dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
    #                                             replace=False)
    #         bids[dropped_pv_index] = 0
    #         tick_value, tick_cost, tick_status,tick_conversion = self.simulate_ad_bidding1(bids,leastWinningCosts)
    #         real_cost = np.sum(tick_cost)
    #         over_cost_ratio = max((real_cost - self.remaining_budget) / (real_cost + 1e-4), 0)
    #     self.remaining_budget -= real_cost
        
    #     self.total_Conversion += np.sum(tick_conversion)
        
    #     temHistoryPValueInfo = [(self.pValue[i], self.pValueSigma[i]) for i in range(self.pValue.shape[0])]
    #     self.historyPValueInfo.append(np.array(temHistoryPValueInfo))
    #     self.historyBid.append(bids)
    #     self.historyLeastWinningCost.append(leastWinningCosts)
    #     temAuctionResult = np.array(
    #         [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(len(tick_status))])
    #     self.historyAuctionResult.append(temAuctionResult)
    #     temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(self.pValue.shape[0])])
    #     self.historyImpressionResult.append(temImpressionResult)
        
        
        
    #     #update information
    #     self.timeStepIndex += 1
        
    #     truncation = 1 if self.timeStepIndex == 48 else 0
    #     reward = np.sum(tick_conversion)
    #     curr_CPA = (self.budget-self.remaining_budget) / (self.total_Conversion+1e-10)
    #     self.coef = max((curr_CPA + 1e-10)/ self.cpa,10)
        
            
        
    #     if not termination and not truncation:
    #         self.row = self.data[(self.data['advertiserNumber'] == self.advertiserNumber) & 
    #                                 (self.data['timeStepIndex'] == self.timeStepIndex)]
    #         self.pValue = self.row['pValue'].values
    #         self.pValueSigma = self.row['pValueSigma'].values
            
    #     infos={
            
    #     }
        
    #     if truncation or termination:
    #         infos['episodic_length'] = self.timeStepIndex
    #         if curr_CPA > self.cpa:
    #             rate = self.cpa / (curr_CPA + 1e-10)
    #             penalty = pow(rate, 2)
    #             score = penalty*(self.total_reward+reward)
    #         else:
    #             score = (self.total_reward+reward)
    #         infos['episodic_score'] = score
    #         if curr_CPA > self.cpa:
    #             coef =(curr_CPA + 1e-10)/ self.cpa
    #             penalty = pow(coef, 2)
    #             penalty = min(penalty*2,10)
    #             reward -= penalty
    #         if termination:
    #             reward -= (48 - self.timeStepIndex)
    #         self.total_reward += reward
    #         infos['episodic_return'] = self.total_reward
    #     else:
    #         self.total_reward += reward
    #     # if truncation or termination:
    #     #     infos['episodic_length'] = self.timeStepIndex
    #     #     if curr_CPA > self.cpa:
    #     #         rate = self.cpa / (curr_CPA + 1e-10)
    #     #         penalty = pow(rate, 2)
    #     #         score = penalty*(self.total_reward+reward)
    #     #     else:
    #     #         score = (self.total_reward+reward)
    #     #     infos['episodic_return'] = self.total_reward+reward
    #     #     reward = score
    #     #     infos['episodic_score'] = score
    #     # else:
    #     #     self.total_reward += reward
    #     #     reward = 0
    #     return self.trans_observation(),reward,termination,truncation,infos
    
    def step(self,alpha):
        
        # bids = alpha * (self.pValue - self.pValueSigma) 
        bids = alpha * self.pValue
        termination = 0
        WinningBid1 = self.row['1WinningBid'].values
        WinningBid2 = self.row['2WinningBid'].values
        WinningBid3 = self.row['3WinningBid'].values
        leastWinningCosts = self.row['leastWinningCost'].values
        
        tick_exposed, tick_status, tick_cost, tick_slot,tick_conversion = self.simulate_ad_bidding(bids,WinningBid1,WinningBid2,WinningBid3,leastWinningCosts)
        real_cost = np.sum(tick_cost*tick_exposed)
        over_cost_ratio = max((real_cost - self.remaining_budget) / (real_cost + 1e-4), 0)
        while over_cost_ratio > 0:
            termination = 1
            pv_index = np.where(tick_exposed == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bids[dropped_pv_index] = 0
            tick_exposed, tick_status, tick_cost, tick_slot,tick_conversion = self.simulate_ad_bidding(bids,WinningBid1,WinningBid2,WinningBid3,leastWinningCosts)
            real_cost = np.sum(tick_cost*tick_exposed)
            over_cost_ratio = max((real_cost - self.remaining_budget) / (real_cost + 1e-4), 0)
        self.BCR = real_cost / self.remaining_budget
        self.CPM = real_cost / np.sum(tick_exposed) * 1000 if np.sum(tick_exposed) > 0 else 0
        self.WR = np.mean(tick_exposed)
        self.avg_convention = np.mean(tick_conversion)
        
        self.remaining_budget -= real_cost
        
        self.total_Conversion += np.sum(tick_conversion)
        
        temHistoryPValueInfo = [(self.pValue[i], self.pValueSigma[i]) for i in range(self.pValue.shape[0])]
        self.historyPValueInfo.append(np.array(temHistoryPValueInfo))
        self.historyBid.append(bids)
        self.historyLeastWinningCost.append(leastWinningCosts)
        temAuctionResult = np.array(
            [(tick_status[i], tick_slot[i], tick_cost[i]) for i in range(len(tick_status))])
        self.historyAuctionResult.append(temAuctionResult)
        temImpressionResult = np.array([(tick_exposed[i], tick_conversion[i]) for i in range(self.pValue.shape[0])])
        self.historyImpressionResult.append(temImpressionResult)
        
        
        
        #update information
        self.timeStepIndex += 1
        
        truncation = 1 if self.timeStepIndex == 48 else 0
        #continous reward
        reward = np.sum(tick_exposed*self.pValue)
        
        #reward 
        # reward = np.sum(tick_conversion)
        curr_CPA = (self.budget-self.remaining_budget) / (self.total_Conversion+1e-10)
        
        
        if not termination and not truncation:
            self.row = self.data[(self.data['advertiserNumber'] == self.advertiserNumber) & 
                                    (self.data['timeStepIndex'] == self.timeStepIndex)]
            self.pValue = self.row['pValue'].values
            self.pValueSigma = self.row['pValueSigma'].values
            self.coef = min((curr_CPA + 1e-10)/ self.cpa,10) / 10 if self.total_Conversion!=0 else 0
            
        infos={
            
        }
        # self.total_reward += reward
        # infos['episodic_length'] = self.timeStepIndex
        # infos['episodic_score'] = self.total_reward
        # infos['episodic_return'] = self.total_reward
        if truncation or termination:
            infos['episodic_length'] = self.timeStepIndex
            if curr_CPA > self.cpa:
                rate = self.cpa / (curr_CPA + 1e-10)
                penalty = pow(rate, 2)
                score = penalty*(self.total_reward+reward)
            else:
                score = (self.total_reward+reward)
            infos['episodic_score'] = score
            if curr_CPA > self.cpa:
                coef =(curr_CPA + 1e-10)/ self.cpa
                penalty = pow(coef, 2)
                penalty = min(penalty*2,10)
                reward -= penalty
            # if termination:
            #     reward -= (48 - self.timeStepIndex)
            self.total_reward += reward
            infos['episodic_return'] = self.total_reward
        else:
            self.total_reward += reward
        # if truncation or termination:
        #     infos['episodic_length'] = self.timeStepIndex
        #     if curr_CPA > self.cpa:
        #         rate = self.cpa / (curr_CPA + 1e-10)
        #         penalty = pow(rate, 2)
        #         score = penalty*(self.total_reward+reward)
        #     else:
        #         score = (self.total_reward+reward)
        #     infos['episodic_return'] = self.total_reward+reward
        #     reward = score
        #     infos['episodic_score'] = score
        # else:
        #     self.total_reward += reward
        #     reward = 0
        return self.trans_observation(),reward,termination,truncation,infos
        
        
