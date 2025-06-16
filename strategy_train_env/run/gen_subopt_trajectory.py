import numpy as np
import math
import logging
import pandas as pd 
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test():
    """
    offline evaluation
    """
    average_score = 0
    # Add column names to the CSV file
    agent = PlayerBiddingStrategy(budget=0, cpa=0)
    with open(f'{agent.name}-evaluation_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'Id', 'Budget', 'CPAconstraint', 'Total-Reward', 'Total-Cost', 'CPA-Real', 'Score'])
    is_Pacing = True
    for day in range(21,22):
        training_data_rows = []
        solved_data = pd.read_csv(f'./data/pulp_v2/period-{day%7 + 7}.csv')
        states = []
        actions = []
        for i in range(48):
            actions.append([
            solved_data.loc[
                (solved_data['advertiserNumber'] == i) & (solved_data['timeStepIndex'] == j), 
                'alpha'
            ].values[0] 
            for j in range(48)
            ])
        data_loader = TestDataLoader(file_path=f'./data/traffic/period-{day}.csv')
        keys, test_dict = data_loader.keys, data_loader.test_dict
        for key in keys:
            if key[1] != 20:
                continue
            env = OfflineEnv()
            agent = PlayerBiddingStrategy(budget=key[2], cpa=key[3])
            print(key[1])

            
            num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
            rewards = np.zeros(num_timeStepIndex)
            reward_continuous = np.zeros(num_timeStepIndex)
            dones = np.zeros(num_timeStepIndex)
            history = {
                'historyBids': [],
                'historyAuctionResult': [],
                'historyImpressionResult': [],
                'historyLeastWinningCost': [],
                'historyPValueInfo': []
            }

            for timeStep_index in range(num_timeStepIndex):
                # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

                pValue = pValues[timeStep_index]
                pValueSigma = pValueSigmas[timeStep_index]
                leastWinningCost = leastWinningCosts[timeStep_index]
                # print(pValue)
                # print(leastWinningCost)
                # exit(0)
                bid,state = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"])
                states.append(state)
                if agent.remaining_budget < env.min_remaining_budget:
                    bid = np.zeros(pValue.shape[0])
                    dones[timeStep_index] = 1
                else:
                    # Calculate the bid based on the action
                    if is_Pacing:
                        if actions[int(key[1])][timeStep_index] == 0:
                            bid =  0 * pValue
                        else:
                            bid = (np.mean([action for action in actions[int(key[1])][:] if action != 0])) * pValue
                    else:
                        bid = (np.mean([action for action in actions[int(key[1])][:] if action != 0])) * pValue
                    dones[timeStep_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                            leastWinningCost)

                # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
                while over_cost_ratio > 0:
                    pv_index = np.where(tick_status == 1)[0]
                    dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                        replace=False)
                    bid[dropped_pv_index] = 0
                    tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                                leastWinningCost)
                    over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

                agent.remaining_budget -= np.sum(tick_cost)
                reward_continuous[timeStep_index] = np.sum(tick_value)
                rewards[timeStep_index] = np.sum(tick_conversion)
                temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
                history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
                history["historyBids"].append(bid)
                history["historyLeastWinningCost"].append(leastWinningCost)
                temAuctionResult = np.array(
                    [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
                history["historyAuctionResult"].append(temAuctionResult)
                temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
                history["historyImpressionResult"].append(temImpressionResult)
                # logger.info(f'Timestep Index: {timeStep_index + 1} End')
            all_reward = np.sum(rewards)
            all_reward_con = np.sum(reward_continuous)
            all_cost = agent.budget - agent.remaining_budget
            cpa_real = all_cost / (all_reward + 1e-10)
            cpa_constraint = agent.cpa
            dones[-1] = 1
            score = getScore_neurips(all_reward, cpa_real, cpa_constraint)
            
            for timeStepIndex in range(num_timeStepIndex):
                training_data_rows.append({
                    'deliveryPeriodIndex': key[0],
                    'advertiserNumber': key[1],
                    'advertiserCategoryIndex': key[2],
                    'budget': key[3],
                    'CPAConstraint': cpa_constraint,
                    'realAllCost': all_cost,
                    'realAllConversion': all_reward,
                    'timeStepIndex': timeStepIndex,
                    'state': tuple(states[timeStepIndex]),
                    'action': actions[int(key[1])][timeStepIndex] if actions[int(key[1])][timeStepIndex] != 0 else np.mean([action for action in actions[int(key[1])][:] if action != 0]),
                    'reward': rewards[timeStepIndex],
                    'reward_continuous': reward_continuous[timeStepIndex],
                    'done': dones[timeStepIndex],
                    'next_state': states[timeStepIndex + 1] if dones[timeStepIndex] == 0 else np.nan
                })
            logger.info(f'Total Reward: {all_reward}')
            logger.info(f'Total Reward conti: {all_reward_con}')
            logger.info(f'Total Cost: {all_cost}')
            logger.info(f'CPA-real: {cpa_real}')
            logger.info(f'CPA-constraint: {cpa_constraint}')
            logger.info(f'Score: {score}')

            # Save the results to a CSV file
            with open(f'{agent.name}-evaluation_results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([day, key[1], key[2], key[3], all_reward, all_cost, cpa_real, score])
            average_score += score
        average_score /= len(keys)
        logger.info(f'Average Score for day {day}: {average_score}')
        # training_data = pd.DataFrame(training_data_rows)
        # training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])
        # training_data.to_csv(f'./data/subopt_rl_data/period-{day}-rlData_fill_action.csv', index=False)

    


if __name__ == '__main__':
    run_test()
