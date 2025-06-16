import numpy as np
import pulp
import pandas as pd


def get_best_alpha(budget,cumulative_cost,beta=2):
    # print(budget)
    
    selected_index = np.searchsorted(cumulative_cost, budget, side="left")

    selected_cost_total = cumulative_cost[selected_index] if selected_index < len(cumulative_cost) else cumulative_cost[-1]
    selected_orders_total = cumulative_orders[selected_index] if selected_index < len(cumulative_orders) else cumulative_orders[-1]
    # if selected_cost_total / selected_orders_total < cpa:
    if selected_index >= len(sorted_indices):
        return cpa
    return leastWinningCosts[sorted_indices[selected_index]] / pValues[sorted_indices[selected_index]] , selected_orders_total


data_all = pd.read_csv("./data/traffic/period-7.csv")
result = []
for advertiserNumber in range(48):
    data = data_all[data_all["advertiserNumber"]==advertiserNumber]
    pValues = data["pValue"].values
    leastWinningCosts = data["leastWinningCost"].values
    real_cpa = leastWinningCosts / np.where(pValues == 0, 1e-10, pValues)
    sorted_indices = np.argsort(real_cpa)
    cumulative_cost = np.cumsum(leastWinningCosts[sorted_indices])
    cumulative_orders = np.cumsum(pValues[sorted_indices])
    cum_cost = np.max(cumulative_cost)
    cum_cost = int(cum_cost)
    for budget in range(0,cum_cost,10):
        cpa = data["CPAConstraint"].values[0]
        best_alpha,orders = get_best_alpha(budget, cumulative_cost)
        result.append([advertiserNumber,budget,best_alpha,orders])
        print(f"Budget: {budget}, Optimal Alpha: {best_alpha}") 
result = pd.DataFrame(result, columns=["advertiserNumber","budget", "alpha", "orders"])
result.to_csv("./linear_solution/alpha_cpa_from_bottom.csv", index=False)

    # real_cost = 0
    # real_orders = 0
    # max_score = 0
    # best_index = -1
    # max_orders = 0
    # for index in sorted_indices:
    #     real_cost += leastWinningCosts[index]
    #     real_orders += pValues[index]
    #     if real_cost > budget:
    #         break
    #     now_cpa = real_cost / (real_orders if real_orders > 0 else 1e-10) 
    #     score = real_orders * (min(1,cpa/ now_cpa)**beta)
    #     if score > max_score:
    #         max_score = score
    #         best_index = index
    #         max_orders = real_orders
    # return real_cpa[best_index] if best_index != -1 else cpa

# data = pd.read_csv("./data/splited_data/period-7_0.csv")
# data = data[data["advertiserNumber"]==25]
# p_values = data["pValue"].values
# costs = data["leastWinningCost"].values

# cpa = data["CPAConstraint"].values[0]
# budget = data["budget"].values[0]/20.0
# # best_alpha = get_best_alpha(cpa, budget, p_values, costs)
# # print(f"Optimal Alpha: {best_alpha}")
# import matplotlib.pyplot as plt

# betas = np.linspace(0.1, 10, 50)
# alphas = []
# max_scores = []
# max_orders = []

# for beta in betas:
#     best_alpha, max_score, max_order= get_best_alpha(cpa, budget, p_values, costs, beta=beta)
#     alphas.append(best_alpha)
#     max_scores.append(max_score)
#     max_orders.append(max_order)

# plt.figure(figsize=(12, 8))

# # Plot Best Alpha
# plt.subplot(3, 1, 1)
# plt.plot(betas, alphas, marker='o', label="Best Alpha")
# plt.title("Relationship between Beta and Best Alpha/Max Score")
# plt.xlabel("Beta")
# plt.ylabel("Best Alpha")
# plt.grid()
# plt.legend()

# # Plot Max Score
# plt.subplot(3, 1, 2)
# plt.plot(betas, max_scores, marker='o', color='orange', label="Max Score")
# plt.xlabel("Beta")
# plt.ylabel("Max Score")
# plt.grid()
# plt.legend()


# # Plot Max Orders
# plt.subplot(3, 1, 3)
# # plt.figure(figsize=(12, 8))
# plt.plot(betas, max_orders, marker='o', color='green', label="Max Orders")
# plt.xlabel("Beta")
# plt.ylabel("Max Orders")
# plt.grid()
# plt.legend()

# plt.tight_layout()
# plt.show()



