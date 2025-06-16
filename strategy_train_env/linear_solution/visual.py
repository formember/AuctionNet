import pandas as pd
import numpy as np
data = pd.read_csv("./linear_solution/alpha_cpa_from_bottom.csv")
data = data[data["alpha"] <= 1000]
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(x, a, b):
    return a / (b-x)
data_all = pd.read_csv("./data/traffic/period-7.csv")

fig, axes = plt.subplots(8, 6, figsize=(20, 15))
axes = axes.flatten()
result = []
for advertiserNumber in range(48):
    data_ad = data[data["advertiserNumber"] == advertiserNumber]
    if data_ad.empty:
        continue
    x = data_ad['budget'].values
    y = data_ad['alpha'].values
    z = data_ad['orders'].values
    data_all_ad= data_all[data_all["advertiserNumber"]==advertiserNumber]
    pValues = data_all_ad["pValue"].values
    leastWinningCosts = data_all_ad["leastWinningCost"].values
    pValue_max = np.max(pValues)
    cost_avg = np.mean(leastWinningCosts)
    N = len(pValues)
    A = cost_avg ** 2 * (N+1) / pValue_max
    B = N * cost_avg
    y_fit = A / (B - x) 
    # Fit a linear function
    # coefficients = np.polyfit(x, y, 2)
    # linear_fit = np.poly1d(coefficients)
    # y_fit = linear_fit(x)

    # popt,pcov = curve_fit(func, x, y)
    # y_fit = func(x, *popt)
    # print(linear_fit)
    # # result.append([advertiserNumber, coefficients[0], coefficients[1]])
    

    coefficients_orders = np.polyfit(x, z, 1)
    linear_fit_orders = np.poly1d(coefficients_orders)
    y_fit_orders = linear_fit_orders(x)
    # Plot the data
    ax = axes[advertiserNumber]
    ax.plot(np.array(x), np.array(y_fit), color='red', linewidth=2)
    ax.plot(np.array(x), np.array(y_fit_orders), color='blue', linewidth=2)
    ax.scatter(data_ad['budget'], data_ad['alpha'])
    ax.scatter(data_ad['budget'], data_ad['orders']) 
    ax.set_xlabel('Budget')
    ax.set_ylabel('Coef / Orders')
    ax.set_title(f'Advertiser {advertiserNumber}') 
    ax.legend()
# print(result)
# plt.show()
plt.tight_layout()

plt.savefig("./linear_solution/alpha_cpa_from_bottom.png", dpi=400)