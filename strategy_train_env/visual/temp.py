import pandas as pd 
import numpy as np
data = pd.read_csv("./data/pulp_v3_withoutcpa/period-7.csv")
data = data[(data["advertiserNumber"] == 1)]
import matplotlib.pyplot as plt
# data_0 = data[data["timeStepIndex"] == 0.0]
# data_10 = data[data["timeStepIndex"] == 10.0]

alpha = 0
for t in range(0,5):
    data_t = data[data["timeStepIndex"] == t]
    plt.scatter(data_t["pValue"], data_t["leastWinningCost"], label=f"timestep {t}",s=10)
    if t == 0 :
        alpha = data_t["alpha"].values[0]
    x = np.arange(float(data["pValue"].min()), float(data["pValue"].max()), 0.001)
    y = [alpha * xi for xi in x]
    # plt.plot(x, y, label=f"timestep {t}")
    below_line = data_t[data_t["leastWinningCost"] < alpha * 1.5 * data_t["pValue"]]
    plt.scatter(below_line["pValue"], below_line["leastWinningCost"], facecolors='none', edgecolors='red', s=30)
x = np.arange(float(data["pValue"].min()), float(data["pValue"].max()), 0.001)
y = [alpha *1.5 * xi for xi in x]
plt.plot(x, y) 
# plt.scatter(data_0["pValue"], data_0["leastWinningCost"], label="timestep 0", color="blue")
# plt.scatter(data_10["pValue"], data_10["leastWinningCost", label="timestep 10", color="orange")



plt.legend()

# plt.scatter(data["pValue"], data["leastWinningCost"])
plt.xlabel("pvalue")
plt.ylabel("leastwinningbid")
plt.title("Relationship between pvalue and leastwinningbid")
plt.gcf().set_dpi(300)
plt.savefig("pvalue_leastwinningbid.png")
