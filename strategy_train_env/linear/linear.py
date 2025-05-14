import numpy as np
import pandas as pd
import pulp
from multiprocessing import Pool


# 读取数据
def process_date(date):
    datafile = f"./data/splited_data/period-{date}_0.csv"
    print(datafile)
    df = pd.read_csv(datafile)
    df['x'] = np.nan
    df['alpha'] = np.nan
    resultset = {}
    for advertiserNumber in range(0,48):
        print(datafile,"   ",advertiserNumber)
        temp_df = df[df['advertiserNumber'] == advertiserNumber]
        budget = temp_df['budget'].iloc[0] / 10
        CPAConstraint = temp_df['CPAConstraint'].iloc[0]

        # 按时间步分组
        groups = temp_df.groupby('timeStepIndex')
        V = []
        C = []
        for name, group in groups:
            V.append(group['pValue'].values)
            C.append(group['leastWinningCost'].values)

        # 创建问题实例（最大化）
        prob = pulp.LpProblem("Maximize Value", pulp.LpMaximize)

        # 决策变量：每个 x[i][j] 是 0 或 1，表示时间步长 i 下的第 j 个广告的投标选择
        x = [[pulp.LpVariable(f"x_{i}_{j}", cat='Binary') for j in range(len(V[i]))] for i in range(len(V))]

        # 创建 z 变量
        z = pulp.LpVariable("z", lowBound=0)  # z 变量，用于 CPA 约束

        # α 的二维数组
        alpha = [pulp.LpVariable(f"alpha_{i}", lowBound=0) for i in range(48)]

        # 定义大 M 常数
        M = 1000  # 确保 M 足够大

        # 目标函数：最大化 ∑ v_i * x_i
        prob += pulp.lpSum([V[i][j] * x[i][j] for i in range(len(V)) for j in range(len(V[i]))]), "Objective"

        # 约束 1: ∑ c_i * x_i ≤ B
        prob += pulp.lpSum([C[i][j] * x[i][j]  for i in range(len(C)) for j in range(len(C[i]))]) <= budget, "Budget Constraint"

        # 约束 2: ∑ C_i * x_i ≤ CPAConstraint * z
        prob += pulp.lpSum([C[i][j] * x[i][j]  for i in range(len(x)) for j in range(len(x[i]))]) <= CPAConstraint * z, "CPA Constraint"

        # 约束 3: z = ∑ V_i * x_i
        prob += z == pulp.lpSum([V[i][j] * x[i][j]  for i in range(len(V)) for j in range(len(V[i]))]), "Z Constraint"

        # 线性化的约束
        for i in range(len(V)):
            for j in range(len(V[i])):
                prob += alpha[i] * V[i][j] - C[i][j] >= -M * (1 - x[i][j]), f"Constraint_ge_{i}_{j}"
                prob += alpha[i] * V[i][j] - C[i][j] <= M * x[i][j], f"Constraint_le_{i}_{j}"

        # 求解问题
        prob.solve()

        # 输出结果
        print("最优解 x:")
        sumsum = 0
        result = []
        for i in range(len(x)):
            for j in range(len(x[i])):
                sumsum += C[i][j] * x[i][j].varValue
                result.append(x[i][j].varValue)

        # 使用 .loc 逐步填充 'alpha' 列
        for i in range(48):
            df.loc[(df['timeStepIndex'] == i) & (df['advertiserNumber'] == advertiserNumber) , 'alpha'] = alpha[i].varValue
        df.loc[df['advertiserNumber'] == advertiserNumber, 'x'] = result
        del temp_df
        print("总成本和:", sumsum)
        print("最大化的目标值:", pulp.value(prob.objective))
        print("CPA:", sumsum / (pulp.value(prob.objective) + 1e-10))
        resultset[advertiserNumber] = [sumsum,pulp.value(prob.objective),sumsum / (pulp.value(prob.objective) + 1e-10)]
    
        with open(f"./data/pulp/period-{date}.txt",'w') as f:
            f.write(str(resultset))
    df.to_csv(f"./data/pulp/period-{date}.csv", index=False)
    del df
    
    
def main():
    dates = range(7, 14)  # 处理一周的数据
    with Pool(7) as p:  # 创建一个包含7个进程的池
        p.map(process_date, dates)
        
if __name__ == "__main__":
    main()