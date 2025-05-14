import pandas
import matplotlib.pyplot as plt
import numpy as np

def plot_budget_consume_chart(data):
    day0data = data[data['Day'] == 21]
    adgroups = day0data.groupby('Id')
    budgets = []
    total_costs = []
    for name, group in adgroups:
        budget = group['Budget'].values[0]
        total_cost = group['Total-Cost'].values[0]
        budgets.append(budget)
        total_costs.append(total_cost)
    # Calculate the proportion of total costs relative to budgets
    proportions = [cost / budget if budget > 0 else 0 for cost, budget in zip(total_costs, budgets)]
    # Combine all proportions into a single pie chart
    plt.figure(figsize=(12, 16))  # Increase the figure size for larger subplots
    num_proportions = len(proportions)
    cols = 6  # Number of columns in the grid
    rows = (num_proportions + cols - 1) // cols  # Calculate the number of rows needed

    for i, proportion in enumerate(proportions):
        plt.subplot(rows, cols, i + 1)
        plt.pie([proportion, 1 - proportion], autopct='%1.1f%%', startangle=250)
        plt.title(f'Ad {i + 1}', fontsize=14)  # Increase title font size

    plt.tight_layout()
    plt.savefig('individual_proportion_pie_charts.png')


def cpa_exceedance(data):
    # Filter the data for Day 21
    day0data = data[data['Day'] == 21]
    adgroups = day0data.groupby('Id')
    cpa_exceedance = []
    cpa_constraints = []
    for name, group in adgroups:
        CPAconstraint = group['CPAconstraint'].values[0]
        CPA_Real = group['CPA-Real'].values[0]
        cpa_exceedance.append((CPA_Real - CPAconstraint) / CPAconstraint)
        cpa_constraints.append(CPAconstraint)
    # Plot a bar chart to visualize CPA exceedance
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for CPA exceedance
    ax1.bar(range(len(cpa_exceedance)), cpa_exceedance, color='skyblue', label='CPA Exceedance')
    ax1.set_xlabel('Advertizer')
    ax1.set_ylabel('CPA Exceedance', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(range(len(cpa_exceedance)))
    ax1.set_xticklabels([f'{i + 1}' for i in range(len(cpa_exceedance))])

    # Second y-axis for CPA constraints
    ax2 = ax1.twinx()
    ax2.plot(range(len(cpa_constraints)), cpa_constraints, color='orange', marker='o', label='CPA Constraint')
    ax2.set_ylabel('CPA Constraint', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add CPA_Real to the chart
    CPA_Real_values = [group['CPA-Real'].values[0] for name, group in adgroups]
    ax2.plot(range(len(CPA_Real_values)), CPA_Real_values, color='green', marker='x', label='CPA Real')

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Title and layout
    plt.title('CPA Exceedance, CPA Constraints, and CPA Real for Each Advertizer')
    fig.tight_layout()
    plt.savefig('cpa_exceedance_constraints_and_real_chart.png')


def get_score(data):
    print(np.mean(data['Score'].values))

data = pandas.read_csv('./Decision-Transformer-PlayerStrategy-evaluation_results.csv')
get_score(data)



