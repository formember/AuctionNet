import matplotlib.pyplot as plt

# Data
strategies = ["GAVE",'Iql', 'Dt', 'Bc', 'Dt_pro', 'Bc_pro', 'Dt_pro_max', 'Bc_pro_max']
values = [14.49,29.51, 28.98, 22.55, 31.33, 30.90, 34.04, 33.56]

last_strategy = ['Iql', 'Dt', 'Bc']
values_last = [29.21,27.27,30.34]

fig, ax = plt.subplots()

# Bar width
bar_width = 0.35

# Positions for the bars
x = range(len(strategies))
x_last = [strategies.index(s) for s in last_strategy]

# Plot all strategies
ax.bar(x, values, bar_width, label='flow-shift')

# Plot last strategies
ax.bar([i + bar_width for i in x_last], values_last, bar_width, label='original')
# Add text above bars
for i, v in enumerate(values):
    ax.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=8)

for i, v in zip([i + bar_width for i in x_last], values_last):
    ax.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=8)
# Add labels and title
ax.set_xlabel('Strategies')
ax.set_ylabel('Score')
ax.set_title('Comparison of Strategies')
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(strategies)
ax.legend()
# Adjust figure size to prevent overlap
fig.set_size_inches(12, 6)
# Show plot
plt.gcf().set_dpi(300)
plt.tight_layout()
plt.savefig('strategy_comparison.png')

