import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('performance_data.csv')

# Create figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot: comparison of execution times
functions = data['Function']
regular_time = data['Regular Time (ms)']
dual_time = data['Dual Number Time (ms)']

x = np.arange(len(functions))
width = 0.35

rects1 = ax1.bar(x - width/2, regular_time, width, label='Regular Calculation')
rects2 = ax1.bar(x + width/2, dual_time, width, label='Dual Number Calculation')

ax1.set_xlabel('Function')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(functions, rotation=45, ha='right')
ax1.legend()

# Second plot: overhead factors
overhead = data['Overhead Factor']

ax2.bar(functions, overhead, color='green')
ax2.axhline(y=1.0, color='r', linestyle='--', label='Equal Performance')
ax2.set_xlabel('Function')
ax2.set_ylabel('Overhead Factor (Dual/Regular)')
ax2.set_title('Overhead of Dual Number Calculations')
ax2.set_xticklabels(functions, rotation=45, ha='right')
ax2.legend()

# Add values on top of bars
for i, v in enumerate(overhead):
    ax2.text(i, v + 0.1, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
plt.close()

print("Plot saved as 'performance_comparison.png'") 