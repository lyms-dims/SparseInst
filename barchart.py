import matplotlib.pyplot as plt
import numpy as np

# Data from your log files
metrics = ['AP (Overall)', 'AP50 (Accuracy)', 'APs (Small)', 'APl (Large)']
base_model_scores = [6.054, 11.886, 0.033, 50.739]   # From base.txt
gabor_model_scores = [6.864, 13.540, 0.151, 53.303]  # From gabor.txt

# Settings for the chart
x = np.arange(len(metrics))  # label locations
width = 0.35  # width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, base_model_scores, width, label='SparseInst (Base)', color='#d95f02')
rects2 = ax.bar(x + width/2, gabor_model_scores, width, label='Gabor-Enhanced', color='#1b9e77')

# Add styling labels, title, and custom x-axis tick labels
ax.set_ylabel('Average Precision (AP)')
ax.set_title('Comparison of Detection Metrics: Base vs. Gabor-Enhanced Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Function to add value labels on top of bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

# Apply the labels
autolabel(rects1)
autolabel(rects2)

# Final layout adjustments
plt.ylim(0, 60) # Set y-limit to accommodate the large object scores
plt.grid(axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()

# Show the plot
plt.show()

# To save the figure for your thesis, uncomment the line below:
# plt.savefig('Figure_4_1_Quantitative_Results.png', dpi=300)