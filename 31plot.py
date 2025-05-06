import matplotlib.pyplot as plt

predict_data = [
    55.225643, 64.02944, 72.63136, 75.075836, 86.054565, 
    124.29917, 112.79528, 103.91117, 107.4092, 102.08875, 
    83.74576, 83.61383, 87.216675, 92.69139, 106.26895, 
    100.83091, 95.022415, 96.86501, 102.75427, 127.5469, 
    153.84514, 168.70865, 173.31244, 177.77373
]

original_data = [
    64, 78, 91, 97, 95, 117, 114, 109, 109, 107, 93, 95, 104, 108, 117,
    114, 102, 111, 118, 147, 163, 183, 198, 204
]

plt.figure(figsize=(12, 6))

plt.plot(predict_data, marker='o', linestyle='-', 
         color='royalblue', linewidth=2, markersize=8, label='Predicted Values')
plt.plot(original_data, marker='s', linestyle='--', 
         color='crimson', linewidth=2, markersize=6, label='Observed Values')


hours = [f"Hour {i+1}" for i in range(len(predict_data))]
plt.xticks(range(len(predict_data)), hours, rotation=45, fontsize=10)


plt.title("Comparison of Predicted vs Observed Values (24 Hours)", 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel("Time (Hourly Intervals)", fontsize=12, labelpad=10)
plt.ylabel("Measurement Value", fontsize=12, labelpad=10)

plt.grid(True, linestyle=':', alpha=0.7) 
plt.legend(fontsize=12, framealpha=1, shadow=True)
plt.tight_layout()
plt.savefig('predicted_vs_observed_one_day.png', dpi=300, bbox_inches='tight')