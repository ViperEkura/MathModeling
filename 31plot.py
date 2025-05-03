import matplotlib.pyplot as plt

predict_data = [
    60.98784, 67.91877, 80.27748, 80.24268, 96.51105, 144.09717, 125.40897,
    116.9431, 121.726265, 112.36243, 87.46846, 86.08644, 93.230675, 100.815384,
    109.77013, 102.53694, 97.38198, 98.21708, 98.59893, 124.77503, 145.60016,
    159.04156, 172.00089, 172.42754
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