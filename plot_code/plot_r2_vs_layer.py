import numpy as np
import matplotlib.pyplot as plt

r2 = [0.061, 0.103, 0.149, 0.148, 0.161, 0.180, 0.177, 0.155]
# r2 = [ 0.161, 0.180, 0.177, 0.155]
layer = ['relu1', 'relu2', 'relu3', 'avgpool', 'layer1', 'layer2', 'layer3', 'layer4']
# Plot the data
plt.rcParams.update({'font.size': 18})
plt.plot(layer, r2)
plt.xticks(rotation=45)
# plt.xlabel('CLIP ResNet50 Layer')
plt.ylabel('$R^2$')
plt.title('$R^2$ vs CLIP ResNet50 Layer')
plt.grid(True, linestyle='--', alpha=0.5)
plt.scatter(layer, r2, zorder=5)
# Find the index of the highest R2 value
max_idx = np.argmax(r2)
# Plot a star marker at the highest point
plt.scatter([layer[max_idx]], [r2[max_idx]], marker='*', s=200, zorder=10, label='Max $R^2$')
plt.legend()



  # Set bigger font for all plot elements


plt.savefig('/home/junruz/prf_model/plot_save_code/r2_vs_layer_v1_5layers.pdf', bbox_inches='tight', dpi=300)
