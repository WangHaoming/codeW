import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), _ = load_mnist(normalize=False, flatten=False)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i in range(10):
    axes[i].imshow(x_train[i][0], cmap='gray')
    axes[i].set_title(str(t_train[i]))
    axes[i].axis('off')
plt.tight_layout()
plt.show()
