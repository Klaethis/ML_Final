from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt
import numpy as np

x, y = loadlocal_mnist(images_path="./Training_Data/train-images.idx3-ubyte", labels_path="./Training_data/train-labels.idx1-ubyte")

temp = np.reshape(x[0], (28,28))

image = temp.squeeze()
plt.imshow(image)
plt.show()