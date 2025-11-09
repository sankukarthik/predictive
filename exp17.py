import numpy as np
import matplotlib.pyplot as plt

A = [0,0,1,1,0,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0]
B = [0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0]

def bin_to_rgb(array, shape):
    rgb_array = np.zeros((shape[0], shape[1], 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if array[i*shape[1]+j] == 1:
                rgb_array[i,j] = [1,0,0]
            else:
                rgb_array[i,j] = [0,1,0]
    return rgb_array 

rgb_A = bin_to_rgb(A, (5,6))
rgb_A[0,0] = [0,0,1]
rgb_A[4,5] = [0,0,1]
plt.imshow(rgb_A)
plt.show()