import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(num_points, noise_factor=0.5):
    """Generates intertwined spiral data with noise."""
    n = np.sqrt(np.random.rand(num_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(num_points,1) * noise_factor
    d1y = np.sin(n)*n + np.random.rand(num_points,1) * noise_factor
    data = np.vstack((np.hstack((d1x, d1y, np.zeros(num_points,1))),
                      np.hstack((-d1x, -d1y, np.ones(num_points,1)))))
    np.random.shuffle(data)
    return data

num_spiral_points = 1000 #increase the number of spiral points
num_distractor_points = 2000 # increase the number of distractor points
noise_level = 5 # increase noise level

spiral_data = generate_spiral_data(num_spiral_points, noise_level)
distractor_data = np.random.rand(num_distractor_points, 2) * 20 - 10 #random points
distractor_labels = np.random.randint(0, 2, num_distractor_points).reshape(num_distractor_points,1)
distractor_data = np.hstack((distractor_data, distractor_labels))

all_data = np.vstack((spiral_data, distractor_data))
np.random.shuffle(all_data)

# Separate features and labels
features = all_data[:, :2]
labels = all_data[:, 2]

# Plot the data
plt.scatter(features[labels == 0, 0], features[labels == 0, 1], label='Class 0', s=10)
plt.scatter(features[labels == 1, 0], features[labels == 1, 1], label='Class 1', s=10)
plt.legend()
plt.title('Intertwined Spirals with Distractors')
plt.show()

# Now, you would train your 5-layer, 50-neuron neural network on 'features' and 'labels'