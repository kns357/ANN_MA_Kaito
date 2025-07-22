import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

#bce
def bce(y_pred, y_true, eps=1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_pred, y_true, eps=1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


smiley_coords = set([
    (2, 3), (6, 3),  #Eyes
    (2, 7), (3, 8), (4, 8), (5, 8), (6, 8), (7, 7),  #Mouth
    (1, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 1),
    (8, 2), (9, 4), (9, 5), (9, 6), (8, 8),
    (7, 9), (6, 9), (5, 9), (4, 9), (3, 9),
    (2, 8), (1, 7), (0, 6), (0, 5), (0, 4), (1, 2)
])

coords = []
pixels = []
img_gt = np.zeros((10, 10))

for y in range(10):
    for x in range(10):
        coords.append([x / 9, y / 9])
        if (x, y) in smiley_coords:
            pixels.append([1.0])
            img_gt[y, x] = 1.0
        else:
            pixels.append([0.0])

X = np.array(coords)
y = np.array(pixels)

plt.ion()
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Layer Weights During Training', fontsize=16)
lines = []

for i in range(6):
    ax = axs.flatten()[i]
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Weight Value')
    ax.set_title(f'Layer {i}')
    line, = ax.plot([], [], 'o-', markersize=2, alpha=0.5)
    lines.append(line)
    ax.grid(True)
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show(block=False)

np.random.seed(42)
Ws = [
    np.random.randn(2, 50) * 0.3,       
    *[np.random.randn(50, 50) * 0.3 for _ in range(4)],  
    np.random.randn(50, 1) * 0.3        
]
bs = [np.zeros((1, 50)) for _ in range(5)] + [np.zeros((1, 1))]

epochs = 200000
lr = 0.05

for epoch in range(epochs):
    #Forward
    a = X
    zs = []
    activations = [X]
    for i in range(len(Ws) - 1): 
        z = a @ Ws[i] + bs[i]
        zs.append(z)
        a = relu(z)
        activations.append(a)

    z = a @ Ws[-1] + bs[-1]
    zs.append(z)
    a2 = sigmoid(z)
    activations.append(a2)

    loss = bce(a2, y)
    

    d = bce_derivative(a2, y) * sigmoid_derivative(zs[-1])
    dWs = [None] * len(Ws)
    dbs = [None] * len(Ws)
    
    dWs[-1] = activations[-2].T @ d   
    dbs[-1] = np.sum(d, axis=0, keepdims=True)
    
    for i in range(len(Ws) - 2, -1, -1):
        d = (d @ Ws[i+1].T) * relu_derivative(zs[i])
        dWs[i] = activations[i].T @ d
        dbs[i] = np.sum(d, axis=0, keepdims=True)

    for i in range(len(Ws)):
        Ws[i] -= lr * dWs[i]
        bs[i] -= lr * dbs[i]

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        for i, (W, line) in enumerate(zip(Ws, lines)):
            if W.shape[1] == 1: 
                x_vals = np.arange(W.shape[0])
                y_vals = W.flatten()
            else:
                x_vals = np.arange(W.shape[1])
                y_vals = np.mean(W, axis=0)  
            line.set_data(x_vals, y_vals)
            axs.flatten()[i].set_xlim(x_vals.min() - 1, x_vals.max() + 1)
            axs.flatten()[i].set_ylim(y_vals.min() - 0.1, y_vals.max() + 0.1)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

plt.ioff()
a = X
for i in range(5):
    a = relu(a @ Ws[i] + bs[i])
output = sigmoid(a @ Ws[5] + bs[5]).reshape(10, 10)
output_binary = (output > 0.5).astype(float)

fig_final, axs_final = plt.subplots(1, 2, figsize=(8, 4))
axs_final[0].imshow(img_gt, cmap='gray')
axs_final[0].set_title("Original Smiley")
axs_final[0].axis('off')

axs_final[1].imshow(output_binary, cmap='gray')
axs_final[1].set_title("NN Prediction")
axs_final[1].axis('off')

plt.tight_layout()
plt.show()