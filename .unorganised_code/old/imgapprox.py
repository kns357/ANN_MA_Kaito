import numpy as np
import matplotlib.pyplot as plt
from vispy import app, scene
from vispy.util.filter import gaussian_filter

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

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

#canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
#view = canvas.central_widget.add_view()
#view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=45, elevation=30)

surface_plot = None

def update_weight_plot(W1):
    global surface_plot
    z = W1.T.copy()
    z = gaussian_filter(z, (1, 1))

    z -= z.min()
    if z.max() > 0:
        z /= z.max()
    z *= 0.2  

    if surface_plot is None:
        surface_plot = scene.visuals.SurfacePlot(z=z, color=(0.2, 0.3, 0.9, 1.0), shading=None) #, parent=view.scene)
        surface_plot.transform = scene.transforms.MatrixTransform()
        surface_plot.transform.scale([1 / z.shape[0], 1 / z.shape[1], 1.0])
        surface_plot.transform.translate([-0.5, -0.5, 0])
    else:
        surface_plot.set_data(z=z)

    #canvas.update()

np.random.seed(42)

Ws = []
bs = []

Ws.append(np.random.randn(2, 50) * 0.3)
bs.append(np.zeros((1, 50)))

for _ in range(4):
    Ws.append(np.random.randn(50, 50) * 0.3)
    bs.append(np.zeros((1, 50)))

Ws.append(np.random.randn(50, 1) * 0.3)
bs.append(np.zeros((1, 1)))

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

    #back
    d = bce_derivative(a2, y) * sigmoid_derivative(zs[-1])
    dWs = []
    dbs = []

    for i in reversed(range(len(Ws))):
        dW = activations[i].T @ d
        db = np.sum(d, axis=0, keepdims=True)
        dWs.insert(0, dW)
        dbs.insert(0, db)

        if i != 0:
            d = (d @ Ws[i].T) * relu_derivative(zs[i - 1])

    for i in range(len(Ws)):
        Ws[i] -= lr * dWs[i]
        bs[i] -= lr * dbs[i]

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        #update_weight_plot(Ws[0])


#output = sigmoid(relu(X @ W1 + b1) @ W2 + b2).reshape(10, 10)
a = X
for i in range(5):
    a = relu(a @ Ws[i] + bs[i])
output = sigmoid(a @ Ws[5] + bs[5]).reshape(10, 10)


output_binary = (output > 0.5).astype(float)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img_gt, cmap='gray')
axs[0].set_title("Original Smiley")
axs[0].axis('off')

axs[1].imshow(output_binary, cmap='gray')
axs[1].set_title("NN Prediction")
axs[1].axis('off')

plt.tight_layout()
plt.show()
