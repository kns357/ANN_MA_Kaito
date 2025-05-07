import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

size = 64
img_gt = np.zeros((size, size), dtype=np.float32)
for y in range(size):
    for x in range(size):
        dx, dy = x - size / 2, y - size / 2
        dist = np.sqrt(dx**2 + dy**2)
        img_gt[y, x] = np.exp(-dist**2 / (2 * (size / 4)**2))

X_np = np.array([[x / (size - 1), y / (size - 1)] for y in range(size) for x in range(size)], dtype=np.float32)
y_np = img_gt.flatten().reshape(-1, 1).astype(np.float32)

X = torch.tensor(X_np, device=device)
y = torch.tensor(y_np, device=device)

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

model = TinyNet().to(device)

weight_history = []
log_interval = 1000


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 50000
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    if epoch % log_interval == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        snapshot = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy()
                snapshot.append(np.mean(W, axis=0))
        weight_history.append(snapshot)


with torch.no_grad():
    pred = model(X).reshape(size, size).cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_gt, cmap='gray')
axs[0].set_title("Ground Truth")
axs[0].axis('off')

axs[1].imshow(pred, cmap='gray')
axs[1].set_title("Prediction")
axs[1].axis('off')

axs[2].set_title("Final Layer Weights")
for i, layer in enumerate(model.children()):
    if isinstance(layer, nn.Linear):
        W = layer.weight.detach().cpu().numpy()
        if W.shape[1] == 1:
            axs[2].plot(W.flatten(), label=f'Layer {i}')
        else:
            axs[2].plot(np.mean(W, axis=0), label=f'Layer {i}')
axs[2].legend()
axs[2].set_xlabel("Neuron Index")
axs[2].set_ylabel("Avg Weight Value")
axs[2].grid(True)

plt.tight_layout()
plt.show()

import matplotlib.animation as animation

weight_array = np.array(weight_history)

fig, ax = plt.subplots(figsize=(8, 5))
lines = []
for i in range(weight_array.shape[1]):
    (line,) = ax.plot([], [], label=f'Layer {i}')
    lines.append(line)

ax.set_xlim(0, weight_array.shape[2] - 1)
ax.set_ylim(np.min(weight_array), np.max(weight_array))
ax.set_xlabel("Neuron Index")
ax.set_ylabel("Average Weight")
ax.set_title("Weight Evolution")
ax.legend()
ax.grid(True)

def animate(frame):
    for i, line in enumerate(lines):
        line.set_data(np.arange(weight_array.shape[2]), weight_array[frame, i])
    ax.set_title(f"Weight Evolution - Step {frame * log_interval}")
    return lines

ani = animation.FuncAnimation(fig, animate, frames=len(weight_array), interval=100, blit=True)
plt.show()
