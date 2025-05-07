import numpy as np
from vispy import app, scene, visuals

class WeightVisualizer(app.Timer):
    def __init__(self, W1, W2):
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=45, elevation=30)

        self.bar_grid = None
        self.update_weights(W1, W2)

        # Set timer for updates
        super().__init__(interval=0.1, connect=self.on_timer, start=True)

    def update_weights(self, W1, W2):
        all_weights = np.concatenate([W1.flatten(), W2.flatten()])
        N = len(all_weights)
        side = int(np.ceil(np.sqrt(N)))
        padded = np.pad(all_weights, (0, side**2 - N))  # Fill missing with zeros
        self.grid = padded.reshape((side, side))

        # Clear previous bars
        if self.bar_grid:
            self.view.scene.children.clear()

        # Add 3D bars
        self.bar_grid = []
        bar_size = 0.8
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                height = self.grid[i, j]
                bar = scene.visuals.Box(width=bar_size, height=bar_size, depth=abs(height),
                                        color=(0.3, 0.3, 1, 1), edge_color='black')
                bar.transform = scene.transforms.STTransform(translate=(i, j, height / 2))
                self.view.add(bar)
                self.bar_grid.append(bar)

    def on_timer(self, event):
        # Example: simulate live updates by modifying weights randomly
        noise = np.random.randn(*self.grid.shape) * 0.1
        new_grid = self.grid + noise
        self.update_weights_from_array(new_grid)

    def update_weights_from_array(self, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                height = grid[i, j]
                bar = self.bar_grid[i * grid.shape[1] + j]
                bar.transform.translate = (i, j, height / 2)
                bar.set_data(depth=abs(height))

# Example W1 and W2
W1 = np.random.randn(4, 8)
W2 = np.random.randn(8, 2)

# Run visualizer
vis = WeightVisualizer(W1, W2)
app.run()
