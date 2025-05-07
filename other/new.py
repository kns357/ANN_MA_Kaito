import sys
import numpy as np

from vispy import app, scene
from vispy.util.filter import gaussian_filter

canvas = scene.SceneCanvas(keys='interactive', bgcolor='w', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60)

# Initialize with zeros; you can call update_surface(z) later with real weights
z = np.zeros((250, 250), dtype=np.float32)
p1 = scene.visuals.SurfacePlot(z=z, color=(0.3, 0.3, 1, 1))
p1.transform = scene.transforms.MatrixTransform()
p1.transform.scale([1/249., 1/249., 0.2])  # Flatten the z-dimension
p1.transform.translate([-0.5, -0.5, 0])
view.add(p1)

# Axes setup
xax = scene.Axis(pos=[[-0.5, -0.5], [0.5, -0.5]], tick_direction=(0, -1),
                 font_size=16, axis_color='k', tick_color='k', text_color='k',
                 parent=view.scene)
xax.transform = scene.STTransform(translate=(0, 0, -0.2))

yax = scene.Axis(pos=[[-0.5, -0.5], [-0.5, 0.5]], tick_direction=(-1, 0),
                 font_size=16, axis_color='k', tick_color='k', text_color='k',
                 parent=view.scene)
yax.transform = scene.STTransform(translate=(0, 0, -0.2))

axis = scene.visuals.XYZAxis(parent=view.scene)

def update_surface(weights: np.ndarray):
    """Update the surface plot with new weights (2D numpy array)."""
    if weights.shape != z.shape:
        raise ValueError(f"Expected shape {z.shape}, got {weights.shape}")

    # Normalize to [-1, 1] and smooth
    norm_z = weights - np.mean(weights)
    norm_z /= (np.std(norm_z) + 1e-8)
    norm_z = gaussian_filter(norm_z, (3, 3))  # optional smoothing

    p1.set_data(z=norm_z.astype(np.float32))  # update surface

# Example: live update every second with random weights
def update(event):
    new_weights = np.random.normal(size=(250, 250))
    update_surface(new_weights)

timer = app.Timer(interval=0.1, connect=update, start=True)

if __name__ == '__main__':
    if sys.flags.interactive == 0:
        app.run()
