import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.geometry import create_cube

# Create a canvas and add a simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Generate grid coordinates (100³ grid) with more space along the x-axis
grid_size = 10  # size of the grid per axis
spacing = 3  # space between cubes along the x-axis
x, y, z = np.meshgrid(np.arange(grid_size) * spacing, np.arange(grid_size), np.arange(grid_size))
pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

# Generate random color values between 0 and 1
color_values = np.random.rand(len(pos))

# Map the color values to a gradient from white to pink
colors = np.array([[1, color, color, 1] for color in color_values], dtype=np.float32)

# Create cube mesh data
cube = create_cube()  # create_cube() returns a single mesh object
cube_vertices = cube.vertices  # Get the vertices
cube_faces = cube.faces      # Get the faces

# Create a visual to render the cubes
cubes = []
for p, color in zip(pos, colors):
    # Translate each cube to its corresponding position
    cube_mesh = visuals.Mesh(vertices=cube_vertices + p, faces=cube_faces, color=color)
    cubes.append(cube_mesh)
    cube_mesh.parent = view.scene  # Add cube to the scene

# Set the camera to a turntable view for easy rotation
view.camera = 'turntable'

# Add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

# Run the application
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
