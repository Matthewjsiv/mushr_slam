import open3d as o3d
import numpy as np
import pdb

# # Create a point cloud or load your data
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))  # Random point cloud data

def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model

armadillo = o3d.data.ArmadilloMesh()
mesh = preprocess(o3d.io.read_triangle_mesh(armadillo.path))

# Create a Visualizer object

mesh.translate([10,10,10])

vis = o3d.visualization.Visualizer()
vis.create_window(width=300, height=300, visible=True)
vis.add_geometry(mesh)
ctr = vis.get_view_control()
# param = ctr.convert_to_pinhole_camera_parameters()
# trans = np.eye(4)
# trans[:-1,-1] = [10,20,30]
# param.extrinsic = trans
# ctr.convert_from_pinhole_camera_parameters(param)
ctr.set_eye([10,10,10])

vis.run()
# vis.destroy_window()

pdb.set_trace()

# Get the view control of the visualizer
view_control = vis.get_view_control()

camera_position = [5, 5, 5]  # Set a different position for the camera
view_control.set_lookat([0, 0, 0])  # Look at the origin
view_control.set_front([1, 1, 1])  # Set the camera's front direction
view_control.set_up([0, 1, 0])  # Set the camera's up direction
view_control.set_constant_z_far(1000)  # Optionally adjust the far clipping plane

# Set the camera to view the origin (0, 0, 0)
view_control.set_lookat([0, 0, 0])  # Look at the origin

# Optionally set the camera's field of view or other parameters
# view_control.set_field_of_view(60.0)  # Adjust field of view if needed

# Let the visualizer render the scene
vis.poll_events()
vis.update_renderer()

# Capture the depth information by rendering the scene
depth = vis.capture_depth_float_buffer(False)  # Capture depth without cleaning the buffer

# Extract the depth map
depth_data = np.array(depth)

# Close the visualizer
vis.destroy_window()

# Process or display the depth map (depth_data)
# For instance, you can visualize it using matplotlib
import matplotlib.pyplot as plt
plt.imshow(depth_data)
plt.colorbar()
plt.show()