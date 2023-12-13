import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d 
import pdb

cubic_size = 1
voxel_resolution = 10

voxel_carving = o3d.geometry.VoxelGrid.create_dense(
    width=cubic_size,
    height=cubic_size,
    depth=cubic_size,
    voxel_size=cubic_size / voxel_resolution,
    origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
    color=[1.0, 0.7, 0.0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))  # Random point cloud data

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame() 
vis = o3d.visualization.Visualizer() 
vis.create_window(visible=True) 
vis.add_geometry(mesh) 
vis.poll_events() 
vis.update_renderer() 

voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)

depth = vis.capture_depth_float_buffer(True) 
color = vis.capture_screen_float_buffer(True) 
vis.destroy_window() 
color = np.asarray(color) 
depth = np.asarray(depth) 
plt.imshow(color) 
plt.show() 
plt.imshow(depth) 
plt.show()  

pdb.set_trace()