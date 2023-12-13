import open3d as o3d
import numpy as np

# Define parameters for the dense block (initial voxel grid)
voxel_size = 0.2  # Voxel size
dense_block_size = 2  # Define dimensions of the dense block

# setup dense voxel grid
dense_block = o3d.geometry.VoxelGrid.create_dense(
    width=dense_block_size,
    height=dense_block_size,
    depth=dense_block_size,
    voxel_size=dense_block_size / voxel_size,
    origin=[-dense_block_size / 2.0, -dense_block_size / 2.0, -dense_block_size / 2.0],
    color=[1.0, 0.7, 0.0])

# Create or load a shape (e.g., a cube) to carve out from the dense block
cube = o3d.geometry.TriangleMesh.create_box(width=20, height=20, depth=20)

# Transform the shape to a desired location and orientation within the dense block
transformation = np.identity(4)  # Define a transformation matrix for positioning the shape
transformation[:3, 3] = [15, 15, 15]  # Set the translation
cube.transform(transformation)

# Carve out the shape from the dense block
dense_block_carved = dense_block.crop(cube)

# Visualize the dense block before and after carving
o3d.visualization.draw_geometries([dense_block], window_name="Before Carving")
o3d.visualization.draw_geometries([dense_block_carved], window_name="After Carving")
