import numpy as np
import open3d as o3d
import pdb

pcd_aerial = o3d.io.read_point_cloud('../pointclouds/point_cloud.pts')
# o3d.visualization.draw_geometries([pcd_aerial])

point_cloud_aerial = np.array(pcd_aerial.points)
point_cloud_aerial[:,2] -= np.min(point_cloud_aerial[:,2])

# Set floor height
lift = 0.75
point_cloud_aerial[point_cloud_aerial[:,2]<lift,2] = lift

# Crop out kitchen (Aerial map)
x_min = -6.3
x_max = -.25
y_min = 7
y_max = 30
pts_to_keep_inds = (point_cloud_aerial[:,0] > x_min) & (point_cloud_aerial[:,0] < x_max) & (point_cloud_aerial[:,1] > y_min) & (point_cloud_aerial[:,1] < y_max)
point_cloud_aerial = point_cloud_aerial[pts_to_keep_inds]
pcd_aerial.points = o3d.utility.Vector3dVector(point_cloud_aerial)
pcd_aerial.colors = o3d.utility.Vector3dVector(np.asarray(pcd_aerial.colors)[pts_to_keep_inds])
# pcd_aerial = pcd_aerial.voxel_down_sample(.05)
pcd_aerial.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.07, max_nn=30))
# Convert to voxel grid
voxel_grid_aerial = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_aerial, voxel_size=0.05)
# o3d.visualization.draw_geometries([voxel_grid_aerial])





pcd_ground = o3d.io.read_point_cloud('../pointclouds/ground_point_cloud_3.pts')
# o3d.visualization.draw_geometries([pcd_ground])

point_cloud_ground = np.array(pcd_ground.points)
point_cloud_ground[:,2] -= np.min(point_cloud_ground[:,2])

# Set floor height for ground map
lift = 0.15
point_cloud_ground[point_cloud_ground[:,2]<lift,2] = lift

# Crop ground map
x_min = -4
x_max = 2.3
y_min = -10
y_max = 10
pts_to_keep_inds = (point_cloud_ground[:,0] > x_min) & (point_cloud_ground[:,0] < x_max) & (point_cloud_ground[:,1] > y_min) & (point_cloud_ground[:,1] < y_max)

# Transform points to align with aerial map
point_cloud_ground = point_cloud_ground[pts_to_keep_inds] + [-2.5,14.5,0.6]
pcd_ground.points = o3d.utility.Vector3dVector(point_cloud_ground)
pcd_ground.colors = o3d.utility.Vector3dVector(np.asarray(pcd_ground.colors)[pts_to_keep_inds])
# pcd_ground = pcd_ground.voxel_down_sample(.05)
pcd_ground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# o3d.visualization.draw_geometries([pcd_aerial])
# o3d.visualization.draw_geometries([pcd_ground])
# o3d.visualization.draw_geometries([pcd_aerial, pcd_ground])

# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_aerial, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([pcd_aerial, rec_mesh])

def normalize_array(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def compute_confidences(dists, dist_penalty_ramp=10, alpha=.5, normals=None, vehicle=None): # https://www.desmos.com/calculator/i9oydbtv2i
    dists[dists>1] = 1 # don't continue to penalize points that are extremely far away (in this case one point cloud is probably smaller than the other)
    closest, farthest = dists.min(), dists.max()
    lowest = np.power(1/(farthest - closest), 1/dist_penalty_ramp)
    confidences = np.power(1/(np.abs(dists) + lowest), dist_penalty_ramp)/farthest
    if normals is None:
        return normalize_array(confidences) # normalize min-max to 0-1 before returning
    else:
        if vehicle == "car": # Higher confidence for points with normals facing sideways
            combined_confidences = alpha*confidences + (1-alpha)*np.abs(np.linalg.norm(normals[:,:-1], axis=1))
            return normalize_array(combined_confidences)
        elif vehicle == "drone": # Higher confidence for points with normals facing up
            combined_confidences = alpha*confidences + (1-alpha)*np.abs(normals[:,-1])
            return normalize_array(combined_confidences)
        else:
            print("unknown vehicle type, ignoring normals")
            return normalize_array(confidences)

# Set color of each pixel in ground point cloud based on proximity to aerial point cloud
dists = np.array(pcd_ground.compute_point_cloud_distance(pcd_aerial))
ground_confidences = compute_confidences(dists, alpha=.55, normals=np.array(pcd_ground.normals), vehicle="car")
original_ground_colors = np.array(pcd_ground.colors)
colors = np.zeros_like(original_ground_colors)
for i, dist in enumerate(dists):
    colors[i] = np.array([1-ground_confidences[i], ground_confidences[i], 0])
pcd_ground.colors = o3d.utility.Vector3dVector(colors)


# Set color of each pixel in aerial point cloud based on proximity to ground point cloud
dists = np.array(pcd_aerial.compute_point_cloud_distance(pcd_ground))
aerial_confidences = compute_confidences(dists, alpha=.5, normals=np.array(pcd_aerial.normals), vehicle="drone")
original_aerial_colors = np.array(pcd_aerial.colors)
colors = np.zeros_like(original_aerial_colors)
for i, dist in enumerate(dists):
    colors[i] = np.array([1-aerial_confidences[i], aerial_confidences[i], 0])
pcd_aerial.colors = o3d.utility.Vector3dVector(colors)


# o3d.visualization.draw_geometries([pcd_aerial])
# o3d.visualization.draw_geometries([pcd_ground])
# o3d.visualization.draw_geometries([pcd_aerial, pcd_ground])


# pcd_aerial.colors = o3d.utility.Vector3dVector(original_aerial_colors)
# pcd_ground.colors = o3d.utility.Vector3dVector(original_ground_colors)

# Only keep high confidence points in each cloud
inds_to_keep = np.where(ground_confidences>.45)[0]
pcd_ground = pcd_ground.select_by_index(inds_to_keep)
inds_to_keep = np.where(aerial_confidences>.45)[0]
pcd_aerial = pcd_aerial.select_by_index(inds_to_keep)


o3d.visualization.draw_geometries([pcd_ground])
o3d.visualization.draw_geometries([pcd_aerial])
o3d.visualization.draw_geometries([pcd_aerial, pcd_ground])



pcd_discrepancy_combined = pcd_aerial + pcd_ground
# pcd_discrepancy_combined = pcd_discrepancy_combined.voxel_down_sample(.05)
# o3d.visualization.draw_geometries([pcd_discrepancy_combined])


# pdb.set_trace()

# Convert to voxel grid
voxel_grid_discrepancy_map = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_discrepancy_combined, voxel_size=0.07)
# o3d.visualization.draw_geometries([voxel_grid_discrepancy_map])


# aerial_voxel_indices = []
# ground_voxel_indices = []
# for voxel in voxel_grid_aerial.get_voxels():
#     aerial_voxel_indices.append(voxel.grid_index.tolist())
# for voxel in voxel_grid_ground.get_voxels():
#     ground_voxel_indices.append(voxel.grid_index.tolist())

# pdb.set_trace()
# # o3d.visualization.draw_geometries([voxel_grid_ground, voxel_grid_aerial])
# voxels_ground = voxel_grid_ground.get_voxels()
# voxel_grid_ground.clear()
# for voxel in voxels_ground:
#     # if voxel.grid_index.tolist() in aerial_voxel_indices:
#     #     voxel.color = [0,1,0]
#     # else:
#     #     voxel.color = [1,0,0]
#     voxel.color = [0,1,0]
#     voxel_grid_ground.add_voxel(voxel)
        
# # for voxel in voxel_grid_aerial.get_voxels():
# #     if voxel.grid_index.tolist() in ground_voxel_indices:
# #         voxel.color = [0,1,0]
# #     else:
# #         voxel.color = [1,0,0]

# pdb.set_trace()


# voxel_grid_aerial.origin = [0,0,0] # breaks manual alignment
# voxel_grid_ground.origin = [0,0,0] # breaks manual alignment
# voxel_grid_combined = voxel_grid_aerial + voxel_grid_ground
# o3d.visualization.draw_geometries([voxel_grid_combined])

# # pdb.set_trace()

# pcd_combined = pcd_aerial + pcd_ground

# voxel_grid_comb = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_combined, voxel_size=0.05)

# o3d.visualization.draw_geometries([voxel_grid_comb])
        