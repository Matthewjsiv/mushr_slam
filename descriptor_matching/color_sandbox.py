import numpy as np
import open3d as o3d
from mod_M2DP import M2DP, custom_descriptor, custom_descriptor_color
# from cM2DP import M2DP

import time

pcd = o3d.io.read_point_cloud('point_cloud.pts')
o3d.visualization.draw_geometries([pcd])

point_cloud = np.array(pcd.points)
point_cloud[:,2] -= np.min(point_cloud[:,2])


lift = 0.8
point_cloud[point_cloud[:,2]<lift,2] = lift

pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd = pcd.voxel_down_sample(.06)

o3d.visualization.draw_geometries([pcd])

point_cloud = np.array(pcd.points)
point_colors = np.array(pcd.colors)

# Box size in X and Y dimensions (M)
#HYPERPARAMS
M = 3.5
HEIGHT_CLIP = 1.8
STRIDE = 0.25
# number of bins in theta, the 't' in paper
NUMT = 16
#number of bins in rho, the 'l' in paper
NUMR = 16
# number of azimuth angles, the 'p' in paper
NUMP = 3
# number of elevation angles, the 'q' in paper
NUMQ = 3
NUMBINS=8

# Find the minimum and maximum coordinates in X and Y dimensions
min_x, min_y = np.min(point_cloud[:, 0:2], axis=0)
max_x, max_y = np.max(point_cloud[:, 0:2], axis=0)

# Calculate the number of boxes in X and Y dimensions
num_x_boxes = int(np.ceil((max_x - min_x) / STRIDE))
num_y_boxes = int(np.ceil((max_y - min_y) / STRIDE))


descs = []
coords = []

# Iterate through the boxes
for i in range(num_x_boxes):
    for j in range(num_y_boxes):
        # Define the box's boundaries
        box_min_x = min_x + i * STRIDE
        box_max_x = min_x + i * STRIDE + M
        box_min_y = min_y + j * STRIDE
        box_max_y = min_y + j * STRIDE + M

        # Extract the points within the current box
        # box_points = point_cloud[
        #     (point_cloud[:, 0] >= box_min_x) & (point_cloud[:, 0] < box_max_x) &
        #     (point_cloud[:, 1] >= box_min_y) & (point_cloud[:, 1] < box_max_y)
        # ]
        #
        # #Exclude
        # # box_points = point_cloud[
        # #     (point_cloud[:, 0] < box_min_x) | (point_cloud[:, 0] >= box_max_x) |
        # #     (point_cloud[:, 1] < box_min_y) | (point_cloud[:, 1] >= box_max_y)
        # # ]
        # # print(np.max(box_points[:,2]))
        # box_points = box_points[box_points[:,2] < HEIGHT_CLIP]


        mask = (point_cloud[:, 0] >= box_min_x) & (point_cloud[:, 0] < box_max_x) & \
        (point_cloud[:, 1] >= box_min_y) & (point_cloud[:, 1] < box_max_y) & (point_cloud[:,2] < HEIGHT_CLIP)

        box_points = point_cloud[mask]
        box_colors = point_colors[mask]
        # print(box_colors)

        # print(len(box_points))
        # if len(box_points) > 1000:
        if len(box_points) > 1000:
            now = time.perf_counter()
            # des,A = M2DP(box_points, NUMT, NUMR, NUMP, NUMQ)
            # des = custom_descriptor(box_points,HEIGHT_CLIP,num_bins=NUMBINS)
            des = custom_descriptor_color(box_points,box_colors,HEIGHT_CLIP,num_bins=NUMBINS)
            # print(des.shape)

            if j % 100 == 0:
                print(time.perf_counter() - now)
            # print(des.shape)
            descs.append(des)
            coords.append([np.mean([box_min_x,box_max_x]), np.mean([box_min_y,box_max_y])])
        # print(box_points.shape)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(box_points)
        # o3d.visualization.draw_geometries([pcd])

descs = np.array(descs)
coords = np.array(coords)
print(descs.shape, coords.shape)
np.save('descriptors',descs)
np.save('coords',coords)
