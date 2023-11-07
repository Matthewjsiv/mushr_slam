import numpy as np
import open3d as o3d
from mod_M2DP import M2DP
import time
# from cM2DP import M2DP


#HYPERPARAMS
M = 2.5
HEIGHT_CLIP = 1.25
STRIDE = 0.25
# number of bins in theta, the 't' in paper
NUMT = 16
#number of bins in rho, the 'l' in paper
NUMR = 16
# number of azimuth angles, the 'p' in paper
NUMP = 2
# number of elevation angles, the 'q' in paper
NUMQ = 1

def split_cloud(cloud, xy, return_cloud=False):
    box_min_x = xy[0] - M/2
    box_max_x = xy[0] + M/2
    box_min_y = xy[1] - M/2
    box_max_y = xy[1] + M/2

    # Extract the points within the current box
    inc_points = cloud[
        (cloud[:, 0] >= box_min_x) & (cloud[:, 0] < box_max_x) &
        (cloud[:, 1] >= box_min_y) & (cloud[:, 1] < box_max_y)
    ]
    inc_points = inc_points[inc_points[:,2] < HEIGHT_CLIP]

    include = o3d.geometry.PointCloud()
    include.points = o3d.utility.Vector3dVector(inc_points)

    # Exclude
    exc_points = cloud[
        (cloud[:, 0] < box_min_x) | (cloud[:, 0] >= box_max_x) |
        (cloud[:, 1] < box_min_y) | (cloud[:, 1] >= box_max_y)
    ]

    exclude = o3d.geometry.PointCloud()
    exclude.points = o3d.utility.Vector3dVector(exc_points)

    if return_cloud:
        return include, exclude
    else:
        return inc_points, exc_points


def get_candidates(query, descs, n=5, self=False):
    dists = np.linalg.norm(descs-query,axis=1)
    if self:
        ids = np.argsort(dists)[1:n+1]
        certs = dists[ids]
    else:
        ids = np.argsort(dists)[:n]
        certs = dists[ids]

    # certs -= np.min(certs)
    # certs /= np.max(certs)
    # print(certs)
    return ids, certs


def test(point_cloud,descs, coords):

    iters = 100
    dists = 0
    times = 0
    for i in range(iters):
        min_x, min_y = np.min(point_cloud[:, 0:2], axis=0)
        max_x, max_y = np.max(point_cloud[:, 0:2], axis=0)
        samplx = np.random.uniform(low=min_x + 1, high=max_x-1, size=(1,))
        samply = np.random.uniform(low=min_y+1, high=max_y-1, size=(1,))
        xy = [samplx[0],samply[0]]

        qpts, _ = split_cloud(point_cloud, xy)
        if len(qpts) < 300:
            i -= 1
            continue

        now = time.perf_counter()
        des,A = M2DP(qpts, NUMT, NUMR, NUMP, NUMQ)
        id,certs = get_candidates(des,descs,n=1)
        # print(time.perf_counter() - now)
        times += time.perf_counter() - now
        # print(np.linalg.norm(coords[id]-xy))
        dists += np.linalg.norm(coords[id]-xy)

    avgdist = dists/iters
    avgtime = times/iters
    print("AVG ERROR: ", avgdist)
    print("AVG RUNTIME: ", avgtime)

    return avgdist

def main():
    descs = np.load('descriptors.npy')
    coords = np.load('coords.npy')
    pcd = o3d.io.read_point_cloud('point_cloud.pts')
    point_cloud = np.array(pcd.points)
    min = np.min(point_cloud[:,2])
    point_cloud[:,2] -= min
    pcd.translate([0,0,-min])

    # q = 30
    # ids,certs = get_candidates(descs[q],descs, self=True)
    # qbox, _ = split_cloud(point_cloud, coords[q], return_cloud = True)

    # xy = [-.5,-3]
    # xy = [-3,-2]
    # Find the minimum and maximum coordinates in X and Y dimensions
    min_x, min_y = np.min(point_cloud[:, 0:2], axis=0)
    max_x, max_y = np.max(point_cloud[:, 0:2], axis=0)
    samplx = np.random.uniform(low=min_x, high=max_x, size=(1,))
    samply = np.random.uniform(low=min_y, high=max_y, size=(1,))
    xy = [samplx[0],samply[0]]

    qpts, _ = split_cloud(point_cloud, xy)
    qbox = o3d.geometry.PointCloud()
    qbox.points = o3d.utility.Vector3dVector(qpts)

    des,A = M2DP(qpts, NUMT, NUMR, NUMP, NUMQ)
    ids,certs = get_candidates(des,descs)


    # boxlist = []
    viz_list = [pcd,qbox]
    for i in range(len(ids)):
        id = ids[i]
        cert = certs[i]
        box, _ = split_cloud(point_cloud, coords[id], return_cloud = True)
        # box.estimate_normals()
        box.translate([0.01,0.01,0.01])
        # box.paint_uniform_color([1-cert,0,cert])
        #assuming ids are sorted by similarity
        box.paint_uniform_color([1-i*.2,0,i*.2])
        viz_list.append(box)

    qbox.paint_uniform_color([0,1,0])
    qbox.translate([-0.01,-0.01,0.01])

    # viz_list = [pcd, qbox] #+ boxlist
    o3d.visualization.draw_geometries(viz_list)

    error = test(point_cloud,descs, coords)


if __name__ == "__main__":
    main()
