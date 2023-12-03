import numpy as np
import open3d as o3d
from mod_M2DP import M2DP, custom_descriptor, custom_descriptor_color
import time
# from cM2DP import M2DP
import sklearn
import matplotlib.pyplot as plt
import copy
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

def split_cloud_ids(cloud, xy, e_height=0, return_cloud=False):
    box_min_x = xy[0] - M/2
    box_max_x = xy[0] + M/2
    box_min_y = xy[1] - M/2
    box_max_y = xy[1] + M/2

    # Extract the points within the current box
    mask = (cloud[:,2]< HEIGHT_CLIP - e_height) & (cloud[:, 0] >= box_min_x) & (cloud[:, 0] < box_max_x) & (cloud[:, 1] >= box_min_y) & (cloud[:, 1] < box_max_y)

    return mask


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

# def get_candidates(query, descs, n=5, self=False):
#     # dists = np.linalg.norm(descs-query,axis=1)
#
#     sims = sklearn.metrics.pairwise.cosine_similarity(query.reshape(1,-1),descs)[0]
#
#     if self:
#         ids = np.argsort(sims)[::-1][1:n+1]
#         certs = sims[ids]
#     else:
#         ids = np.argsort(sims)[::-1][:n]
#         certs = sims[ids]
#
#     # certs -= np.min(certs)
#     # certs /= np.max(certs)
#     # print(certs)
#     return ids, certs


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
    pcd = pcd.voxel_down_sample(.06)
    point_cloud = np.array(pcd.points)
    point_colors = np.array(pcd.colors)
    min = np.min(point_cloud[:,2])
    point_cloud[:,2] -= min
    pcd.translate([0,0,-min])


    # # qbox = o3d.io.read_point_cloud('realsense_sample/t1/7.pts')
    # # qbox = o3d.io.read_point_cloud('realsense_sample/t3/18.pts')
    # qbox = o3d.io.read_point_cloud('realsense_sample/t2/20.pts')
    # o3d.visualization.draw_geometries([qbox])
    #
    # qpts = np.array(qbox.points)
    # tqpts = qpts.copy()
    # qpts[:,1] = tqpts[:,2]
    # qpts[:,2] = -tqpts[:,1]
    # qpts[:,2] -= np.min(qpts[:,2])
    # qpts = qpts[qpts[:,1] < M]
    # qpts = qpts[np.abs(qpts[:,0]) < M/2]
    # qpts = qpts[qpts[:,2] < HEIGHT_CLIP]
    #
    # ogpoints = np.load('kiss_clouds/550.npy')#[::50]
    # qbox = o3d.geometry.PointCloud()
    # qbox.points = o3d.utility.Vector3dVector(ogpoints)
    qbox = o3d.io.read_point_cloud('office2.pts')
    ogqbox = copy.deepcopy(qbox)
    xy = [-2.1,.2]
    # qbox = o3d.io.read_point_cloud('office3.pts')
    # xy = [2.1,1.2]
    # qbox = qbox.voxel_down_sample(.005)
    qbox = qbox.voxel_down_sample(.06)
    qbox, ind = qbox.remove_statistical_outlier(nb_neighbors=50,
                                                        std_ratio=2.0)

    o3d.visualization.draw_geometries([qbox])


    qpts = np.array(qbox.points)
    qcolors = np.array(qbox.colors)
    mask = split_cloud_ids(qpts,xy,e_height = 1.65)
    qpts = qpts[mask]
    qmean = qpts.mean(axis=0)
    qpts[:,0] -= qmean[0]
    qpts[:,1] -= qmean[1]
    ogqbox.translate([-qmean[0],-qmean[1],0])
    qcolors = qcolors[mask]
    qbox.points = o3d.utility.Vector3dVector(qpts)
    qbox.colors = o3d.utility.Vector3dVector(qcolors)
    o3d.visualization.draw_geometries([qbox])
    # qpts = np.array(qbox.points)
    # tqpts = qpts.copy()
    # qpts[:,1] = tqpts[:,2]
    # qpts[:,2] = -tqpts[:,1]
    # qpts[:,2] -= np.min(qpts[:,2])
    # qpts[:,0] -= qpts[:,0].mean()
    # qpts[:,1] -= qpts[:,1].mean()
    # # print(np.mean(qpts))
    # qpts = qpts[np.abs(qpts[:,1]) < M/2]
    # qpts = qpts[np.abs(qpts[:,0]) < M/2]
    # # qpts = qpts[qpts[:,2] < HEIGHT_CLIP-.6]
    # qpts = qpts[qpts[:,2] < HEIGHT_CLIP]
    # qpts[:,2] += .5

    # tqpts = qpts.copy()
    # qpts[:,0] = -tqpts[:,1]
    # qpts[:,1] = tqpts[:,0]

    # qpts = qpts[::10]
    #TODO more processing, downsample?

    qbox.points = o3d.utility.Vector3dVector(qpts)

    #cross check against similar location crop
    xy = [-5,15]
    # xy = [-3,-1]
    mask = split_cloud_ids(point_cloud, xy, e_height=0)
    cpts = point_cloud[mask]
    ccolors = point_colors[mask]
    cpts[:,0] -= cpts[:,0].mean()
    cpts[:,1] -= cpts[:,1].mean()
    cpts[cpts[:,2]<0.8,2] = 0.8
    # print(cpts.shape)
    cbox = o3d.geometry.PointCloud()
    cbox.points = o3d.utility.Vector3dVector(cpts)
    cbox.colors = o3d.utility.Vector3dVector(ccolors)
    o3d.visualization.draw_geometries([cbox, qbox])

    o3d.visualization.draw_geometries([pcd, qbox])

    # qpts = cpts

    num_cands = 1
    # des,A = M2DP(qpts, NUMT, NUMR, NUMP, NUMQ)
    # print(qcolors)
    des = custom_descriptor_color(qpts,qcolors,HEIGHT_CLIP,num_bins=NUMBINS)
    # print(des)
    # plt.imshow(des.reshape(10,5))
    # plt.show()


    # cdes, _ = M2DP(cpts, NUMT, NUMR, NUMP, NUMQ)
    cdes = custom_descriptor_color(cpts,ccolors,HEIGHT_CLIP,num_bins=NUMBINS)
    # # plt.imshow(cdes.reshape(10,5))
    # # plt.show()
    print(des)
    print(cdes)
    # print(des-cdes)
    print(np.linalg.norm(des-cdes))
    # print(des.min(),des.max())
    # print(cdes.min(),cdes.max())
    # print(qpts.min(axis=0),qpts.max(axis=0))
    # print(cpts.min(axis=0),cpts.max(axis=0))

    # des = custom_descriptor(qpts,HEIGHT_CLIP)

    db_mask = (np.abs(coords[:,0]- -5) < 4) & (np.abs(coords[:,1]- 15) < 4)
    descs = descs[db_mask]
    coords = coords[db_mask]

    ids,certs = get_candidates(des,descs,n=num_cands)
    print(certs)


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
        color_desc = 1.0/num_cands
        box.paint_uniform_color([1-i*color_desc,0,i*color_desc])
        viz_list.append(box)

    qbox.paint_uniform_color([0,1,0])
    # qbox.translate([-0.01,-0.01,0.01])
    qbox.translate([coords[ids[0]][0]+.3,coords[ids[0]][1]+.5,1.65])

    # viz_list = [pcd, qbox] #+ boxlist
    o3d.visualization.draw_geometries(viz_list)

    reg_p2p = o3d.pipelines.registration.registration_icp(
    qbox, pcd, .02, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    qbox.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([pcd,qbox])

    ogqbox.translate([coords[ids[0]][0]+.3,coords[ids[0]][1]+.5,1.65])
    ogqbox.transform(reg_p2p.transformation)
    # ogqbox.translate([0,.3,0])
    o3d.visualization.draw_geometries([pcd,ogqbox])

    # error = test(point_cloud,descs, coords)




if __name__ == "__main__":
    main()
