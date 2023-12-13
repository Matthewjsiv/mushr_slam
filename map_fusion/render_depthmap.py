import open3d as o3d
import numpy as np
import pdb

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


vis = o3d.visualization.Visualizer()
vis.create_window(width=300, height=300, visible=True)
vis.add_geometry(mesh)
# vis.get_render_option().mesh_show_back_face = True
ctr = vis.get_view_control()
param = ctr.convert_to_pinhole_camera_parameters()

vis.run()
vis.destroy_window()

pdb.set_trace()

vis.poll_events()
vis.update_renderer()

depth = vis.capture_depth_float_buffer(True)
pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), param.intrinsic, param.extrinsic, depth_scale=1)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd, mesh])