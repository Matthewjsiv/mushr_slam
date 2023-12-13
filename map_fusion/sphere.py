import open3d as o3d

sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=10)
o3d.visualization.draw_geometries([sphere], mesh_show_wireframe=True)
o3d.io.write_triangle_mesh("test_data/sphere.ply", sphere)