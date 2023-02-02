import open3d as o3d
import numpy as np


print("Testing file for point cloud ...")
pcd = o3d.io.read_point_cloud("fountain_trICP.xyz",format='xyz')
print(pcd)
#o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([downpcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

# print("Recompute the normal of the downsampled point cloud")
# downpcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([downpcd],
#                                 #   zoom=0.3412,
#                                 #   front=[0.4257, -0.2125, -0.8795],
#                                 #   lookat=[2.6172, 2.0475, 1.532],
#                                 #   up=[-0.0694, -0.9768, 0.2024],
#                                   point_show_normal=True)



# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

# bbox = pcd.get_axis_aligned_bounding_box()
# p_mesh_crop = poisson_mesh.crop(bbox)
# p_mesh_crop.compute_vertex_normals()

# pcl = p_mesh_crop.sample_points_poisson_disk(number_of_points=30000)
# hull, _ = p_mesh_crop.compute_convex_hull()
# hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
# hull_ls.paint_uniform_color((1, 0, 0))
# pcl.paint_uniform_color([1, 0.706, 0])
# o3d.visualization.draw_geometries([pcl, hull_ls])