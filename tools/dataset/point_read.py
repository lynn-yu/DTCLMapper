import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import laspy
# las = laspy.read('/data1/lsy/20231025163547103.las')
# point_data = np.stack([las.x, las.y, las.z], axis=0).transpose((1, 0))
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_data)
# xyz = np.asarray(pcd.points)
pcd_path = "/data1/lsy/1.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
xyz = np.asarray(pcd.points,dtype=np.float32)
#o3d.visualization.draw_geometries([pcd], 'part of cloud', width=500, height=500)
# xyz[:,0]= xyz[:,0]-7056
# xyz[:,1] = xyz[:,1] - 30806
# x1 = np.where(xyz[:,0]<-30)
# xyz_de = np.delete(xyz,x1,0)
# x2 = np.where(xyz_de[:,0] > 30)
# xyz_de = np.delete(xyz_de,x2,0)
# print(xyz_de.shape)
ig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})
num=int(len(xyz)/10)
axs[0].scatter(xyz[:num, 0], xyz[:num, 1], xyz[:num, 2], c='r', marker='o', s=1)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_zlabel('Z')
axs[0].set_title('Original Point Cloud')
axs[0].text2D(0.05, 0.95, f'Points: {num}', transform=axs[0].transAxes)
# Display the figure
plt.show()

