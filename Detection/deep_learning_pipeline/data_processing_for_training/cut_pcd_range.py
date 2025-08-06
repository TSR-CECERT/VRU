import open3d as o3d
import numpy as np
import os

def crop_point_cloud(pcd, x_range, y_range, z_range):
    """
    根据 x、y、z 范围切割点云。

    :param pcd: open3d.geometry.PointCloud 对象
    :param x_range: x 范围 (x_min, x_max)
    :param y_range: y 范围 (y_min, y_max)
    :param z_range: z 范围 (z_min, z_max)
    :return: 切割后的 open3d.geometry.PointCloud 对象
    """
    # 获取点云数据
    points = np.asarray(pcd.points)

    # 根据范围过滤点
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )

    # 创建新的点云对象
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])

    return cropped_pcd

def main(pathorg, whichlidar):
    # 定义 x, y, z 的范围
    x_range = (-30, 62)
    y_range = (-50, 55)
    z_range = (-7, 0)
    #matrix lidar1->lidar2
    Extrinstic_matrix = np.array([[-0.918378, 0.165086, -0.359624, 28.4987],
                                [-0.169571, -0.985329, -0.0192793, -26.0743],
                                [-0.357531, 0.0432762, 0.932898, 0.348447],
                                [0, 0, 0, 1]])
    R_label_to_velo = np.array([[0,-1,0,0],
                                [1,0,0,0],
                                [0,0,1,0],
                                [0,0,0,1]])
    print('### Range reduction ###')
    # Get a list of all items in the directory
    sub_srcs = os.listdir(pathorg)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(pathorg, sub_src))]

    for sub_dir in sub_dirs:
        # if sub_src == 'Run_48': #for training data
        src = pathorg + sub_dir + '/' + whichlidar + '_pcd/strongest/'
        target = pathorg + sub_dir + '/' + whichlidar + '_pcd_reduced/strongest/'
        if not os.path.exists(target):
            os.makedirs(target)
        sub_items = os.listdir(src)
        for sub_item in sub_items:
            # 读取点云文件
            pcd = o3d.io.read_point_cloud(src+sub_item)
            if whichlidar == 'Lidar1':
                ### lidar2 = R_label_to_velo@Extrinstic_matrix@(inv(R_label_to_velo)@lidar1) ###
                pcd.transform(np.linalg.inv(R_label_to_velo))
                pcd.transform(Extrinstic_matrix)
                pcd.transform(R_label_to_velo)
            # 切割点云
            cropped_pcd = crop_point_cloud(pcd, x_range, y_range, z_range)
            # o3d.visualization.draw_geometries([cropped_pcd])
            # 保存切割后的点云
            o3d.io.write_point_cloud(target+sub_item, cropped_pcd)
    
    print(whichlidar+'range reduction finished')


if __name__ == "__main__":
    pathorg = '/home/gene/Documents/Validation Data/'
    main(pathorg, 'Lidar1')

