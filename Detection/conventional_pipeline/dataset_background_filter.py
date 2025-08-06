import open3d as o3d
import numpy as np
import os


def filter_matching_points(cloud1, cloud2, scale):
    """
    Filter out the points that are matched between two point clouds.
    """

    # Get correspondence distances
    distances = cloud1.compute_point_cloud_distance(cloud2)
    threshold = np.mean(distances) + scale * np.std(distances)

    # Filter out points based on distance threshold
    filtered_indices = np.where(distances > threshold)[0]

    # Filtered point clouds
    filtered_cloud1 = cloud1.select_by_index(filtered_indices)

    return filtered_cloud1

def delete_objects_from_cloud(cloud, cloudob):
    distances = np.asarray(cloud.compute_point_cloud_distance(cloudob))
    threshold = 0.5

    # Filter out points based on distance threshold
    filtered_indices = np.where(distances > threshold)[0]
    filtered_cloud = cloud.select_by_index(filtered_indices)

    return filtered_cloud

def main(pathorg, scale, nb_points, radius):
    print('### Background filtering ###')
    voxeldis = 0.01
    # Get a list of all items in the directory
    sub_srcs = os.listdir(pathorg)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(pathorg, sub_src))]
    for sub_dir in sub_dirs:
        # if sub_src == 'Run_48': #for training data
        src = pathorg + sub_dir + '/Lidar12_pcd_reduced/strongest/'
        target = pathorg + sub_dir + '/Lidar12_pcd_filtered/strongest/'
        if not os.path.exists(target):
            os.makedirs(target)
        sub_items = os.listdir(src)
        nb_items = len(sub_items)
        #calculate background outlier
        # Load point clouds
        cloud1 = o3d.io.read_point_cloud(src+'000010.pcd')
        cloud2 = o3d.io.read_point_cloud(src+str((nb_items-10)).zfill(6)+'.pcd')
        cloud1_down = cloud1.voxel_down_sample(voxeldis)
        cloud2_down = cloud2.voxel_down_sample(voxeldis)
        # Filter out matching points
        filtered_cloud1 = filter_matching_points(cloud1_down, cloud2_down, scale = 7)
        cl1, ind1 = filtered_cloud1.remove_radius_outlier(nb_points=3, radius=1)
        outlier_cloud = delete_objects_from_cloud(cloud1_down,cl1)
                
        for sub_item in sub_items:            
            cloud3 = o3d.io.read_point_cloud(src+sub_item)
            cloud3_down = cloud3.voxel_down_sample(voxeldis)
            filtered_cloud3 = filter_matching_points(cloud3_down, outlier_cloud, scale = scale)
            cl3, ind3 = filtered_cloud3.remove_radius_outlier(nb_points=nb_points, radius=radius)
            # cl3, _ = cl3.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.5)
            # o3d.visualization.draw_geometries([cl3])
            o3d.io.write_point_cloud(target+sub_item, cl3)

    print('Background filtered')

if __name__ == "__main__":
    pathorg = '/home/gene/Documents/Validation Data/'
    main(pathorg, scale=4, nb_points=4, radius=2)
