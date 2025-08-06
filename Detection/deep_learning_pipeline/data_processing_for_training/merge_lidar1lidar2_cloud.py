import os
import open3d as o3d

def load_pcd(file_path):
    """load PCD file"""
    return o3d.io.read_point_cloud(file_path)

def save_pcd(cloud, file_path):
    """save PCD file"""
    o3d.io.write_point_cloud(file_path, cloud)

def merge_pcd(cloud1, cloud2):
    """combine PCD files"""
    cloud_combined = cloud1 + cloud2
    return cloud_combined

def process_folders(input_folder1, input_folder2, output_folder):
    """merge file with same name"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = set(os.listdir(input_folder1))
    files2 = set(os.listdir(input_folder2))

    common_files = files1.intersection(files2)

    for file_name in common_files:
        if file_name.endswith('.pcd'):
            file_path1 = os.path.join(input_folder1, file_name)
            file_path2 = os.path.join(input_folder2, file_name)
            output_path = os.path.join(output_folder, file_name)

            cloud1 = load_pcd(file_path1)
            cloud2 = load_pcd(file_path2)

            cloud_combined = merge_pcd(cloud1, cloud2)
            save_pcd(cloud_combined, output_path)
    # print(f'pcd file saved')

def main(pathorg):
    print('### Merging ###')
    # Get a list of all items in the directory
    sub_srcs = os.listdir(pathorg)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(pathorg, sub_src))]

    for sub_dir in sub_dirs:
        # if sub_src == 'Run_48':  #for training data
        input_folder1 = pathorg+sub_dir+'/Lidar1_pcd_reduced/strongest/'
        input_folder2 = pathorg+sub_dir+'/Lidar2_pcd_reduced/strongest/'
        output_folder = pathorg+sub_dir+'/Lidar12_pcd_reduced/strongest/'

        process_folders(input_folder1, input_folder2, output_folder)
    
    print('Merge finished')

if __name__ == "__main__":
    pathorg = '/home/gene/Documents/Training Data/'
    main(pathorg)
    
