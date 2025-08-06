import numpy as np
from pypcd import pypcd
import os

def main(src):
    # Get a list of all items in the directory
    sub_srcs = os.listdir(src)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(src, sub_src))]
    for sub_dir in sub_dirs:
            # if sub_dir == '?Run_465': #for training data
            target = src+sub_dir+'/Lidar12_bin_reduced/strongest'
            if not os.path.exists(target):
                os.makedirs(target)
            for root, _, fs in os.walk(src+sub_dir+'/Lidar12_pcd_reduced/strongest'):
                for f in fs:
                    path = os.path.join(root, f)
                    pcd_data = pypcd.PointCloud.from_path(path)
                    points = np.zeros([pcd_data.width, 4], dtype=np.float32)
                    points[:, 0] = pcd_data.pc_data['x'].copy()
                    points[:, 1] = pcd_data.pc_data['y'].copy()
                    points[:, 2] = pcd_data.pc_data['z'].copy()
                    # points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
                    with open(src+sub_dir+f'/Lidar12_bin_reduced/strongest/{f[:-3]}bin', 'wb') as a:
                        a.write(points.tobytes())

if __name__ == "__main__":
    # input your folder location
    src = '/home/gene/Documents/Validation Data2/'
    main(src)

