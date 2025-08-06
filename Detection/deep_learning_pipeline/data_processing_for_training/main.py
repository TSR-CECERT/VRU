# Run this script to preprocess the ISC data into Kitti data for training in MMdet3D
import sys

from isc2kitti import main as isc2kitti
from cut_pcd_range import main as cut_pcd_range
from merge_lidar1lidar2_cloud import main as merge_lidar1lidar2_cloud
from pcd2bin import main as pcd2bin
from isc_rename_bin_moveto_isc_dataset import main as isc_rename_bin_moveto_isc_dataset
from isc_imagesets_generation import main as isc_imagesets_generation
from isc_rename_calib import main as isc_rename_calib

def main():
    # Check command line arguments -- folder location of testing data
    if len(sys.argv) < 3:
        print("Please provide the folder location of source testing data and target testing data with '\\' before blank and '/' in the end.\nUsage: python test.py <src_path> <target_path>\nI.e., /home/gene/Documents/Validation\\ Data2/ /home/gene/mmdetection3d/data/isc_full/")
        sys.exit(1)  # Exit the program with status code 1 indicating an error

    # Get the first command line argument
    loc = sys.argv[1]
    target = sys.argv[2]

    # Generate kitti label
    isc2kitti(loc) #annotate this line if you have generated kittiGT  

    # cut pcd range
    cut_pcd_range(loc, 'Lidar1') #annotate this line if you have gotten the range cut pcds
    cut_pcd_range(loc, 'Lidar2') 

    # merge lidar1 and lidar2
    merge_lidar1lidar2_cloud(loc) #annotate this line if you have merged the range cut pcds
     
    # pcd2bin for training in Kitti format
    pcd2bin(loc)

    # remove data to isc_full
    # Please provide the ISC loc and MMdet3d target location
    # If you are planning to generate training data, set test=0; testing data, set test=1
    isc_rename_bin_moveto_isc_dataset(loc, target, test=0)
    isc_rename_bin_moveto_isc_dataset(loc, target, test=1) # if you just want to train and validate the data, please just uncomment this line and run this script in one step； if you are processing testing data, please first unwrap the pcap under Training Data (without labels), then run cut_pcd_range->merge_lidar1lidar2_cloud->pcd2bin->isc_rename_bin_moveto_isc_dataset->isc_imagesets_generation->isc_rename_calib and annotate isc2kitti; if you meet 'UnboundLocalError: local variable 'ignore_class_name' referenced before assignment', please make sure both training and testing folder under isc_full is filled with data.

    # generate imagesets (split train and val randomly), train/test split ratio 0.8/0.2
    isc_imagesets_generation(target, split_ratio=0.8)

    # rename calib files according to imagesets
    isc_rename_calib(target)

if __name__ == "__main__":
    main()