import os
import shutil
import random


def find_different_extension_file(folder, file_path):
    # Get the file name and extension of the given file
    base_name = os.path.basename(file_path)  # Get the file name part
    file_name, file_ext = os.path.splitext(base_name)  # Separate the file name and extension
    
    # Iterate over files in the folder
    for filename in os.listdir(folder):
        if filename.startswith(file_name) and filename != base_name:
            # Find a file with the same name but different extension
            _, ext = os.path.splitext(filename)
            if ext != file_ext:
                return True, filename  # Return True indicating a matching file exists
    return False, None  # No matching file found


def save_training_data(loc, isctrainloc, isclabelloc):
    # Get a list of all items in the directory
    sub_srcs = os.listdir(loc)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(loc, sub_src))]
    for sub_dir in sub_dirs:
        path = loc + sub_dir + '/Lidar12_bin_reduced/strongest'
        pathgt = loc + sub_dir + '/Kitti_GT/'
        sub_items = os.listdir(path)
        sub_dir = sub_dir.replace('Run_', '')
        sub_dir = sub_dir.replace('?', '')
        i = 0
        for item in sub_items:
            item_path = os.path.join(path, item)
            _, gt_item = find_different_extension_file(pathgt, item_path)
            if _:
                shutil.copy(pathgt + gt_item, isclabelloc + sub_dir + gt_item)
                # Use shutil.copy() to copy the file
                old_name = item_path
                new_name = isctrainloc + sub_dir + item
                shutil.copy(old_name, new_name)


# Testing data

def save_test_data(loc, isctestloc):
    # Get a list of all items in the directory
    sub_srcs = os.listdir(loc)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(loc, sub_src))]
    for sub_dir in sub_dirs:
        pathtest = loc + sub_dir + '/Lidar12_bin_reduced/strongest'
        sub_items = os.listdir(pathtest)
        sub_dir = sub_dir.replace('Run_', '')
        sub_dir = sub_dir.replace('?', '')
        for item in sub_items:
            item_path = os.path.join(pathtest, item)
            old_name = item_path
            new_name = isctestloc + sub_dir + item
            shutil.copy(old_name, new_name)

def main(src, target, test):
    if test == 1:
        isctestloc = target + 'testing/velodyne/'
        if not os.path.exists(isctestloc):
            os.makedirs(isctestloc)
        save_test_data(src, isctestloc)
    else:
        isctrainloc = target + 'training/velodyne/'
        isclabelloc = target + 'training/label_2/'
        if not os.path.exists(isctrainloc):
            os.makedirs(isctrainloc)
        if not os.path.exists(isclabelloc):
            os.makedirs(isclabelloc)
        save_training_data(src, isctrainloc, isclabelloc)


if __name__ == '__main__':
    target = '/home/gene/mmdetection3d/data/isc_full/'
    loc = '/home/gene/Documents/Validation Data2/'
    main(loc, target, test=0)
