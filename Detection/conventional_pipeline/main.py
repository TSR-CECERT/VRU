import sys

# from isc2kitti import main as isc2kitti
from cut_pcd_range import main as cut_pcd_range
from merge_lidar1lidar2_cloud import main as merge_lidar1lidar2_cloud
from dataset_background_filter import main as dataset_background_filter
from clustering_labelfeature_extractor import main as clustering_labelfeature_extractor
from clustering_classification_bbox_video_generation import main as clustering_classification_bbox_video_generation

def main():
    # Check command line arguments -- folder location of testing data
    if len(sys.argv) < 2:
        print("Please provide the folder location of testing data with '\\' before blank and '/' in the end.\nUsage: python test.py <path>")
        sys.exit(1)  # Exit the program with status code 1 indicating an error

    # Get the first command line argument
    loc = sys.argv[1]

    # isc2kitti
    # isc2kitti(loc) #annotate this line if you have generated kittiGT in the deep learning pipeline, this kitti label is used for training the classifier

    # cut pcd range
    cut_pcd_range(loc, 'Lidar1') #annotate this line if you have gotten the range cut pcds
    cut_pcd_range(loc, 'Lidar2') 

    # merge lidar1 and lidar2
    merge_lidar1lidar2_cloud(loc) #annotate this line if you have merged the range cut pcds

    # filter the background
    # filter outlier paras: scale, bigger->filter more
    # remove outlier based on radius: nb_points, radius, at least nb_points in radius
    dataset_background_filter(loc, scale=4, nb_points=4, radius=2) #annotate this line if you have filtered the background

    # get the training feature and label
    # clustering_labelfeature_extractor(loc) #annotate this line if you have gotten the label_features (label_features_12_all) 

    # cluster, classification and bbox_generation and bev video generation
    # cluster paras: eps, min_points
    # returned detections for tracking
    bounding_boxes_axis_aligned_lidar2_allrun, bounding_boxes_oriented_lidar2_allrun = clustering_classification_bbox_video_generation(loc, eps = 0.7, min_points = 4, video = False)



if __name__ == "__main__":
    main()

