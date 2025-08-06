from math import inf
from pickle import GLOBAL
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import cv2
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import math

def draw_bounding_box(bev_image, bbox_2d, label, color=(0, 255, 0), font_scale=0.5, thickness=1):
    """ Draw bounding box on an image """
    for i in range(4):
        pt1 = tuple(map(int, bbox_2d[i]))
        pt2 = tuple(map(int, bbox_2d[(i + 1) % 4]))
        cv2.line(bev_image, pt1, pt2, color, 2)
    for i in range(4, 8):
        pt1 = tuple(map(int, bbox_2d[i]))
        pt2 = tuple(map(int, bbox_2d[(i + 1) % 8]))
        cv2.line(bev_image, pt1, pt2, color, 2)
    for i in range(4):
        pt1 = tuple(map(int, bbox_2d[i]))
        pt2 = tuple(map(int, bbox_2d[i + 4]))
        cv2.line(bev_image, pt1, pt2, color, 2)
    # Draw label
    label_position = (int(bbox_2d[0][0]), int(bbox_2d[0][1]) - 10)  # Position slightly above the first corner
    cv2.putText(bev_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_principal_directions(bev_image, center, eigenvectors, scale=50, color=(0, 0, 255), thickness=1):
    """ Draw principal directions on BEV image """
    # for vec in range(eigenvectors.shape[1]):
        # Convert 3D eigenvector to 2D line on BEV image
    end_point = (int(center[0] + eigenvectors[0] * scale), int(center[1] - eigenvectors[1] * scale))
    cv2.arrowedLine(bev_image, center, end_point, color, thickness)

def scale_change(x, y, resolution=(460, 525)):
    # Define BEV (Bird's Eye View) image range
    x_min, x_max = -30, 62
    y_min, y_max = -50, 55
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Normalize
    x_normalized = (x - x_min) / x_range * (resolution[0]-1)
    y_normalized = (y - y_min) / y_range * (resolution[1]-1)
    # Reverse the y-axis (flip vertically)
    y_normalized = resolution[1] - 1 - y_normalized

    # Ensure normalized results are within valid range
    x_normalized = np.clip(x_normalized, 0, resolution[0]-1)
    y_normalized = np.clip(y_normalized, 0, resolution[1]-1)

    # Convert normalized values to integers
    px, py = np.round(x_normalized).astype(int), np.round(y_normalized).astype(int)

    return px, py

def point_cloud_to_bev(points, resolution=(460, 525)):
    """ Convert point cloud to BEV image """
    
    # Create BEV image
    bev_image = 255 * np.ones((resolution[1], resolution[0], 3), dtype=np.uint8)

    # Projection in BEV view
    # Assume BEV view's x and y coordinates map to the image's width and height
    x, y = points[:, 0], points[:, 1]
    # Define BEV image range
    x_min, x_max = -30, 62
    y_min, y_max = -50, 55
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Normalize
    x_normalized = (x - x_min) / x_range * (resolution[0]-1)
    y_normalized = (y - y_min) / y_range * (resolution[1]-1)
    # Reverse the y-axis (flip vertically)
    y_normalized = resolution[1] - 1 - y_normalized

    # Ensure normalized results are within valid range
    x_normalized = np.clip(x_normalized, 0, resolution[0]-1)
    y_normalized = np.clip(y_normalized, 0, resolution[1]-1)

    # Convert normalized values to integers
    px, py = np.round(x_normalized).astype(int), np.round(y_normalized).astype(int)
    # Plot point cloud onto BEV image
    for i in range(len(points)):
        bev_image[py[i], px[i]] = (0, 0, 0)  # Black points
    
    return bev_image

def compute_principal_directions(points):
    # Compute the covariance matrix
    cov_matrix = np.cov(points, rowvar=False)
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # The eigenvectors correspond to the principal directions
    directions = eigenvectors[:,:2]
    combined_direction = np.mean(directions, axis=1)
    combined_direction = combined_direction / np.linalg.norm(combined_direction)  # Normalize the vector

    return eigenvectors, combined_direction

def calculate_features(bbox):

    tranformation_matrix = np.array([[0,-1,0],
                                     [0,0,-1],
                                     [1,0,0]])

    #feature extraction
    min_bound = bbox.min_bound  # (x_min, y_min, z_min)
    max_bound = bbox.max_bound  # (x_max, y_max, z_max)
    size = max_bound - min_bound  # (width, height, depth)
    bbox_2d = [
        (scale_change(min_bound[0], min_bound[1])),
        (scale_change(max_bound[0], min_bound[1])),
        (scale_change(max_bound[0], max_bound[1])),
        (scale_change(min_bound[0], max_bound[1])),
        (scale_change(min_bound[0], min_bound[1])),
        (scale_change(max_bound[0], min_bound[1])),
        (scale_change(max_bound[0], max_bound[1])),
        (scale_change(min_bound[0], max_bound[1]))
    ]
    #object height in lidar data
    height = size[1]
    #object length in lidar data
    length = max(size[0],size[2])
    #nearest distance from object points to lidar
    # get the cneter of the bbox
    center = bbox.get_center()
    px, py = scale_change(center[0], center[1])
    center2d = (int(px),int(py))
    distance = np.linalg.norm(center)
    
    return height, length, distance, bbox_2d, center2d

def get_sorted_files(directory):
    """ Return a list of files sorted by the numeric value in their names. """
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Define a function to extract numeric part from the filename
    def extract_number(filename):
        # Find all numbers in the filename
        numbers = re.findall(r'\d+', filename)
        # Return the first number found as an integer, or 0 if no numbers are found
        return int(numbers[0]) if numbers else 0
    
    # Sort files based on the numeric part of their names
    sorted_files = sorted(files, key=extract_number)
    
    return sorted_files

def limit_period(val,
                 offset: float = 0,
                 period: float = 2*np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    
    limited_val = val - np.floor(val / period + offset) * period
    limited_val = math.degrees(limited_val)
    return limited_val

def bbox_generation(bbox_axis_aligned, bbox_oriented, sub_item, name):

    R_velo_to_label = np.array([[0,1,0],
                                [-1,0,0],
                                [0,0,1]])

    #bbox structure: frame, subclass, x_center, x_length, y_center, y_length, z_center, z_length, yaw, score(calculated with distance to the 2D detected location)
    extent_axis_aligned = bbox_axis_aligned.get_extent()
    center_axis_aligned = bbox_axis_aligned.get_center()
    center_axis_aligned = R_velo_to_label@center_axis_aligned
    bbox_axis_aligned_lidar2 = [sub_item,name,center_axis_aligned[0],extent_axis_aligned[0],center_axis_aligned[1],extent_axis_aligned[1],center_axis_aligned[2],extent_axis_aligned[2],0,0]
    # bbox_axis_aligned_kitti = [sub_item, labelgt, 0, 0, 0, 0, 0, height, ]

    extent_oriented = bbox_oriented.extent
    center_oriented = bbox_oriented.center
    center_oriented = R_velo_to_label@center_oriented
    rotation_oriented = bbox_oriented.R
    raw = math.atan2(rotation_oriented[1, 0], rotation_oriented[0, 0])-np.pi/2
    raw = limit_period(raw, 0, np.pi*2)
    # raw = math.degrees(raw)
    bbox_oriented_lidar2 = [sub_item,name,center_oriented[0],extent_oriented[0],center_oriented[1],extent_oriented[1],center_oriented[2],extent_oriented[2],raw,0]

    return bbox_axis_aligned_lidar2, bbox_oriented_lidar2 


def main(path, eps, min_points, video):
    # video paras
    fps = 7
    frame_size = (460, 525)  # BEV image size

    #labels
    name_to_class = {
        'VRU_Adult_Using_Motorized_Bicycle': 0,
        'Passenger_Vehicle': 1,
        'VRU_Child': 2,
        'VRU_Adult': 3,
        'VRU_Adult_Using_Cane': 4,
        'VRU_Adult_Using_Manual_Scooter': 5,
        'VRU_Adult_Using_Crutches': 6,
        'VRU_Adult_Using_Cardboard_Box': 7,
        'VRU_Adult_Using_Walker': 8,
        'VRU_Adult_Using_Manual_Wheelchair': 9,
        'VRU_Adult_Using_Stroller': 10,
        'VRU_Adult_Using_Skateboard': 11,
        'VRU_Adult_Using_Manual_Bicycle': 12,
        }


    cluster_point_counts = {}
    bounding_boxes_axis_aligned_lidar2_allrun = {}
    bounding_boxes_oriented_lidar2_allrun = {}

    ### Classification ###
    print('### Classification ###')
    label_features = np.load('label_features_12_all.npy')
    print(f'The number of classification training data:{label_features.shape[0]}')
    X_Train = label_features[:,1:]
    Y_Train = label_features[:,0]
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    # Fitting the classifier into the Training set
    classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
    classifier.fit(X_Train,Y_Train)

    # Get a list of all items in the directory
    sub_srcs = os.listdir(path)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(path, sub_src))]
    for sub_dir in sub_dirs:
        print('----- ' + sub_dir + ' -----')
        print('### Clustering and Classifier Testing ###')
        # bbox save
        bounding_boxes_axis_aligned_lidar2 = []
        bounding_boxes_oriented_lidar2 = []
        # path
        pathorg = path+sub_dir+'/Lidar12_pcd_filtered/strongest/'
        # create video
        if video == True:
            out = cv2.VideoWriter(path+sub_dir+'/'+sub_dir+'clustering_bev_test.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
        # sort frame order
        sub_items = get_sorted_files(pathorg)
        for sub_item in sub_items:

            ### Clustering ###
            pcd = o3d.io.read_point_cloud(pathorg+sub_item)
            points = np.asarray(pcd.points)
            # Add a small perturbation to avoid numerical precision issues
            perturbation = 1e-10
            points += np.random.uniform(-perturbation, perturbation, points.shape)
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(
                    pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
            if len(labels) == 0:
                continue

            # calculate BEV
            bev_image = point_cloud_to_bev(points, resolution=frame_size)
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # jump outliers
                cluster_points = points[labels == label]
                if len(cluster_points) <= 4:
                    continue

                ### BBox Generation ###
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                bbox_axis_aligned = cluster_pcd.get_axis_aligned_bounding_box()
                bbox_oriented = cluster_pcd.get_oriented_bounding_box()

                ### Features ###
                #feature: cluster_point_count
                cluster_point_counts[label] = np.sum(labels == label)
                #feature: principal_directions
                principal_directions, combine_direction = compute_principal_directions(cluster_points)
                principal_directions = principal_directions.flatten().tolist()
                #feature: height, length, distance
                height, length, distance, bbox_2d, center2d = calculate_features(bbox_axis_aligned)
                feature = np.array([height, length, distance, cluster_point_counts[label]]+ principal_directions)

                #### Classifier Testing ###
                X_Test = feature.reshape(1,-1)
                X_Test = sc_X.transform(X_Test)
                # Predicting the test set results
                labelgt = classifier.predict(X_Test)
                name = next((n for n,c in name_to_class.items() if labelgt == c), None)

                ### Draw Bounding Box in Video
                if labelgt != -1:
                    if video == True:
                        bbox_color = (0, 255, 0) #green
                        # draw bounding box
                        draw_bounding_box(bev_image, bbox_2d, name, bbox_color)
                        draw_principal_directions(bev_image, center2d, combine_direction, scale=30, color=bbox_color)
                    # bbox generation output in label format (lidar2 coordinate)
                    bbox_axis_aligned_lidar2, bbox_oriented_lidar2 = bbox_generation(bbox_axis_aligned, bbox_oriented, sub_item, name)
                    bounding_boxes_axis_aligned_lidar2.append(bbox_axis_aligned_lidar2)
                    bounding_boxes_oriented_lidar2.append(bbox_oriented_lidar2)
                
                else:
                    if video == True:
                        bbox_color = (0, 0, 255) #red
                        draw_bounding_box(bev_image, bbox_2d, name, bbox_color)
                        draw_principal_directions(bev_image, center2d, combine_direction, scale=30, color=bbox_color)

            # write to image in video
            if video == True:
                out.write(bev_image)
        if video == True:
            print('### Saving BEV Videos ###' )
            out.release()
            cv2.destroyAllWindows()

        # save bbox
        print('### Saving Detections ###' )
        outputpath = path+sub_dir+'/detections/'
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        with open(outputpath+sub_dir+'_detections_axis_aligned_lidar2.txt', 'w') as file:
            for sublist in bounding_boxes_axis_aligned_lidar2:
                file.write(', '.join(map(str, sublist)) + '\n')
        with open(outputpath+sub_dir+'_detections_oriented_lidar2.txt', 'w') as file:
            for sublist in bounding_boxes_oriented_lidar2:
                file.write(', '.join(map(str, sublist)) + '\n')
        bounding_boxes_axis_aligned_lidar2_allrun[sub_dir] = bounding_boxes_axis_aligned_lidar2
        bounding_boxes_oriented_lidar2_allrun[sub_dir] = bounding_boxes_oriented_lidar2

        print('----- '+sub_dir+' finished -----')
    return bounding_boxes_axis_aligned_lidar2_allrun, bounding_boxes_oriented_lidar2_allrun
        
if __name__ == "__main__":
    path = '/home/gene/Documents/Validation Data2/'
    main(path, eps = 0.7, min_points = 4, video=False)


