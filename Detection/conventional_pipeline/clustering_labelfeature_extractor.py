from math import inf
from pickle import GLOBAL
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def compute_principal_directions(points):
    # Compute the covariance matrix
    print(points)
    cov_matrix = np.cov(points, rowvar=False)
    # print(cov_matrix)
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    # print(eigenvectors)
    # The eigenvectors correspond to the principal directions
    return eigenvectors

def calculate_features(bbox, txtlabel):
    global Y_real, Y_pred
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

    tranformation_matrix = np.array([[0,-1,0],
                                     [0,0,-1],
                                     [1,0,0]])

    #feature extraction
    min_bound = bbox.min_bound  # (x_min, y_min, z_min)
    max_bound = bbox.max_bound  # (x_max, y_max, z_max)
    size = max_bound - min_bound  # (width, height, depth)
    #object height in lidar data
    height = size[1]
    #object length in lidar data
    length = max(size[0],size[2])
    #nearest distance from object points to lidar
    #get center
    center = bbox.get_center()
    distance = np.linalg.norm(center)
    #read_labels
    labelgt = -1
    a = inf
    with open(txtlabel, 'r') as file:
        lines = csv.reader(file, delimiter=' ')
        objectnum = 0
        for line in lines:
            objectnum += 1 
            centergt = np.array([float(line[11]),float(line[12]),float(line[13])])
            variance = np.array([0,0,float(line[8])/2])
            centergt = centergt@tranformation_matrix+variance
            if a>np.linalg.norm(centergt-center):
                a = np.linalg.norm(centergt-center)
                
                if np.linalg.norm(centergt-center) < 2:
                    name = line[0]
                    for n,c in name_to_class.items():
                        if name == n:
                            labelgt = c
                            print(c)
    if labelgt != -1:
        Y_real.append(1)
        Y_pred.append(1)
    else:
        Y_real.append(0)
        Y_pred.append(1)
    
    return height, length, distance, labelgt, objectnum

def main(path):

    bounding_boxes = []
    pcds = []
    features = []
    cluster_point_counts = {}
    # Get a list of all items in the directory
    sub_srcs = os.listdir(path)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(path, sub_src))]
    for sub_dir in sub_dirs:

        pathorg = path+sub_dir+'/Lidar12_pcd_filtered/strongest/'
        pathlabel = path+sub_dir+'/Kitti_GT/'
        txtlabelitems = os.listdir(pathlabel)
        for txtlabelitem in txtlabelitems:
            print(txtlabelitem)
            txtlabel = pathlabel+txtlabelitem
            sub_item = txtlabelitem[:-3]+'pcd'
            #cluster
            pcd = o3d.io.read_point_cloud(pathorg+sub_item)
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(
                    pcd.cluster_dbscan(eps=0.7, min_points=4, print_progress=False))
            # print(labels)
            if len(labels) == 0:
                continue
            max_label = labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            max_label = labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            # print(colors)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            points = np.asarray(pcd.points)
            # print(len(points))
            unique_labels = np.unique(labels)
            count = 0
            for label in unique_labels:
                if label == -1:
                    continue  # jump outlier
                cluster_points = points[labels == label]
                if len(cluster_points) == 1:
                    continue
                cluster_point_counts[label] = np.sum(labels == label)
                principal_directions = compute_principal_directions(cluster_points)
                principal_directions = principal_directions.flatten().tolist()
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                bbox.color = (0,0,0)
                height, length, distance, labelgt, objectnum = calculate_features(bbox, txtlabel)
                feature = [labelgt, height, length, distance, cluster_point_counts[label]]+ principal_directions
                features.append(feature)
                if labelgt != -1:
                    count += 1
                    bbox.color = (255,0,0)
                bounding_boxes.append(bbox)
            for i in range(objectnum-count):
                Y_real.append(1)
                Y_pred.append(0)
            # if txtlabelitem == '000575.txt':
            #     o3d.visualization.draw_geometries([pcd])
            pcds.append([sub_dir, sub_item])
            # bounding_boxes_all.append(bounding_boxes)
    features = np.array(features)
    np.save('label_features_12_all.npy', features)
            # print(features)
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_real, Y_pred)
    print(cm)

#used to determine the comfusion matrix of clustering
Y_pred = []
Y_real = []

if __name__ == "__main__":
    path = '/home/gene/Documents/Validation Data2/'
    main(path)


