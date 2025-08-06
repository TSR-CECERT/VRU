from math import pi
import numpy as np
# mmdet3d apis
from mmdet3d.structures import (LiDARInstance3DBoxes,Box3DMode)
from mmdet3d.structures import limit_period

import sys
import os

def main(loc):
    R_label_to_velo = np.array([[0,1,0],
                            [-1,0,0],
                            [0,0,1]])
    # Get a list of all items in the directory
    sub_srcs = os.listdir(loc)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(loc, sub_src))]
    for sub_dir in sub_dirs:
        path = loc+'/'+sub_dir+'/'
        if not os.path.exists(path+'Kitti_GT'):
            os.makedirs(path+'Kitti_GT')
        bbxs_header = np.genfromtxt(path+sub_dir+'_GT.csv', delimiter=',', dtype='|U')
        bbxs_org = np.genfromtxt(path+sub_dir+'_GT.csv', delimiter=',', skip_header=1, skip_footer=1)
        bbxs = bbxs_org[:,1:]
        print(bbxs.shape)
        header = bbxs_header[0,:]
        for i in range(len(bbxs[:,1])):
            bbxs_9 = bbxs[i].reshape(-1,9)
            bbxs_9_clean = bbxs_9[~np.isnan(bbxs_9).any(axis=1)]
            kitti_15 = np.zeros((len(bbxs_9_clean), 15),dtype=object)
            #extract x,y,z,dx,dy,dz,yaw
            bbxs_7 = bbxs_9_clean[:,[0,1,2,3,4,5,8]]
            for j in range(len(bbxs_9_clean)):
                indices = np.where((bbxs_9 == bbxs_9_clean[j]).all(axis=1))
                index = indices[0][0]
                object_name= header[index*9+1][:-5]
                object_name = object_name.replace(" ", "_")
                #label->velo
                bbxs_7[j,0:3] = bbxs_7[j,0:3] @ R_label_to_velo
                bbxs_7[j,-1:] = bbxs_7[j,-1:]/180*np.pi + np.pi/2
                alpha = np.arctan(bbxs_7[j,0]/bbxs_7[j,1])-np.pi/2 #lidar-alpha(-arctan(x/y))->camera-alpha(-alpha-pi/2)
                alpha = limit_period(alpha, period=np.pi * 2)
                #center z->bottom z
                bbxs_7[j,2] = bbxs_7[j,2]-bbxs_7[j,5]/2
                #raw: lidar->camera (kitti)

                # Coordinates in GT label:
                #             up z
                #                ^
                #                |
                #                |
                # left x <------ 0
                #               /
                #              /
                #             y back
                
                # Coordinates in LiDAR:
                #             up z
                #                ^   x front
                #                |  /
                #                | /
                # left y <------ 0

                # Coordinates in Camera:
                #         z front
                #        /
                #       /
                #      0 ------> x right
                #      |
                #      |
                #      v
                # down y
                # default: rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                gt_bboxes_3d = LiDARInstance3DBoxes(bbxs_7[j].reshape(-1,7)).convert_to(Box3DMode.CAM)
                # if you are trying to implement other rotation matrix between two coordinates
                # rt_mat = [] #tensor
                # gt_bboxes_3d = LiDARInstance3DBoxes(bbxs_7[j].reshape(-1,7)).convert_to(rt_mat, Box3DMode.CAM)
                gt_bboxes_3dnp = gt_bboxes_3d.numpy()
                #lwh(lidar)->lhw(camera)->hwl(kitti)
                kitti_7 = gt_bboxes_3dnp[0,[4,5,3,0,1,2,6]]
                kitti_15[j,0] = object_name
                kitti_15[j,3] = alpha
                kitti_15[j,-7:] = kitti_7
                
            with open(path+'Kitti_GT/'+str(i+1).zfill(6)+'.txt', 'w') as f:
                for text in kitti_15:
                    line = ' '.join(str(item) for item in text)
                    f.write(line+'\n')

if __name__ == "__main__":
    # input your folder location
    loc = '/home/gene/Documents/Validation Data2' 
    main(loc)