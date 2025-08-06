Functions:

#### 1. isc2kitti
data_processing_for_training/isc2kitti.py
Edit the coordinate tranformation if you want to get 3d labels in image and train the camera-based detector.
If you do not run the code in mmdetection3d, you need to add the code of coordinate transformation from lidar to camera and edit the 000000.txt calib file.
  - box_3d_mode.py
    - This is for you to understand the tranformation logic, not for running.

#### 2. remove data to isc_full
data_processing_for_training/isc_rename_bin_moveto_isc_dataset.py

Please provide the ISC loc and MMdet3d target location.
If you are planning to generate training data, set test=0; testing data, set test=1.

#### 3. generate imagesets (split train and val randomly)
data_processing_for_training/isc_imagesets_generation.py

#### 4. rename calib files according to imagesets
data_processing_for_training/isc_rename_calib.py

#### 5. generate kitti pickles (under mmdetection3d)
python tools/create_data.py kitti --root-path ./data/isc_full --out-dir ./data/isc_full --extra-tag isc_full

#### 6. train (under mmdetection3d)
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class.py --work-dir work_dirs/xxx

#### 7. test (under mmdetection3d)
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class.py work_dirs/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class/latest.pth --work-dir work_dirs/xxx

#### 8. continue training from last pth (under mmdetection3d)
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class.py --work-dir work_dirs/lidar12 --resume work_dirs/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class/latest.pth
