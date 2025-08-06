#for loop processing
import os

def main(loc):
    
    # Get a list of all items in the directory
    sub_srcs = os.listdir(loc)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(loc, sub_src))]
    loc = loc.replace(" ", "\ ")
    for sub_dir in sub_dirs:
        print(sub_dir)
        Input1 = loc+sub_dir+'/Lidar1_' + sub_dir + '.pcap'
        Output1 = loc+sub_dir+'/Lidar1_pcd'
        os.system('cargo run --release -- convert     -i '+Input1+' -o ' + Output1 + ' -f pcap.velodyne      -t pcd.libpcl     --velodyne-model VLP32C     --velodyne-return-mode strongest')
        Input2 = loc+sub_dir+'/Lidar2_' + sub_dir + '.pcap'
        Output2 = loc+sub_dir+'/Lidar2_pcd'
        os.system('cargo run --release -- convert     -i '+Input2+' -o ' + Output2 + ' -f pcap.velodyne      -t pcd.libpcl     --velodyne-model VLP32C     --velodyne-return-mode strongest')

if __name__ == "__main__":
    loc = '/home/gene/Documents/Validation Data2/'
    main(loc)