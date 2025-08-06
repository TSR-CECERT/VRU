import random
import os

def split_text_file(input_file, output_file1, output_file2, split_ratio=0.8):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Calculate the split index
    split_index = int(len(lines) * split_ratio)

    # Shuffle the list randomly
    random.shuffle(lines)

    # Split into two parts
    part1 = lines[:split_index]
    part2 = lines[split_index:]

    # Write to the first output file
    with open(output_file1, 'w') as f1:
        f1.writelines(part1)

    # Write to the second output file
    with open(output_file2, 'w') as f2:
        f2.writelines(part2)


def list_files_to_txt(folder_path, output_file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                f.write(filename[:-4] + '\n')

def main(target, split_ratio):
    # Folder paths
    folder_path_train = target+ 'training/velodyne/'
    folder_path_test = target + 'testing/velodyne/'
    # Output file paths
    output_file_train = target + 'ImageSets/trainval.txt'
    output_file_test = target + 'ImageSets/test.txt'
    if not os.path.exists(target + 'ImageSets/'):
            os.makedirs(target + 'ImageSets/')

    # Call the function to write filenames from the folder to text files
    list_files_to_txt(folder_path_train, output_file_train)
    list_files_to_txt(folder_path_test, output_file_test)

    print(f"File list from folder {folder_path_test} has been written to {output_file_test}")

    # Input file and output file paths
    input_file = target + 'ImageSets/trainval.txt'
    output_file1 = target + 'ImageSets/train.txt'
    output_file2 = target + 'ImageSets/val.txt'
    # Call the function to randomly split the content of the input file into two output files
    split_text_file(input_file, output_file1, output_file2, split_ratio)

    print(f"Content of file {input_file} has been randomly split and saved to {output_file1} and {output_file2}")

if __name__ == "__main__":
    target = '/home/gene/mmdetection3d/data/isc_full/'
    main(target, split_ratio=0.8)