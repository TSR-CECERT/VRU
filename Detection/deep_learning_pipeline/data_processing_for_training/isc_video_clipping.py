import os
import cv2

# Define function to save image
# image: the image to be saved
# pic_address: address to save the image
# num: image suffix number for differentiation, int type
def save_image(image, address, num):
    pic_address = address + str(num).zfill(6) + '.png'
    cv2.imwrite(pic_address, image)

def video_to_pic(video_path, save_path, frame_rate):
    # Read the video file
    videoCapture = cv2.VideoCapture(video_path)
    j = 0
    i = 0
    # Read frames
    success, frame = videoCapture.read()
    while success:
        i = i + 1
        # Save one image every fixed number of frames
        if i % frame_rate == 0:
            j = j + 1
            save_image(frame, save_path, j)
            print('Image saved at address:', save_path + str(j) + '.png')
        success, frame = videoCapture.read()

if __name__ == '__main__':
    # Video file and image save address
    SAMPLE_VIDEO = '/home/gene/Documents/Validation Data/Run_214/VisualCamera5_Run_214.mp4'
    SAVE_PATH = '/home/gene/Documents/Validation Data/Run_214/images/'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Set fixed frame rate
    FRAME_RATE = 3
    video_to_pic(SAMPLE_VIDEO, SAVE_PATH, FRAME_RATE)
