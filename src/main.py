import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import os
import cv2

from pipeline import process_frame
from lane import LaneProcessor

def process_single_image(path):
    input_image = mpimg.imread(path)
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)  
    lane_processor = LaneProcessor()
    output_image = process_frame(input_image, 1, lane_processor)
    
    plt.figure(10)
    plt.imshow(output_image)
    plt.show()

def process_video_start():
    global frame_counter
    frame_counter = 0

def process_video_frame(image):
    global frame_counter, test_video_output_images_path, lane_processor
    frame_counter += 1
#     mpimg.imsave(test_video_output_images_path + "/input/" + str(frame_counter) + ".jpg", image)
    
    output_image = process_frame(image, frame_counter, lane_processor)
    
#     mpimg.imsave(test_video_output_images_path + "/output/" + str(frame_counter) + ".jpg", output_image)
#     mpimg.imsave(test_video_output_images_path + "/output/top view_" + str(frame_counter) + ".jpg", image_undistorted_perspective_transformed)
    
    return output_image


def process_videos(test_videos):

    global test_video_output_images_path, lane_processor
    
    OUTPUT_IMAGES_PATH = "../output_images"
    OUTPUT_VIDEO_PATH = "../output_videos"
    
    if not os.path.exists(OUTPUT_VIDEO_PATH):
            os.makedirs(OUTPUT_VIDEO_PATH)
    
    for test_video_path in test_videos:
        test_video_output_path = OUTPUT_VIDEO_PATH + '/' + os.path.basename(test_video_path)
        
        test_video_output_images_path = OUTPUT_IMAGES_PATH + '/' + os.path.basename(test_video_path).split(".")[0]
        if not os.path.exists(test_video_output_images_path+"/input"):
            os.makedirs(test_video_output_images_path+"/input")
        if not os.path.exists(test_video_output_images_path+"/output"):
            os.makedirs(test_video_output_images_path+"/output")
            
        lane_processor = LaneProcessor()
            
        clip1 = VideoFileClip(test_video_path)
        process_video_start()
        white_clip = clip1.fl_image(process_video_frame) 
        white_clip.write_videofile(test_video_output_path, audio=False)



# process_single_image('../test_images/straight_lines2.jpg')
# process_single_image('../output_images/project_video/input/643.jpg')
# process_videos(["../project_video.mp4"])
process_videos(["../project_video.mp4", "../challenge_video.mp4", "../harder_challenge_video.mp4"])
