import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

PREPROCESSING_S_CHANNEL_THRESHOLD       = 230
PREPROCESSING_SOBEL_MAGNITUDE_THRESHOLD = (40, 255)
PREPROCESSING_SOBEL_DIRECTION_THRESHOLD = (np.deg2rad(45), np.deg2rad(90))
PREPROCESSING_SOBEL_KERNEL_SIZE         = 3

CAMERA_CALIBRATION_MATRIX_FILE = '../camera_cal/cal_matrix.p'
camera_mtx = None
camera_dist = None

M_perspective_transform = None


def undistort(image):
    global camera_mtx, camera_dist
    
    # if camera calibration parameters are not yet loaded, load them
    if (camera_mtx is None) or (camera_dist is None):
        camera_calib = pickle.load(open(CAMERA_CALIBRATION_MATRIX_FILE, "rb"))
        camera_mtx = camera_calib["mtx"]
        camera_dist = camera_calib["dist"]
    assert ((camera_mtx is not None) and (camera_dist is not None)), "Camera Calibration Parameters not available"
    
    # Apply Camera Lens Distortion Correction
    undistorted_image = cv2.undistort(image, camera_mtx, camera_dist)
    return undistorted_image


def transform_perspective(image):
    global M_perspective_transform
    if M_perspective_transform is None:
        src = np.float32([[580, 460],
                          [205, 720],
                          [1103, 720],
                          [704, 460]])
        dst = np.float32([[image.shape[1] / 4 , 0],   
                          [image.shape[1] / 4,  image.shape[0]],    
                          [image.shape[1] * 3 / 4, image.shape[0]],   
                          [image.shape[1] * 3 / 4, 0]])
        # Get new perspective transformation Matrix
        M_perspective_transform = cv2.getPerspectiveTransform(src, dst)
    assert M_perspective_transform is not None
   
    # Use M_perspective_transform to warp the input image to top-down view
    warped_image = cv2.warpPerspective(image, M_perspective_transform, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return warped_image


# Define a function that applies Sobel x and y, 
# then computes the magnitude direction of the gradient
# and applies a threshold.
def sobel_mag_dir_threshold(grayscale_image, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel_xy = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_abs_sobel_xy = np.uint8((abs_sobel_xy/np.max(abs_sobel_xy))*255)
    # 5) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 6) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    masked_sobel_dir = np.zeros_like(sobel_dir)
    masked_sobel_dir[
        (sobel_dir >= dir_thresh[0]) & 
        (sobel_dir <= dir_thresh[1]) &
        (scaled_abs_sobel_xy >= mag_thresh[0]) & 
        (scaled_abs_sobel_xy <= mag_thresh[1])
        ] = 1
    # 6) Return this mask as your binary_output image
    binary_output = masked_sobel_dir
    return binary_output


def preprocess_input_image(image):
    # 1. Apply perspective transformation to remove the effect of Camera Lens Distorion
    undistorted_image = undistort(image)
    
    # 2. Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # 3. Perform histogram equalization on S Channel
    s_channel_hist_equalized = cv2.equalizeHist(s_channel)
    
    # 4. Apply thresholding on S Channel (Normalized)
    s_channel_binary = np.zeros_like(s_channel_hist_equalized)
    s_channel_binary[(s_channel_hist_equalized>=PREPROCESSING_S_CHANNEL_THRESHOLD)] = 1
    
    # 5. Apply magnitude and direction thresholds on S Channel (Normalized)
    mag_dir_binary = sobel_mag_dir_threshold(s_channel_hist_equalized, sobel_kernel=PREPROCESSING_SOBEL_KERNEL_SIZE, mag_thresh=PREPROCESSING_SOBEL_MAGNITUDE_THRESHOLD, dir_thresh=PREPROCESSING_SOBEL_DIRECTION_THRESHOLD)
    
    # 6. Combine S Channel and Sobel binary images
    binary_thresh_image = np.zeros_like(s_channel_binary)
    binary_thresh_image[(mag_dir_binary == 1) | (s_channel_binary == 1)] = 1
    
    # 7. Perform perspective transformation to a 'Bird's Eye' view required by Lane Detection pipeline
    preprocessed_image  = transform_perspective(binary_thresh_image)
    
#     # Plot pipeline stages
#     plt.figure(1)
#     plt.subplot(231).set_title("1. Undistorted Original Image")
#     plt.imshow(undistorted_image)
#     plt.subplot(232).set_title("2. S Channel (Histogram Equalized)")
#     plt.imshow(s_channel_hist_equalized)
#     plt.subplot(233).set_title("3. S Channel Threshold")
#     plt.imshow(s_channel_binary, cmap='gray')
#     plt.subplot(234).set_title("4. Sobel with Magnitude and Direction Threshold")
#     plt.imshow(mag_dir_binary, cmap='gray')
#     plt.subplot(235).set_title("5. Combined S Channel and Sobel Threhsolds")
#     plt.imshow(binary_thresh_image, cmap='gray')
#     plt.subplot(236).set_title("6. Perspective transformation")
#     plt.imshow(preprocessed_image, cmap='gray')

#     # Plot Perspective Transform
#     plt.figure(1)
#     plt.subplot(211).set_title("Undistorted Original Image")
#     plt.imshow(undistorted_image)
#     plt.subplot(212).set_title("Warped Undistorted Original Image")
#     plt.imshow(transform_perspective(undistorted_image))

    
    return preprocessed_image, undistorted_image


def unwarp_image(image):
    assert M_perspective_transform is not None
    unwarped_image = cv2.warpPerspective(image, M_perspective_transform, image.shape[1::-1], flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    return unwarped_image


def draw_lane(ref_image, left_lane_poly, right_lane_poly, lane_color=(0,255, 0)):
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, ref_image.shape[0]-1, ref_image.shape[0] )
    left_lane_line = left_lane_poly.evaluate(ploty)
    right_lane_line = right_lane_poly.evaluate(ploty)
    
    # Create an ref_image to draw the lines on
    warp_zero = np.zeros_like(ref_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane_line, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_line, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), lane_color)
    
    
    output_image = color_warp
    return output_image


def overlay_lane(original_image, lane_polygon_image):
    overlayed_image = cv2.addWeighted(original_image, 1, lane_polygon_image, 0.3, 0)
    return overlayed_image

def process_frame(input_image, frame_number, lane_processor):
    
    # Run the Image Preprocessing Pipeline
    preprocessed_image, undistorted_image = preprocess_input_image(input_image)
    
    # Run the Lane Detection Pipeline
    smoothed_left_lane_poly, smoothed_right_lane_poly, lane_curvature, distance_to_center = lane_processor.process_frame(preprocessed_image)
    
    if (smoothed_left_lane_poly is not None) and (smoothed_right_lane_poly is not None) and (lane_curvature is not None):
        
        # draw lanes on the warped image
        lane_polygon_image = draw_lane(preprocessed_image, smoothed_left_lane_poly, smoothed_right_lane_poly)
        
        # Unwarped the image back to the original perspective
        lane_polygon_image_unwarped = unwarp_image(lane_polygon_image)
        
        # Overlay the lane image and information on the original frame
        output_image = overlay_lane(undistorted_image, lane_polygon_image_unwarped)
        cv2.putText(output_image,'Frame Number: {}'.format(frame_number), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(output_image,'Lane Curvature: {:.02f}m'.format(lane_curvature), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(output_image,'Distance to Lane Center: {:.02f}m'.format(distance_to_center), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    else:
        output_image = undistorted_image
        cv2.putText(output_image,'Frame Number: {}'.format(frame_number), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(output_image,'Lane Curvature: UNKNOWN', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(output_image,'Distance to Lane Center: UNKNOWN', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    
    return output_image

