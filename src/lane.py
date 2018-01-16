import numpy as np
import matplotlib.pyplot as plt
import cv2

# Assumptions about real length and width of lane markings
LANE_WIDTH_METRES = 3.7
LANE_MARKING_LENGTH_METRES = 3.0

# Measurments of lane marking length and width measured from warped image of a straight lane 
LANE_WIDTH_PIXELS = 610
LANE_MARKING_LENGTH_PIXELS = 110
    
# Define conversions in x and y from pixels space to meters
M_PER_PIXEL_X = LANE_WIDTH_METRES/LANE_WIDTH_PIXELS # meters per pixel in x dimension
M_PER_PIXEL_Y = LANE_MARKING_LENGTH_METRES/LANE_MARKING_LENGTH_PIXELS # meters per pixel in y dimension


FIT_LANES_POLYNOMIAL_DEGREE = 2
# Set the number of sliding windows
FIT_LANES_NUM_WINDOWS = 10
# Set the width of the windows +/- FIT_LANES_WINDOW_MARGIN
FIT_LANES_WINDOW_MARGIN = 100
# Set minimum number of pixels found to recenter window
FIT_LANES_WINDOW_RECENTER_THRESHOLD = 100
FIT_LANES_LOOKAHEAD_MARGIN = 100
FIT_LANES_BLINDSCAN_HISTOGRAM_NUM_WINDOWS = 5

LANE_SMOOTHING_NUM_SAMPLES = 10
FIT_LANES_FORCE_BLINDSCAN_PERIOD = 5
LANE_NEW_SAMPLE_WEIGHT_FACTOR = 0.2

LANE_MAX_COEFFICIENT_ERROR = 0.30
LANE_MAX_C_ERROR = 0.15


class LanePolynomial:
    
    def __init__(self, coefficients):
        self.__coefficients = coefficients
    
    def evaluate(self, x):
        result = np.zeros_like(x, dtype=np.float32)
        for power, coefficient in enumerate(reversed(self.__coefficients)):
            result += coefficient*(x**power)
        
        return result
    
    def get_coefficient(self, power):
        assert power < len(self.__coefficients)
        return self.__coefficients[len(self.__coefficients)-(power+1)]
        
    def get_coefficients(self):
        """ Returns Polynomial Coefficients as a list """
        return self.__coefficients
    
    
    def is_parallel_to(self, other_lane):
        for this_lane_coeff, other_lane_coeff in zip(self.__coefficients[:-1], other_lane.__coefficients[:-1]):
            if (abs(this_lane_coeff) > LANE_MAX_COEFFICIENT_ERROR) or (abs(other_lane_coeff) > LANE_MAX_COEFFICIENT_ERROR):
                if (abs(1 - (this_lane_coeff/other_lane_coeff)) > LANE_MAX_COEFFICIENT_ERROR):
                    return False
            
        if abs(1 - (abs(self.__coefficients[-1] - other_lane.__coefficients[-1])/LANE_WIDTH_PIXELS)) > LANE_MAX_C_ERROR:
            return False
            
        return True

class Lane:
    
    def __init__(self, smoothing_num_samples):
        self.__lane_poly_list = []
        self.__lane_list_length = smoothing_num_samples
    
    
    def add_lane_sample(self, lane_poly):
        if lane_poly is not None:
            self.__lane_poly_list.append(lane_poly)
            while (len(self.__lane_poly_list) > self.__lane_list_length):
                del self.__lane_poly_list[0]
    
    
    def get_latest_lane_sample(self):
        return  self.__lane_poly_list[-1]
    
    
    def get_smoothed_lane_poly(self):
        if len(self.__lane_poly_list) > 0:
            r = LANE_NEW_SAMPLE_WEIGHT_FACTOR / (1-LANE_NEW_SAMPLE_WEIGHT_FACTOR)
            result = np.copy(self.__lane_poly_list[0].get_coefficients())
            weights = [1.0]
            
            for lane_poly in self.__lane_poly_list[1:]:
                element_weight = sum(weights)*r
                result += lane_poly.get_coefficients() * element_weight
                weights.append(element_weight)
            result = result / sum(weights)
            return LanePolynomial(result)
        else:
            return None

class LaneProcessor:
    
    def __init__(self):
        self.__right_lane = Lane(LANE_SMOOTHING_NUM_SAMPLES)
        self.__left_lane = Lane(LANE_SMOOTHING_NUM_SAMPLES)
        self.__sample_count = 0
    
    
    def fit_lanes_blindscan(self, image):
        # Assuming you have created a warped binary image called "image"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[(image.shape[0]*(FIT_LANES_NUM_WINDOWS-FIT_LANES_BLINDSCAN_HISTOGRAM_NUM_WINDOWS))//FIT_LANES_NUM_WINDOWS:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

#         # Print Histogram (for debugging)
#         _, [ax1, ax2] = plt.subplots(2, 1, sharex=True, num=3)
#         ax1.imshow(image[(image.shape[0]*(FIT_LANES_NUM_WINDOWS-FIT_LANES_BLINDSCAN_HISTOGRAM_NUM_WINDOWS))//FIT_LANES_NUM_WINDOWS:,:], cmap="gray")
#         ax2.plot(histogram)
        
        # Set height of windows
        window_height = np.int(image.shape[0]//FIT_LANES_NUM_WINDOWS)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        left_windows = []
        right_windows = []
        
        # Step through the windows one by one
        for window in range(FIT_LANES_NUM_WINDOWS):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - FIT_LANES_WINDOW_MARGIN
            win_xleft_high = leftx_current + FIT_LANES_WINDOW_MARGIN
            win_xright_low = rightx_current - FIT_LANES_WINDOW_MARGIN
            win_xright_high = rightx_current + FIT_LANES_WINDOW_MARGIN
    
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > FIT_LANES_WINDOW_RECENTER_THRESHOLD pixels, recenter next window on their mean position
            if len(good_left_inds) > FIT_LANES_WINDOW_RECENTER_THRESHOLD:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > FIT_LANES_WINDOW_RECENTER_THRESHOLD:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
            left_windows.append([(leftx_current-FIT_LANES_WINDOW_MARGIN, win_y_high), (leftx_current-FIT_LANES_WINDOW_MARGIN, win_y_low), (leftx_current+FIT_LANES_WINDOW_MARGIN, win_y_low), (leftx_current+FIT_LANES_WINDOW_MARGIN, win_y_high)])
            right_windows.append([(rightx_current-FIT_LANES_WINDOW_MARGIN, win_y_high), (rightx_current-FIT_LANES_WINDOW_MARGIN, win_y_low), (rightx_current+FIT_LANES_WINDOW_MARGIN, win_y_low), (rightx_current+FIT_LANES_WINDOW_MARGIN, win_y_high)])
        
#         # Print left and right lane windows (for debugging
#         windows_image = cv2.cvtColor(image*255, cv2.COLOR_GRAY2RGB)
#         cv2.polylines(windows_image,np.array(left_windows+right_windows, dtype=np.int32),True,(255,0,0))
#         plt.figure(4)
#         plt.imshow(windows_image)
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        left_lane_poly = None
        right_lane_poly = None
        num_left_lane_points = len(lefty)
        num_right_lane_points = len(righty)
        
        if num_left_lane_points > 0:
            left_lane_poly = np.polyfit(lefty, leftx, FIT_LANES_POLYNOMIAL_DEGREE)
        if num_right_lane_points > 0:
            right_lane_poly = np.polyfit(righty, rightx, FIT_LANES_POLYNOMIAL_DEGREE)
        
        return LanePolynomial(left_lane_poly), LanePolynomial(right_lane_poly), num_left_lane_points, num_right_lane_points
    
    
    def fit_lanes_lookahead(self, image):
        prev_left_lane_poly = self.__left_lane.get_smoothed_lane_poly()
        prev_right_lane_poly = self.__right_lane.get_smoothed_lane_poly()
        
        # If no lane information previously exists, run a blind scan
        assert ((prev_left_lane_poly is not None) and (prev_right_lane_poly is not None))
        
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (prev_left_lane_poly.evaluate(nonzeroy) - FIT_LANES_LOOKAHEAD_MARGIN)) 
                          & (nonzerox < (prev_left_lane_poly.evaluate(nonzeroy) + FIT_LANES_LOOKAHEAD_MARGIN))) 
        
        right_lane_inds = ((nonzerox > (prev_right_lane_poly.evaluate(nonzeroy) - FIT_LANES_LOOKAHEAD_MARGIN)) 
                           & (nonzerox < (prev_right_lane_poly.evaluate(nonzeroy) + FIT_LANES_LOOKAHEAD_MARGIN)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_lane_poly = None
        right_lane_poly = None
        num_left_lane_points = len(lefty)
        num_right_lane_points = len(righty)
        
        if (num_left_lane_points > 0) and (num_right_lane_points > 0):
            left_lane_poly = LanePolynomial(np.polyfit(lefty, leftx, FIT_LANES_POLYNOMIAL_DEGREE))
            right_lane_poly = LanePolynomial(np.polyfit(righty, rightx, FIT_LANES_POLYNOMIAL_DEGREE))
        
        return left_lane_poly, right_lane_poly, num_left_lane_points, num_right_lane_points
    
    
    def calculate_curvature(self, ref_image, reference_point=None):
        left_lane_poly = self.__left_lane.get_smoothed_lane_poly()
        right_lane_poly = self.__right_lane.get_smoothed_lane_poly()
        
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, ref_image.shape[0]-1, ref_image.shape[0])
        
        # Fit a second order polynomial to pixel positions in each fake lane line
        left_lane_line = left_lane_poly.evaluate(ploty)
        right_lane_line = right_lane_poly.evaluate(ploty)
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*M_PER_PIXEL_Y, left_lane_line*M_PER_PIXEL_X, FIT_LANES_POLYNOMIAL_DEGREE)
        right_fit_cr = np.polyfit(ploty*M_PER_PIXEL_Y, right_lane_line*M_PER_PIXEL_X, FIT_LANES_POLYNOMIAL_DEGREE)
        # Calculate the new radii of curvature
        if reference_point is None:
            reference_point = ref_image.shape[0]
        left_lane_curvature_radius = ((1 + (2*left_fit_cr[0]*reference_point*M_PER_PIXEL_Y + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_lane_curvature_radius = ((1 + (2*right_fit_cr[0]*reference_point*M_PER_PIXEL_Y + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return (left_lane_curvature_radius + right_lane_curvature_radius)/2
    
    
    
    def calculate_distance_to_lane_center(self, ref_image):
        
        # Assume that the camera is mounted in the center of the car, so the car is at the center of the frame
        x_car = ref_image.shape[1]//2
        
        # Calculate distance to left lane
        x_left_lane = self.__left_lane.get_smoothed_lane_poly().evaluate(ref_image.shape[0])
        
        # Calculate distance to right lane
        x_right_lane = self.__right_lane.get_smoothed_lane_poly().evaluate(ref_image.shape[0])
        
        lane_center = (x_right_lane - x_left_lane)//2  + x_left_lane
        distance_to_lane_center_px = x_car - lane_center
        return distance_to_lane_center_px*M_PER_PIXEL_X


    def perform_sanity_checks(self, left_lane_poly, right_lane_poly, num_left_lane_points, num_right_lane_points):
        
        if (left_lane_poly is None) or (left_lane_poly is None):
            return False
        
        if self.__sample_count == 0:
            return True
        
        if not left_lane_poly.is_parallel_to(right_lane_poly):
            return False
            
        return True
    
    
    def process_frame(self, image):
        
        if (self.__sample_count % FIT_LANES_FORCE_BLINDSCAN_PERIOD) == 0:
            left_lane_poly, right_lane_poly, num_left_lane_points, num_right_lane_points = self.fit_lanes_blindscan(image)            
        else:
            left_lane_poly, right_lane_poly, num_left_lane_points, num_right_lane_points = self.fit_lanes_lookahead(image)
                
        if self.perform_sanity_checks(left_lane_poly, right_lane_poly, num_left_lane_points, num_right_lane_points) == True:
            self.__left_lane.add_lane_sample(left_lane_poly)
            self.__right_lane.add_lane_sample(right_lane_poly)
        
        smoothed_left_lane_poly = self.__left_lane.get_smoothed_lane_poly()
        smoothed_right_lane_poly = self.__right_lane.get_smoothed_lane_poly()
        
        lane_curvature = self.calculate_curvature(image)
        distance_to_lane_center = self.calculate_distance_to_lane_center(image)
        
        self.__sample_count += 1
        
        return smoothed_left_lane_poly, smoothed_right_lane_poly, lane_curvature, distance_to_lane_center
        