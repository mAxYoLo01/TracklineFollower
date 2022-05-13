try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None

from utils import concat_tile_resize, project_point_on_line
import numpy as np
import cv2
import time
import os

text = None
if GPIO:
    left_vibrator_pin = 32
    right_vibrator_pin = 36
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(left_vibrator_pin, GPIO.OUT)  
    GPIO.setup(right_vibrator_pin, GPIO.OUT)  


TESTING_ENV = not (os.uname()[1] == "raspberrypi")
if TESTING_ENV:
    print("Running on testing environment!")


def distance_two_points(x1, y1, x2, y2):
    return np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

def slope_intercept_to_x_y(slope: float, intercept: float, y):
    return (int((y - intercept) / slope), int(y))

def analyze_frame(frame):
    # Add a mask to keep only white colors
    sensitivity = 30
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convert to gray image (remove color channels from the image)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply blur to the image (filters out image noise)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Detect edges using the Canny algorithm
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Line detection using the Hough transform algorithm
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 70  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    raw_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Analyzing lines to exclude unwanted ones
    y_min = frame.shape[0]
    y_max = int(y_min * 0.2)
    lines = []
    line_image = np.copy(frame) * 0  # creating a blank image to draw lines on
    if raw_lines is not None:
        for line in raw_lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                
                if abs(slope) < 0.2: # Exclude horizontal lines
                    continue
                
                # Exclude line if it is too similar to another line already present (to exclude duplicates)
                for slope2, intercept2 in lines:
                    # Separate line points into low and higher points
                    low_point1 = slope_intercept_to_x_y(slope, intercept, y_min)
                    low_point2 = slope_intercept_to_x_y(slope2, intercept2, y_min)
                    high_point1 = slope_intercept_to_x_y(slope, intercept, y_max)
                    high_point2 = slope_intercept_to_x_y(slope2, intercept2, y_max)
                    
                    distance_tolerance = 100
                    if distance_two_points(*low_point1, *low_point2) < distance_tolerance and distance_two_points(*high_point1, *high_point2) < distance_tolerance:
                        break
                else:
                    lines.append((slope, intercept))      

    # Drawing lines
    print("--- Current frame ---")
    avg_x_max, avg_x_min = [], []
    if not lines:
        print("No lines!")
    for slope, intercept in lines:
        x1, _ = slope_intercept_to_x_y(slope, intercept, y_min)
        x2, _ = slope_intercept_to_x_y(slope, intercept, y_max)
        cv2.line(line_image, (x1, y_min), (x2, y_max), (0, 0, 255), 10)
        print(f"Slope: {slope} - intercept: {intercept}")
        if len(lines) == 2:
            avg_x_min.append(x1)
            avg_x_max.append(x2)
        
    # Drawing middle line
    if len(lines) == 2:
        middle_x_max, middle_x_min = int(np.average(avg_x_max)), int(np.average(avg_x_min))
        cv2.line(line_image, (middle_x_min, y_min), (middle_x_max, y_max), (255, 0, 0), 10)
        
    # Send a vibration depending on the lines' position
    global text
    if len(lines) == 2:
        center_image = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))
        if middle_x_max != middle_x_min:
            # Find the distance between the center of the image and the center of the average line
            avg_slope = (y_max - y_min) / (middle_x_max - middle_x_min)
            avg_intercept = y_max - (avg_slope * middle_x_max)
            x_mid, y_mid = project_point_on_line(avg_slope, avg_intercept, center_image)
            distance = distance_two_points(*center_image, x_mid, y_mid)
            print("Distance:", distance)
            
            # Find the distance between one lane and the middle line
            x1_mid, _ = slope_intercept_to_x_y(*lines[0], center_image[1])
            x2_mid, _ = slope_intercept_to_x_y(*lines[1], center_image[1])
            distance_lane_to_middle = distance_two_points(x1_mid, center_image[1], x2_mid, center_image[1]) / 2
            # If the distance is too large, then it means it needs to vibrate
            
            text = None
            if distance > 0.6 * distance_lane_to_middle:
                color = (0, 0, 155) # Dark color
                if center_image[0] > x_mid:
                    text = "TURN LEFT"
                else:
                    text = "TURN RIGHT"
            else:
                color = (0, 255, 0)
            cv2.line(line_image, (int(x_mid), int(y_mid)), center_image, color, 10)
    if text:
        cv2.putText(line_image, text, (200, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), thickness=5)
        
        if GPIO:
            if text == "TURN LEFT":
                GPIO.output(left_vibrator_pin, GPIO.HIGH)
            else:
                GPIO.output(right_vibrator_pin, GPIO.HIGH)
    elif GPIO:
        GPIO.output(left_vibrator_pin, GPIO.LOW)
        GPIO.output(right_vibrator_pin, GPIO.LOW)

    
    if TESTING_ENV:
        lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        result = concat_tile_resize([
            [frame, masked_image],
            [cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), lines_edges]
        ])
        cv2.imshow("Result", result)


def main():
    cap = cv2.VideoCapture("videos/2.mp4", cv2.CAP_FFMPEG)
    # cap = cv2.VideoCapture(2)
    # Default image size is 1280x720, converting to 800x600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not cap.isOpened():
        print("Error: Could not open Camera!")
        return

    i = 0
    initial_time = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                # print(f"Frame #{i} failed!")
                print("Frame reading failed, shutting down...")
                break
            else:
                analyze_frame(frame)
                key_pressed = cv2.waitKey(1)
                if key_pressed & 0xFF == ord('q'):
                    break
                if key_pressed & 0xFF == ord('p'):
                    cv2.waitKey() # Pause the program until another key is pressed
                # cv2.imwrite(f"images/{i}.jpg", frame)
            
            i += 1
    except KeyboardInterrupt:
        pass

    cap.release()
    if GPIO:
        GPIO.output(left_vibrator_pin, GPIO.LOW)
        GPIO.output(right_vibrator_pin, GPIO.LOW)
    print(f"Algorithm finished in {time.time() - initial_time} seconds.")

if __name__ == "__main__":
    main()