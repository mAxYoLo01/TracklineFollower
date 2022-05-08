from utils import concat_tile_resize
import numpy as np
import cv2
import time
import os

TESTING_ENV = not (os.uname()[1] == "raspberrypi")
if TESTING_ENV:
    print("Running on testing environment!")


def distance_two_points(x1, y1, x2, y2):
    return np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

def analyze_frame(frame):
    initial_time = time.time()
    
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
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    raw_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    # Analyzing lines to exclude unwanted ones
    y_min = frame.shape[0]
    y_max = int(frame.shape[0] * 0.2) 
    lines = []
    line_image = np.copy(frame) * 0  # creating a blank image to draw lines on
    if raw_lines is not None:
        for line in raw_lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                
                if abs(slope) < 0.1: # Exclude horizontal lines
                    continue
                
                # Exclude line if it is too similar to another line already present (to exclude duplicates)
                for slope2, intercept2 in lines:
                    # Separate line points into low and higher points
                    low_point1 = ((y_min - intercept) / slope, y_min)
                    low_point2 = ((y_min - intercept2) / slope2, y_min)
                    high_point1 = ((y_max - intercept) / slope, y_max)
                    high_point2 = ((y_max - intercept2) / slope2, y_max)
                    
                    distance_tolerance = 100
                    if distance_two_points(*low_point1, *low_point2) < distance_tolerance and distance_two_points(*high_point1, *high_point2) < distance_tolerance:
                        break
                else:
                    lines.append((slope, intercept))      

    # Drawing lines
    print("--- Current frame ---")
    if not lines:
        print("No lines!")
    for slope, intercept in lines:
        x1 = int((y_min - intercept) / slope)
        x2 = int((y_max - intercept) / slope)
        cv2.line(line_image, (x1, y_min), (x2, y_max), (0, 0, 255), 10)
        print(f"Slope: {slope} - intercept: {intercept}")

    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    # print(f"Algorithm finished in {time.time() - initial_time} seconds.")
    if TESTING_ENV:
        result = concat_tile_resize([
            [frame, masked_image],
            [cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), lines_edges]
        ])
        cv2.imshow("Result", result)


def main():
    cap = cv2.VideoCapture("videos/2.mp4")
    # cap = cv2.VideoCapture(2)
    # Default image size is 1280x720, converting to 800x600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not cap.isOpened():
        print("Error: Could not open Camera!")
        return

    initial_time = time.time()
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

    cap.release()
    print(f"Algorithm finished in {time.time() - initial_time} seconds.")

if __name__ == "__main__":
    main()