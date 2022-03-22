import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open Camera!")
    exit()

t = time.time()
for i in range(10):
    ret, frame = cap.read()
    
    while not ret:
       print("Can't receive frame. Retrying ...")
       cap.release()
       cap = cv2.VideoCapture(0)                                                                              
       ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(f"images/{i}.jpg", frame)
    else:
        print(f"Frame #{i} failed!")
    
    # time.sleep(10)

cap.release()
print(f"Algorithm finished in {time.time() - t} seconds.")