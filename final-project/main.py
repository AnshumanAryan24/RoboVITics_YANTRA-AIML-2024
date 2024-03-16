import cv2
import numpy as np
import mediapipe as mp  #Handles hand landmark detection.
import time #: Tracks time for gesture detection delay.
import supervision
import ultralytics# YOLO object detection library.
from collections import defaultdict #Used for the defaultdict data structure.
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation #Speed estimation module from ultralytics.

import serial 
cnt=0
def speed_estimation():
    track_history = defaultdict(lambda: [])
    crossed_track_ids = set()
    crossed_objects_count = 0  
    final_output = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")
        for box, track_id, cls in zip(results[0].boxes.xywh.cpu(), results[0].boxes.id.int().cpu().tolist(), results[0].boxes.cls.int().cpu().tolist()):
            x, y, w, h = box
            track = track_history[track_id] #This is likely a dictionary that stores historical position data for each object being tracked.
            track.append((float(x), float(y))) #This variable holds a unique identifier assigned to the current object during detection.
            if len(track) > 30: #he append method adds a new element to the end of the list track
                track.pop(0)
            if cls in [2, 3, 5, 7] and track_id not in crossed_track_ids:# both conditions needs to be true
                crossed_objects_count += 1
                                    # Add track_id to the set
                crossed_track_ids.add(track_id)
            speed=calculate_speed(track_id,track)
            object=(speed>2).any()
            if object:
                print(1)
            else:
                print(0)
def count_vehicles():
    track_history = defaultdict(lambda: [])
    crossed_track_ids = set()
    crossed_objects_count = 0  
    final_output = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")
        for box, track_id, cls in zip(results[0].boxes.xywh.cpu(), results[0].boxes.id.int().cpu().tolist(), results[0].boxes.cls.int().cpu().tolist()):
            x, y, w, h = box
            track = track_history[track_id] #This is likely a dictionary that stores historical position data for each object being tracked.
            track.append((float(x), float(y))) #This variable holds a unique identifier assigned to the current object during detection.
            if len(track) > 30: #he append method adds a new element to the end of the list track
                track.pop(0)
            if cls in [2, 3, 5, 7] and track_id not in crossed_track_ids:# both conditions needs to be true
                crossed_objects_count += 1
                                    # Add track_id to the set
                crossed_track_ids.add(track_id)
                    
                arduino.write(str.encode(str(crossed_objects_count)))
def calculate_speed(self, trk_id, track):
    """
    Calculation of object speed.

    Args:
        trk_id (int): object track id.
        track (list): tracking history for tracks path drawing
    """

    if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
        return
    if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
        direction = "known"

    elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
        direction = "known"

    else:
        direction = "unknown"

    if self.trk_previous_times[trk_id] != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
        self.trk_idslist.append(trk_id)

        time_difference = time() - self.trk_previous_times[trk_id]
        if time_difference > 0:
            dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
            speed = dist_difference / time_difference
            self.dist_data[trk_id] = speed

    self.trk_previous_times[trk_id] = time()
    self.trk_previous_points[trk_id] = track[-1]

def count_fingers(lst):

    #It calculates a threshold based on the vertical distance between the thumb and pinky landmarks.
    
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    # Check only for index finger (finger 1)
    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1

    # Check only for middle finger (finger 2)
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1
#It checks if the index finger and middle finger are bent (based on landmark positions) and increments a counter (cnt) for each bent finger.

    return cnt #returns the finger count


CamCap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils #Hand landmark detection objects (hands and hand_obj) are created.
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)# only 
#start_init and prev variables are used for gesture detection timing.
start_init = False
prev = -1
model = YOLO('yolov8x.pt')

                        # Open the video capture object
cap = cv2.VideoCapture(r"sample.mp4")
arduino = serial.Serial(port='COM7', baudrate=115200, timeout=.1) #Serial communication with the Arduino is established (arduino).

try:
    while True:
       
        end_time = time.time() #The current time (end_time) is recorded.
        _, frm = CamCap.read()
        if frm is None:
            print("Unable to read from camera") #Error handling
            continue
        frm = cv2.flip(frm, 1) 

        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) 

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]

            if prev != cnt:
                if not start_init:#first detection
                    start_time = time.time()# 
                    start_init = True
                elif (end_time - start_time) > 0.2:
                    
                    if cnt == 1:
                        count_vehicles()
                        
                      
                        # Release resources
                        cap.release()

                        
                    elif cnt == 2:
                        speed_estimation()
                           

                        # Release resources
                        cap.release()
                    

        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27:
            break

finally:
    arduino.close()
    CamCap.release()
    cap.release()
    cv2.destroyAllWindows()
