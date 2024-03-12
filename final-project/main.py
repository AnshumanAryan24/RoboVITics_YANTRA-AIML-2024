import cv2
import mediapipe as mp
import time

def count_fingers(lst):
  cnt = 0
  thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

  # Check only for index finger (finger 1)
  if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
    cnt += 1

  # Check only for middle finger (finger 2)
  if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
    cnt += 1

  return cnt

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1

try:
  while True:
    end_time = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
      hand_keyPoints = res.multi_hand_landmarks[0]
      cnt = count_fingers(hand_keyPoints)

      if not prev == cnt:
        if not start_init:
          start_time = time.time()
          start_init = True
        elif (end_time - start_time) > 0.2:
          # Perform action based on finger count
          if cnt == 1:
              import cv2
              import supervision
              import ultralytics
              from collections import defaultdict

              from ultralytics import YOLO

              # Define the start and end points for the line (optional, remove if not used)
              # START = supervision.Point(716, 150)
              # END = supervision.Point(1732, 146)

              # Load the YOLO model (replace with your model path)
              model = YOLO('yolov8x.pt')

              # Open the video capture object
              cap = cv2.VideoCapture(r"C:\Users\yash2\OneDrive\Desktop\sample_video.mp4")

              # Dictionary to store track history
              track_history = defaultdict(lambda: [])

              # Set to store unique track IDs that have crossed
              crossed_track_ids = set()  # Use a set to efficiently check for unique IDs

              crossed_objects_count = 0  # Variable to track total count
              final_output = 0  # Variable for final output

              while cap.isOpened():
                  success, frame = cap.read()

                  if not success:
                      break

                  # Perform object detection and tracking (analyze entire video)
                  results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")

                  # Process detections to count vehicles
                  for box, track_id, cls in zip(results[0].boxes.xywh.cpu(), results[0].boxes.id.int().cpu().tolist(),
                                                results[0].boxes.cls.int().cpu().tolist()):
                      x, y, w, h = box

                      track = track_history[track_id]
                      track.append((float(x), float(y)))
                      if len(track) > 30:
                          track.pop(0)

                      # Count only relevant classes (cars and trucks) and unique track IDs
                      if cls in [2, 3, 5, 7] and track_id not in crossed_track_ids:
                          crossed_objects_count += 1
                          crossed_track_ids.add(track_id)  # Add track_id to the set

                  # Check and set final output
                  if crossed_objects_count > 7:
                      final_output = 1
                      break  # Optional: break if you only care about the first instance
                  else:
                      final_output = 0

              # Release resources
              cap.release()

              # Print the final output
              print(final_output)
          elif cnt == 2:
              from ultralytics import YOLO
              from ultralytics.solutions import speed_estimation
              import cv2

              model = YOLO("yolov8m.pt")  # Load YOLO model
              names = model.model.names

              # Capture video and set writer properties
              cap = cv2.VideoCapture(r"C:\Users\yash2\OneDrive\Desktop\sample_video.mp4")
              w, h, fps = (int(cap.get(x)) for x in
                           (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
              Video_Writer = cv2.VideoWriter("Speed_estimation.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

              # Define line points for speed estimation
              line_pts = [(0, h // 2), (w, h // 2)]

              # Create speed estimator object and set arguments
              speed_obj = speed_estimation.SpeedEstimator()
              speed_obj.set_args(reg_pts=line_pts, names=names, view_img=True)  # Adjust view_img as needed

              while cap.isOpened():
                  success, im0 = cap.read()
                  if not success:
                      print("Empty Video Frame")
                      break

                  # Track objects in the frame
                  tracks = model.track(im0, persist=True, show=False)

                  # Estimate speeds using the speed estimator
                  speed_estimates = speed_obj.estimate_speed(im0, tracks)  # Assuming returns array of speeds

                  # Check for speed above 5 km/hr in any detected object
                  has_fast_object = (speed_estimates > 2).any()

                  if has_fast_object:
                      print(1)  # Print 1 if any object has speed > 5 km/hr
                  else:
                      print(0)

                  # Write processed frame to output video
                  Video_Writer.write(im0)

              # Release resources
              cap.release()
              Video_Writer.release()
          prev = cnt
          start_init = False

      drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
      break

finally:
  cap.release()
  cv2.destroyAllWindows()
