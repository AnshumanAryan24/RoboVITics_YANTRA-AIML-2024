# Importing Libraries
import cv2
import mediapipe as mp
import time

# This line creates a video capture object (video) that connects to your webcam (index 0 usually refers to the default camera).
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Unable to open video source.")
    exit()

# This line creates a hand tracking object (mpHands) from the MediaPipe library. Also setting max hands as 1.
mpHands = mp.solutions.hands
# This creates an instance of the hand tracking model
hands = mpHands.Hands(max_num_hands=1)

# This creates a drawing utility object (mpDraw) used to draw the hand landmarks on the video frame.
mpDraw = mp.solutions.drawing_utils

# This list stores the landmark IDs (indices) corresponding to the fingertips of each finger (thumb to little finger).
tipIds = [4, 8, 12, 16, 20]
# This list stores the names of each finger corresponding to their tip IDs.
fingerName = ['thumb', 'index finger', 'middle finger', 'ring finger', 'little finger']


prevTime = 0

# This loop continuously captures frames from the webcam and processes them.
while True:
    # This line reads a frame from the webcam. success is a boolean indicating if the frame was read successfully. img stores the captured frame as an image.
    success, img = video.read()
    img = cv2.flip(img, 1)

    # This line converts the image from BGR (OpenCV's default color format) to RGB format, which is expected by the MediaPipe hand tracking model. This step might be unnecessary depending on the MediaPipe version.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []
    # his checks if the results from the hand tracking model (mpHands.process(imgRGB)) contain any information about detected hands (multi_hand_landmarks). If hands are detected (not None), the inner loop executes.
    if results.multi_hand_landmarks:
        # This loop iterates through each detected hand (handLms) in the results. Each handLms object contains information about the landmarks for that specific hand
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # This line gets the height (h), width (w), and number of channels (c) of the image (img). This information is needed to convert the normalized landmark coordinates (lm.x and lm.y) from the hand tracking model (between 0.0 and 1.0) to actual pixel coordinates within the image.
                h, w, c = img.shape
                # This calculates the pixel coordinates (cx, cy) for the current landmark. It multiplies the normalized x-coordinate (lm.x) by the image width (w) and the normalized y-coordinate (lm.y) by the image height (h), and then converts the results to integers using int().
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # This if statement will help in pointing out a specific index which is here, the thumb
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            # This line will draw the landmarks and their connection between them. 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    # print(lmList)
    # If the hand is detected , this list will contain the pixel coordinates of each landmark with their IDs.
    if len(lmList):
        fingers = []

        # thumb
        if ((lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] and lmList[tipIds[0]][2] <= lmList[tipIds[0] - 1][2]) or (lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] and lmList[tipIds[0]][2] >= lmList[tipIds[0] - 1][2])):
            fingers.append(1)
        else:
            fingers.append(0)

        # fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = sum(fingers)

        if totalFingers == 1:
            for fin in range(5):
                if fingers[fin] == 1:
                    upFinger = fingerName[fin]
                    cv2.putText(img, str(upFinger), (275, 445), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        elif totalFingers == 5:
            cv2.putText(img, 'palm', (275, 445), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        elif totalFingers == 0:
            cv2.putText(img, 'fist', (275, 445), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


        totalFingers = sum(fingers)
        print(totalFingers)

        cv2.putText(img, str(totalFingers), (575, 45), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    # This displays the processed image (with hand landmarks, finger count, and FPS) in a window titled "Image".
    cv2.imshow("Image", img)
    # This line waits for a key press for 1 millisecond. If the pressed key is 'q', the bitwise AND operation with 0xFF ensures only the lower 8 bits (ASCII code) are considered. A match with 'q' (ASCII code 113) breaks the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
