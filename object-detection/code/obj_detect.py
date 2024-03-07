'''
Detect and classify objects and label them in video captured through cv2
Class with utility functions.
Create object of class and use start() method of class to start capturing video and 
simultaneously detecting.
Other utility and control functions added.
'''

import os
import cv2
import torch
import keyboard as kb

class DetectorY5():
    def __init__(self):
        self.detector = 'ultralytics/yolov5'
        self.specifier = 'yolov5s'
        self.path = './media'
        self.model = torch.hub.load(self.detector, self.specifier)

    # Private utility functions
    # Add more using multithreading and call in start() function, as needed
    def to_stop(self, key:str = "ctrl") ->bool:
        '''
        TEMPORARY FUNCTION: IMPLEMENTATION TO BE CHANGED.
        obj is and argument to keyboard.is_pressed() function.
        Return true if we have to stop, as in, the key is pressed
        return not keyboard.is_pressed(\"ctrl\")
        '''
        return kb.is_pressed(key)

    # User visible functions

    #START FUNCTION
    def start(self, show_window: bool = True, save_video: bool = False, vid_file_name: str = "video"):
        '''
        NOTE: MORE LOGIC TO BE ADDED
        Start object detection using webcam.
        save_video decides whether to SAVE the frames as a collective video.
        show_window decides whether to SHOW the frames as a collective video.

        Function is able to detect and classify objects in almost all orientations and even when obstructed 
        by some other object in front, which is also usually classified.
        '''

        if (not os.path.exists("./media")):
            os.mkdir("media")

        capture = cv2.VideoCapture(0)
        if (not capture.isOpened()):
            print("Camera not working")
            print("Exiting ...")
        
        writer = None
        if (save_video):
            writer = cv2.VideoWriter("media/"+vid_file_name+".avi", 
                                     cv2.VideoWriter_fourcc(*'MJPG'), 10, 
                                     (int(capture.get(3)), int(capture.get(4))))  # (width, height)
        
        # else
        while (not self.to_stop()):
            ret, image = capture.read()
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(self.path+"/frame.jpg", image)

            results = self.model(self.path+"/frame.jpg")
            frame = results.render()[0]

            if (show_window):
                cv2.imshow("Result Window", frame)
                cv2.waitKey(1)
            
            if (save_video):
                writer.write(frame)
        
        if (show_window):
            cv2.destroyAllWindows()
        
        try: os.remove(self.path+"/frame.jpg")
        except: print("Error locating last captured frame, might not be removed.")
        
        capture.release()
        if (save_video):
            writer.release()


if __name__ == "__main__":
    detector = DetectorY5()
    detector.start()
