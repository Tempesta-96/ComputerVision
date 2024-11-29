import cv2
import mediapipe as mp
import time
import math
import numpy as np

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def calculateFPS(lastFrameTime):
    # fps calculations
    thisFrameTime = time.time()
    fps = 1 / (thisFrameTime - lastFrameTime)
    lastFrameTime = thisFrameTime
    return fps, lastFrameTime

def identify_right_left_hand(result, idx:int, hand):
    '''using the results to identify if it is right or left hand. '''
    # Get handedness (left or right)
    handedness = result.multi_handedness[idx].classification[0].label
    if handedness == 'Right': handedness = 'Left-volume'
    elif handedness == 'Left': handedness = 'Right-camera'

    # Draw landmarks on the hand
    mpdraw.draw_landmarks(img, hand, mphands.HAND_CONNECTIONS)

    # Get the coordinates of the first landmark (wrist) for labeling
    wrist = hand.landmark[mphands.HandLandmark.WRIST]
    h, w, _ = img.shape
    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
    
    # Put the label (Right or Left) on the screen
    cv2.putText(img, handedness, (wrist_x, wrist_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return handedness, wrist_x, wrist_y

def zoom_image(img, scale, index_x, index_y):
    '''scale= 2 means zoom in by 2x'''
    # Get the dimensions of the image
    height, width = img.shape[:2]
    # center_x, center_y = width // 2, height // 2

    # Calculate zoomed region
    new_width, new_height = int(width / scale), int(height / scale)
    x1, y1 = index_x - new_width // 2, index_y - new_height // 2
    x2, y2 = index_x + new_width // 2, index_y + new_height // 2

    if x1 < 0 : x1 = 0
    if y1 < 0 : y1 = 0
    if x2 > width : x2 = width
    if y2 > height : y2 = height
    # x1, y1 = min(0, x1), min(0, y1)
    # x2, y2 = max(width, x2), max(height, y2)
    
    # Crop and resize
    cropped = img[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (width, height))

    return zoomed

def detect_hand(result):
    if result.multi_hand_landmarks:
        for idx, hand in enumerate(result.multi_hand_landmarks):
            handedness, wrist_x, wrist_y = identify_right_left_hand(result, idx, hand)
            
            # draw the dots on each our image for vizual help
            for datapoint_id, point in enumerate(hand.landmark):
                h, w, c = img.shape
                x, y = int(point.x * w), int(point.y * h)
                # cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)
                
                # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
                if datapoint_id == 4:
                    thumb_x, thumb_y = x, y
                    cv2.circle(img, (thumb_x, thumb_y),
                               6, (255, 0, 0), cv2.FILLED)
                if datapoint_id == 8:
                    index_x, index_y = x, y
                    cv2.circle(img, (index_x, index_y),
                               6, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y),
                     (index_x, index_y), (0, 0, 255), 5)
            
            mpdraw.draw_landmarks(img, hand, mphands.HAND_CONNECTIONS)
            length = int(math.hypot(thumb_x - index_x, thumb_y - index_y))
            print(f"{handedness} hand, Wrist location : ({wrist_x, wrist_y}), thumb location: ({thumb_x, thumb_y}), index location ({index_x, index_y}), length: {length}")
            
            if handedness == 'Right-camera':
                scale = int(np.interp(length, [10,150], [1,10]))
                print(scale)
                if scale > 1: 
                    zoomed_img = zoom_image(img, scale, index_x, index_y)    
                    # cv2.imshow("zoomed", zoomed_img)
                    return zoomed_img
            
            if handedness == 'Left-volume':
                vol = np.interp(length, [10,150], [minVol, maxVol])
                print(vol)
                volume.SetMasterVolumeLevel(vol, None)                
    return None


def detect_face(result):
    # Draw the face detection annotations on the image.
    if result.detections:
        for detection in result.detections:
            mpdraw.draw_detection(img, detection)
    return


# INITIALIZATION
lastFrameTime = 0
DETECTFACE = False
DETECTHAND = True

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# for drawing
mpdraw = mp.solutions.drawing_utils

# for hands
mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands = 2, min_detection_confidence = 0.7)

# for face
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# opening camera (0 for default camera)
videoCap = cv2.VideoCapture(0)

while videoCap:
    # reading image
    success, img = videoCap.read()

    fps, lastFrameTime = calculateFPS(lastFrameTime=lastFrameTime)
    # write on image fps
    cv2.putText(img, f'FPS:{int(fps)}',
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # conver from bgr to rgb
    RGBframe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.flip(img, 1)
    
    # for hand detection
    if DETECTHAND:
        hand_results = hands.process(RGBframe)
        temp_img = detect_hand(hand_results)
        if temp_img is not None:
            img = temp_img
        

    # for face detection
    if DETECTFACE:
        face_result = face_detection.process(RGBframe)
        detect_face(face_result)

    cv2.imshow("CamOutput", img)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
videoCap.release()
cv2.destroyAllWindows()
