import cv2 as cv2
import mediapipe as mp
import math


class Rectangle():
    def __init__(self, fill, cx, cy, width, height):
        self.fill = fill
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.x1 = self.cx - self.width //2
        self.y1 = self.cy - self.height //2
        self.x2 = self.cx + self.width //2
        self.y2 = self.cy + self.width //2
        
    def update(self, index_x, index_y):
        self.x1 = index_x - self.width//2
        self.y1 = index_y - self.height//2
        self.x2 = index_x + self.width//2
        self.y2 = index_y + self.height//2
        
    def inside_rect(self, index_x, index_y):
        if self.x1 < index_x < self.x2 and self.y1 < index_y < self.y2:
            return True
        return False

def detect_hand(result, occupied, rect_id):
    if result.multi_hand_landmarks:
        for idx, hand in enumerate(result.multi_hand_landmarks):
            
            # draw the dots on each our image for vizual help
            for datapoint_id, point in enumerate(hand.landmark):
                h, w, c = img.shape
                x, y = int(point.x * w), int(point.y * h)
                # cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)
                
                # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
                if datapoint_id == 8:
                    index_x, index_y = x, y
                    cv2.circle(img, (index_x, index_y),
                               6, (255, 0, 0), cv2.FILLED)
                    
                if datapoint_id == 12:
                    middle_x, middle_y = x, y
                    cv2.circle(img, (middle_x, middle_y),
                               6, (255, 0, 0), cv2.FILLED)
            
            cv2.line(img, (middle_x, middle_y),
                     (index_x, index_y), (0, 0, 255), 5)
            length = int(math.hypot(middle_x - index_x, middle_y - index_y))
            mpdraw.draw_landmarks(img, hand, mphands.HAND_CONNECTIONS)                      
            
            if occupied == False:
                for rect_id, rect in enumerate(rect_list):
                    if rect.inside_rect(index_x, index_y) and length <= 30:
                        print(rect_id)
                        occupied = True
                        rect.fill = blue_color
                        rect.update(index_x, index_y)
                        break
            elif occupied == True and rect_id is not None and rect_list[rect_id].inside_rect(index_x, index_y) and length <= 30:
                print('occupied == True and rect_list[rect_id].inside_rect(index_x, index_y) and length <= 30')
                rect_list[rect_id].update(index_x, index_y)
            elif length > 30 and rect_id is not None:
                print('length>30')
                occupied = False
                rect_list[rect_id].fill = red_color

            # print(occupied, rect_id, rect_list[rect_id].inside_rect(index_x, index_y), length)
    print(occupied, rect_id)
    return occupied, rect_id

# create rectangles
red_color = (0,0,255)
blue_color = (255,0,0)
rect_list = []


for x in range(5):
    rect = Rectangle(red_color, (x * 250 + 150), 150, 200, 200)
    rect_list.append(rect)
    
cap = cv2.VideoCapture(0)
cap.set(3,1280) # length
cap.set(4, 720) # height

# for drawing
mpdraw = mp.solutions.drawing_utils

# for hands
mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands = 2, min_detection_confidence = 0.7)
rect_id = None
occupied = False
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # conver from bgr to rgb
    RGBframe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(RGBframe)
    occupied, rect_id = detect_hand(hand_results, occupied, rect_id)
    
    # add rectangles
    for rect in rect_list:
        cv2.rectangle(img, (rect.x1, rect.y1), (rect.x2, rect.y2), rect.fill, cv2.FILLED)
        
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
