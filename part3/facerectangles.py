import dlib
import cv2
import pyautogui
import numpy as np
import time



def read_image(path):
    image = cv2.imread(path)
    return image

def check_rectangles(rectangles):
    return True if len(rectangles) == 1 else False

def press_button(button, sleep_time=1):
    pyautogui.keyDown(button)
    time.sleep(sleep_time)
    pyautogui.keyUp(button)


def unpress_button(button, sleep_time=1):
    time.sleep(sleep_time)
    pyautogui.keyUp(button)


def landmark_area(img):
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    rectangles = detector(gray)
    #check if there is only one rectangle
    check_rectangles(rectangles)
    # Locate rectangle
    br_x = rectangles[0].br_corner().x
    br_y = rectangles[0].br_corner().y
    tl_x = rectangles[0].tl_corner().x
    tl_y = rectangles[0].tl_corner().y
    cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0), 2)
    horizontal = abs
    vertical = abs(abs(br_y) - abs(tl_y))
    area = abs((abs(br_x) - abs(tl_x)) * (abs(br_y) - abs(tl_y)))
    #print(area)
    return area



time.sleep(3)

im = np.array(pyautogui.screenshot())
area = landmark_area(im)
faceareamax = area
#keys = ['w','a','s','d']
offset = 0
while True:
    im = np.array(pyautogui.screenshot())
    area = landmark_area(im)

    if area < faceareamax: #if shocked turn 90 degree
        print("im shocked")
        if offset == 3:
            offset = 0
        else:
            offset += 1
    else:
        print("im normal")

    if offset == 0:
        press_button('w', sleep_time=0.15)
        print('key pressed w')
    elif offset == 1:
        press_button('a', sleep_time=0.15)
        print('key pressed a')
    elif offset == 2:
        press_button('s', sleep_time=0.15)
        print('key pressed s')
    elif offset == 3:
        press_button('d', sleep_time=0.15)
        print('key pressed d')

















