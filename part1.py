import numpy as np
import cv2
import pyautogui
import time

def hough_circles(image,outpath='detected circles.png'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = np.uint16(np.around(circles))
    return circles[0, :]

def get_screenshot(latency=0):
    time.sleep(latency)
    ss = np.array(pyautogui.screenshot())
    ss = cv2.cvtColor(np.array(ss), cv2.COLOR_RGB2BGR)
    return ss

time.sleep(3)
while True:
    original = get_screenshot()
    circles = hough_circles(original)
    img = original.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=100,
                            maxLineGap=80)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    #cv2.imwrite("linesDetected.png", img)

    hold_values = []
    for circle in circles:
        axis_vertical = circle[1]
        axis_horizontal = circle[0]
        left = axis_horizontal
        while (img[axis_vertical, left, 2] != 255):
            left = left - 1
        hold_values.append(left)

        """right = axis_horizontal
        while (img[axis_vertical, right,2] != 255):
            right = right + 1
        """

    hold_values.sort()
    my_dict = {i: hold_values.count(i) for i in hold_values}
    max_key = max(my_dict, key=my_dict.get)
    mylist = list(my_dict)
    index = mylist.index(max_key)  # returns the max key index of the dot counts 0 1 or 2

    if index == 0:
        print("a pressed")
        pyautogui.press('a')
    elif index == 1:
        print("s pressed")
        pyautogui.press('s')
    elif index == 2:
        print("d pressed")
        pyautogui.press('d')
    else:
        print("nothing pressed")


