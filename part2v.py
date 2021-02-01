import pyautogui
import time
import numpy as np
import cv2
import os.path
import sys
import matplotlib.pyplot as plt
from PIL import Image

directions = np.array(['w', 's', 'a', 'd'])

def get_screenshot(skip_if_exist=False, latency=5):
    time.sleep(latency)
    image = pyautogui.screenshot()
    return image

def press_button(button, sleep_time=1):
    pyautogui.keyDown(button)
    time.sleep(sleep_time)
    pyautogui.keyUp(button)

def unpress_button(button, sleep_time=1):
    time.sleep(sleep_time)
    pyautogui.keyUp(button)



if __name__ == '__main__':
    time.sleep(5)
    '''#to get the screenshots of the shocked faces 
    ss_shockedface = get_screenshot(skip_if_exist=False, latency=3)
    ss_shockedface.save(r"ss_shockedface.png")
    img = cv2.imread('ss_shockedface.png')
    w, h = img.shape[:-1]
    top=900
    right=1700
    down=0
    left=0
    crop = img[top:((w-down)+top), right:((h-left)+right)]
    cv2.imwrite("ss_shockedface1.png", crop) 
    ss_shockedface = get_screenshot(skip_if_exist=False, latency=3)
    ss_shockedface.save(r"ss_shockedface.png")
    img = cv2.imread('ss_shockedface.png')
    w, h = img.shape[:-1]
    top=900
    right=1700
    down=0
    left=0
    crop = img[top:((w-down)+top), right:((h-left)+right)]
    cv2.imwrite("ss_shockedface2.png", crop) 
    ss_shockedface = get_screenshot(skip_if_exist=False, latency=3)
    ss_shockedface.save(r"ss_shockedface.png")
    img = cv2.imread('ss_shockedface.png')
    w, h = img.shape[:-1]
    top=900
    right=1700
    down=0
    left=0
    crop = img[top:((w-down)+top), right:((h-left)+right)]
    cv2.imwrite("ss_shockedface3.png", crop) 
    '''
    shocked1 = cv2.imread('ss_shockedface1.png')
    shocked2 = cv2.imread('ss_shockedface2.png')
    shocked3 = cv2.imread('ss_shockedface3.png')
    #cv2.imshow('Image', shocked1)
    #cv2.imshow('Image2', shocked2)
    #cv2.imshow('Image3', shocked3)
    #cv2.waitKey(0)
    
    #while True:
    for i in range (5):
        #get screenshot
        my_screenshot = get_screenshot(skip_if_exist=False, latency=1)
        my_screenshot.save(r"ss.png")
        img = cv2.imread('ss.png')
        #crop it (face)
        w, h = img.shape[:-1]
        top=900
        right=1700
        down=0
        left=0
        crop = img[top:((w-down)+top), right:((h-left)+right)]
        #cv2.imshow('Image', crop)
        #cv2.waitKey(0)

        #try to move to directions
        for i in range (4):
            direction = directions[i]
            #print(direction)

            #move to this direction
            #get screen shots (maybe mor than once)

            #compare it with the shockedface images
                #keep flag same=1
        
            #if same
              #recursion??

      
        
    
