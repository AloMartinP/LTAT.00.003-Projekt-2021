from time import time
import cv2 as cv
from vision import Vision
from windowcapture import WindowCapture
import pygetwindow as gw

# Find runescape client should work with any
def runeliteClient():
    for wnd in gw.getAllTitles():
        if wnd.startswith('Rune'):
            return wnd
    raise Exception('Window not found')

wincap = WindowCapture(runeliteClient())

cascade_cow = cv.CascadeClassifier('cascade_training/cascade.xml')

vision_cow = Vision(None)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    rectangles = cascade_cow.detectMultiScale(screenshot)

    detection_image = vision_cow.draw_rectangles(screenshot, rectangles)

    # display the processed image
    cv.imshow('Osrs-OpenCV', detection_image)
    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break