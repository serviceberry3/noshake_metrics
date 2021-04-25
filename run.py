'''
Realtime 3D Human Reconstruction using Posenet and Facebook's VideoPose3D
3D drawing using pygtagrph based on OpenGL
Speed: TBD
'''
import os
import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.opengl import *
import cv2
import timeit

# progress bar animator
from tqdm import tqdm
import numpy as np
import time
import math
import argparse
import imutils


# initialize some global vars
item = 0
item_num = 0

# initialize pos_init to 17x3 zeros matrix
pos_init = np.zeros((17, 3))

animate = False

#number of frames in which square couldn't be found
missed_frames = 0


class ShapeDetector:
    def __init__(self):
        # nothing to initialize
        pass

    # shape detector, takes contour (outline) of shape we're trying to identify as argument
    # contour approx is an algo for reducing num of pts in a curve w/a reduced set of pts (Ramer-Douglas-Peucker or "split-and-merge" algorithm)
    # curve is approximated by series of short line segments, leading to approximated curve that consists of subset of pts of original curve
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"

        # first compute perimeter of the contour
        peri = cv2.arcLength(c, True)

        # construct actual contour approximation, which consists of a list of vertices
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape has 4 vertices, it is either a square or a rectangle
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            
            
        #return the name of the shape, only if it's been identified successfully (we don't want to draw all the random contours found)
        if shape != "unidentified":
            return shape


# instantiate new shape detector
sd = ShapeDetector()


class Visualizer(object):
    def __init__(self):
        # creat a qt app with no arguments
        self.app = QtGui.QApplication([])

        # Open up a VideoCapture for live frame feed
        self.cap = cv2.VideoCapture(0)

        # set video name
        #self.video_name = input_video.split('/')[-1].split('.')[0]

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            print("Starting qt app")

            # start up the qtgraph gui
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    def invert_image(self, img):
        # get height and width of the image
        height, width = img.shape

        return 255-img

    # Run the model on 30 frames at a time

    def update(self):
        # these globals get updated on every callback
        global item
        global item_num

        # read in a frame from the VideoCapture (webcam)
        _, image = self.cap.read()  # ignore the other returned value

        # rotate the frame
        #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # display the feed
        #cv2.imshow('Camera preview', image)

        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        thresh = self.invert_image(thresh)

        # image has now been binarized and inverted, display it
        cv2.imshow("Thresh", thresh)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:

            shape = None

            # compute the "centers of mass" of each contour in the image
            M = cv2.moments(c)

            compute_com_success = True

            # make sure not div by 0
            if (M["m00"] != 0):
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
            else:
                compute_com_success = False

            # detect name of shape using contour
            shape = sd.detect(c)


            if shape != None:
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

                if (compute_com_success):
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Image", image)
        # cv2.waitKey(0)

        # check for quit signal
        # 'q' button is set as quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.app.quit()

    def animation(self):
        # instantiate a QTimer object to keep track of time during animation
        timer = QtCore.QTimer()

        # start the timer
        timer.start(1)

        # connect the "callback" fxn for timer timeout (to update drawing)
        # lets Python interpreter run each millisecond
        timer.timeout.connect(self.update)

        # start the QApplication
        self.start()


def invert_image(img):
    # get height and width of the image
    height, width = img.shape

    return 255 - img


# Run the model on 30 frames at a time
def find_square_on_img(image):
    # these globals get updated on every callback
    global item
    global item_num
    global animate
    global missed_frames

    #crop and resize image
    cropped = image[180:800, 200:1720] # startY:endY, startX:endX
    resized = imutils.resize(cropped, width=300)

    #print("Resized shape is (rows, cols, channels)", resized.shape)

    #compute resize ratio: ratio of original image height to new image height (pixels of original per pixel of new)
    ratioY = cropped.shape[0] / float(resized.shape[0])
    ratioX = cropped.shape[1] / float(resized.shape[1])

    #print("X and Y ratios are", ratioX, ",", ratioY)

    # convert the resized image to grayscale, blur it slightly, and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    #invert image colors
    thresh = invert_image(thresh)

    #print("Thresh shape is (rows, cols, channels)", thresh.shape)

    # image has now been binarized and inverted, display it
    cv2.imshow("Thresh", thresh)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    correct_contour = None
    winning_yval = 100000
    winning_xval = 0

    print(len(cnts), "contours found in frame")

    # loop over the contours
    for c in cnts:
        #initialize shape var
        shape = None

        #compute the "centers of mass" of each contour in the image
        M = cv2.moments(c)

        compute_com_success = True

        # make sure not div by 0
        if (M["m00"] != 0):
            cX = int((M["m10"] / M["m00"]) * ratioX)
            cY = int((M["m01"] / M["m00"]) * ratioY)
        else:
            compute_com_success = False

        # detect name of shape using contour
        shape = sd.detect(c)

        if shape != None:
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")

            #print(c)
            print("Length of this cntr is", c.size/2)

            #placeholder to keep track of top left and bottom rt pts
            top_left = []
            top_left_sum = 1000000
            btm_rt = []
            btm_rt_sum = 0

            #find corresponding max y val for that x val
            for i in range(0, c.shape[0]): #iterate over all points in contour
                this_x = c[i, :, 0]
                this_y = c[i, :, 1]

                xysum = this_x + this_y

                #iterate over all points in contour checking sum of x and y vals
                if xysum > btm_rt_sum:
                    btm_rt_sum = xysum
                    btm_rt = np.array([this_x, this_y])

                if xysum < top_left_sum:
                    top_left_sum = xysum
                    top_left = np.array([this_x, this_y])
                    
            print("Top left is", top_left, "bottom rt is", btm_rt)

            #if top left or btm rt couldn't be found, jump to next contour
            if btm_rt.size == 0 or top_left.size == 0:
                print("An extraneous rect or square was found in a frame/img, because btm_rt or top_left is []")
                continue

            #otherwise we can successfully compute pixel dimensions of the square
            width = btm_rt[0] - top_left[0]
            height = btm_rt[1] - top_left[1]
            dimsum = width + height

            #to weed out contours that aren't the black square, put threshhold on size of square and diff between height and width
            if dimsum < 45 and dimsum > 10 and abs(width - height) < 20:
                print("This contour width is", width, "and height is", height) 

                #we could still have some extraneous contours that were found, so we'll keep track of which one is highest in image (extraneous ones tend to be found on bottom of drone)

                #if we were able to compute the center of mass
                if compute_com_success and cY < winning_yval:
                    print("Center of mass of this contour found was", cX, cY)
                    print("New highest contour in image found, setting correct_contour")

                    winning_yval = cY
                    winning_xval = cX
                    correct_contour = c
                else:
                    print("FAIL on compute_com_success and cY < winning_yval")
            '''else:
                print("FAIL on dimension and squarish check")'''
        '''else:
            print("FAIL on shape != None")'''

    #if we found some contour
    if correct_contour is not None:
        #print("Correct_contour is an array")

        #scale up the contour appropriately in both x and y dims
        correct_contour[:, :, 0] *= ratioX
        correct_contour[:, :, 1] *= ratioY

        correct_contour = correct_contour.astype("int")

        #draw the contour in green on original cropped color image
        cv2.drawContours(cropped, [correct_contour], -1, (0, 255, 0), 2)

        #write name of shape at the "center of mass" of the contour
        cv2.putText(cropped, shape, (winning_xval, winning_yval), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #otherwise we didn't find the square, increment missed_frames    
    else:
        #print("This contour width is", width, "and height is", height) 
        #print("FAIL on correct_contour is not None")
        cv2.waitKey(0)

    
        missed_frames += 1
            

    # show the output image
    cv2.imshow("Image", cropped)

    #if a was pressed, play back the processed video, waiting 10 ms on each frame
    if (animate):
        k = cv2.waitKey(5)

    else:
        #wait indefinitely for a key press
        k = cv2.waitKey(0)

        #if key is n, move to next frame of vid
        if k == ord('n'):
            cv2.destroyAllWindows()

        #if key is q, close all windows and quit program
        elif k == ord('q'): 
            cv2.destroyAllWindows()
            quit()

        #if key is a, play the video frame by frame, processing the data
        elif k == ord('a'):
            animate = True



def process_img(img):
    #open the image using OpenCV
    opened_image = cv2.imread(img)

    print("The passed image shape is (rows, cols, channels) ", opened_image.shape)

    find_square_on_img(opened_image)
    print("Processing single image...")


def process_video(vid):
    # read the specified vid using OpenCV
    # Read the video from specified path
    cam = cv2.VideoCapture(vid)

    currentframe = 0

    while (True):
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if there's still video left to process, continue processing images
            # load the image and resize it to a smaller factor so that
            # the shapes can be approximated better
            find_square_on_img(frame)

            #cv2.imshow("Current frame", frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    print(currentframe, "frames found from video")
    print(missed_frames, "frames missed")

    # Release all space and windows once done
    cam.release()


def start_realtime():
    # Instantiate a Visualizer object for the input video file
    v = Visualizer()

    v.animation()

    # Close all open windows after animation ends
    cv2.destroyAllWindows()


# Main entrance point
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    # arg to input a single image or video
    ap.add_argument("-i", "--image", required=False,
                    help="path to the input image")
    ap.add_argument("-v", "--video", required=False,
                    help="path to the input video")
    args = vars(ap.parse_args())

    if args["image"] != None and args["video"] != None:
        print("You can't pass in an image and a video at the same time.")
    elif args["image"] != None:
        process_img(args["image"])
    elif args["video"] != None:
        process_video(args["video"])

    else:
        try:
            start_realtime()
        except KeyboardInterrupt:
            print('Interrupted')
            cv2.destroyAllWindows()
            sys.exit(0)
