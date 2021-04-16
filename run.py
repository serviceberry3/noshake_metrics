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

#progress bar animator
from tqdm import tqdm
import numpy as np
import time
import math


#initialize some global vars
item = 0
item_num = 0

#initialize pos_init to 17x3 zeros matrix
pos_init = np.zeros((17,3))


class Visualizer(object):
    def __init__(self):
        #creat a qt app with no arguments
        self.app = QtGui.QApplication([])

        #Open up a VideoCapture for live frame feed
        self.cap = cv2.VideoCapture(0)

        #set video name
        #self.video_name = input_video.split('/')[-1].split('.')[0]


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            print("Starting qt app")

            #start up the qtgraph gui
            QtGui.QApplication.instance().exec_()    


    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    
    #Run the model on 30 frames at a time
    def update(self):
        #these globals get updated on every callback
        global item
        global item_num

        #read in a frame from the VideoCapture (webcam)
        _, frame = self.cap.read() #ignore the other returned value

        #rotate the frame
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #display the feed
        cv2.imshow('Camera preview', frame)

        #check for quit signal
        #'q' button is set as quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
            self.app.quit()


    def animation(self):
        #instantiate a QTimer object to keep track of time during animation
        timer = QtCore.QTimer()

        #start the timer
        timer.start(1)

        #connect the "callback" fxn for timer timeout (to update drawing)
        #lets Python interpreter run each millisecond
        timer.timeout.connect(self.update)

        #start the QApplication
        self.start()

def main():
    #Instantiate a Visualizer object for the input video file
    v = Visualizer()

    v.animation()

    #Close all open windows after animation ends
    cv2.destroyAllWindows()

#Main entrance point
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        cv2.destroyAllWindows()
        sys.exit(0)