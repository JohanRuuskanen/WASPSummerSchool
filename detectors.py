# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:44:45 2018

@author: mayerick
"""

import numpy as np
import cv2 
import os

class Detector():
    def __init__(self):
        pass
    def detect(self):
        raise NotImplementedError
        


class HSVDetector(Detector):

    def __init__(self, preset_path=None):
        super(HSVDetector, self).__init__()
        
        self.lower_hsv = []
        self.higher_hsv = []
        
        
        if preset_path is not None:
            if os.path.isfile(preset_path):
                f = np.load('hsv_thresh.npz')
                self.lower_hsv = f['min_thresh']
                self.higher_hsv = f['max_thresh']   
            else:
                print('Preset files was not found! Quitting ..')
                raise RuntimeError
        else:
            print('Preset files must be provided.')
            raise RuntimeError
        
        
    def detect(self, frame):
        bb = None
        
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the frame
        mask = cv2.inRange(hsv, self.lower_hsv, self.higher_hsv)
        
    
        # Find contours 
        _, cnts, hierarchy = cv2.findContours(mask, 1, 2)
        
        # Sort contours and select the largest
        if len(cnts) > 0:
            cnts_sorted = sorted(cnts, key = cv2.contourArea, reverse=True)
            cnt = cnts_sorted[0]
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 :
                solidity = float(area)/hull_area
            else:
                solidity=0
                
            if  area > 100 and solidity > 0.7:
                # Fit a bounding box and plot
                x,y,w,h = cv2.boundingRect(cnt)
                bb = [x,y,w,h]
        
        return bb
        