import sys
import math
import pickle
import cv2
import numpy as np
import calVal

#CONSTANTS
LAB_ORANGE = [166, 135, 163]

MIN_SIZE = 50
SIZE_BIAS = 160
MAX_DELTA_E = 150
DELTA_E_BIAS = 0.6
MAX_DEFECT = 0.5
DEFECT_BIAS = 0

def estimatePose(contour):
	rvecs = []
	tvecs = [0, 0, 0]
	#get image points of corners of rounding rect
	pos, size, rot = cv2.minAreaRect(contour)
	#find rotation of rectangle
	if size[0] > size[1]:
		rvecs = [0, 0, rot + 90]
	else:
		rvecs = [0, 0, rot]

	return rvecs, tvecs
	

def displayResults(contours, contourConfidence, originalImg, thresh):
	for index, cnt2 in enumerate(contours):
		if contourConfidence[index] != 0:
			cv2.drawContours(originalImg, [cnt2], 0, (100, 255, 255), 3)
			x,y,w,h = cv2.boundingRect(cnt2)
			cv2.rectangle(originalImg, (x, y), (x + w, y + h), (50, 0, 50), 2)
			cv2.putText(originalImg, str(contourConfidence[index]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
	
	cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
	cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
	cv2.imshow("Contours", originalImg)
	cv2.imshow("Threshold", thresh)
	cv2.waitKey(0)

def alternateDisplay(contour, originalImg):
	#get image points of corners of rounding rect
	rect = cv2.minAreaRect(contour)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(originalImg, [box], 0, (0, 0, 255), 10)
	cv2.namedWindow("Bounding Rect", cv2.WINDOW_NORMAL)
	cv2.imshow("Bounding Rect", originalImg)
	cv2.waitKey(0)

def sizeTest(contour):
	discard = False
	conf = 0
	size = cv2.contourArea(contour)
	if size < MIN_SIZE:
		#too small
		discard = True
	else:
		conf = (1 - (0.995 ** (float(size - MIN_SIZE)))) * SIZE_BIAS
	
	return conf, discard

def colorTest(contour, labImg):
	discard = False
	conf = 0
	#find centroid of tested contour
	M = cv2.moments(contour)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	labColor = labImg[cY][cX]
	#find deltaE at centroid
	deltaE = math.sqrt((labColor[0] - LAB_ORANGE[0]) ** 2 + (labColor[1] - LAB_ORANGE[1]) ** 2 + (labColor[2] - LAB_ORANGE[2]))
	if deltaE > MAX_DELTA_E:
		#color not close enough
		discard = True
	else:
		conf = (MAX_DELTA_E - deltaE) * DELTA_E_BIAS
	
	return conf, discard

def convexityTest(contour):
	discard = False
	conf = True
	size = cv2.contourArea(contour)
	hullSize = cv2.contourArea(cv2.convexHull(contour, returnPoints=True))
	totalDefect = float(hullSize - size) / hullSize
	#Find if there are too many "serious" defects, and it is for sure not convex
	if totalDefect > MAX_DEFECT:
		#discard
		discard = True
	else:
		conf = (MAX_DEFECT - totalDefect) * DEFECT_BIAS
	
	return conf, discard

def detectRect(originalImg, real):
	#detection preparation adjustments
	img = cv2.GaussianBlur(originalImg, (7, 7), 0)
	labImg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	thresh = cv2.Canny(img, 20, 30)

	#search for contours
	im, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contourConfidence = [0] * len(contours)
	bestScore = 0
	bestIndex = 0	

	#run tests on all contours
	for index, cnt in enumerate(contours):
		peri = cv2.arcLength(cnt, True)
		cnt = cv2.approxPolyDP(cnt, 0.05 * peri, True)
		#for all tests, conf = confidence rating and dis = discard
		#size
		conf, dis = sizeTest(cnt)
		if dis:
			contourConfidence[index] = 0
			continue
		else:
			contourConfidence[index] += conf
		
		#color
		conf, dis = colorTest(cnt, labImg)
		if dis:
			contourConfidence[index] = 0
			continue
		else:
			contourConfidence[index] += conf

		#convexity
		conf, dis = convexityTest(cnt)
		if dis:
			contourConfidence[index] = 0
			continue
		else:
			contourConfidence[index] += conf
		
		if contourConfidence[index] > bestScore:
			bestScore = contourConfidence[index]
			bestIndex = index
	
	#pose estimation on best guess
	if real == False:
		displayResults(contours, contourConfidence, img, thresh)
		cv2.destroyAllWindows()

	rotations, translations = estimatePose(contours[bestIndex])

	if real == False:
		print("Rotation: (Yaw, Pitch, Roll)")
		print(rotations)
		print("Confidence:" + str(contourConfidence[bestIndex]))
	
	return rotations, translations, contourConfidence[bestIndex]
