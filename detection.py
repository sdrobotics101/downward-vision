import vision
import cv2
import io
import numpy
import time
import sys
import picamera
sys.path.insert(0, "DistributedSharedMemory/build")
import pydsm
sys.path.insert(0, "RobotLogger")
import logger
sys.path.insert(0, "PythonSharedBuffers/src")
from Constants import *
from Vision import *
from Serialization import *

#var for whether or not this run of the program is a test
isReal = True

if len(sys.argv) > 1:
	isReal = False

#initialize log
log = logger.LogWriter("downward")
log.write("Beginning downward detection")

#initialize buffer
if isReal:
	client = pydsm.Client(DOWNWARD_VISION_SERVER_ID, 60, True)
	client.registerLocalBuffer(TARGET_LOCATION_AND_ROTATION, sizeof(LocationAndRotation), False)
	loc = LocationAndRotation()

#initialize capture device
print("Initializing camera")
if isReal:
	camera = picamera.PiCamera()
	camera.resolution = (1920, 1080)
	camera.start_preview()
	time.sleep(1)

# capture frames sequentially
while True:
	if isReal:
		#Create a memory stream so photos doesn't need to be saved in a file
		stream = io.BytesIO()

		#Get the picture (low resolution, so it should be quite fast)
		#Here you can also specify other parameters (e.g.:rotate the image)
		camera.capture(stream, format='jpeg')

		#Convert the picture into a numpy array
		buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

		#Now creates an OpenCV image
		image = cv2.imdecode(buff, 1)	
		print("Getting image")
	
	else:
		image = cv2.imread("testImages/picture2.png")
	
	#send to vision module
	print("Sending to vision")
	rot, tran, conf = vision.detectRect(image, isReal)

	log.write("Location: " + str(tran))
	log.write("Orientation: " + str(rot))
	log.write("Confidence: " + str(conf))
	log.write(" ")

	#update buffers (translations to location buffer)
	if isReal:
		loc.confidence = max(0, min(int(conf), 255-1))
		client.setLocalBufferContents(TARGET_LOCATION_AND_ROTATION, Pack(loc))
	else:
		exit()
