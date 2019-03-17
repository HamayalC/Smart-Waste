import time


import tensorflow as tf
import os

graph_def = tf.GraphDef()
labels = []

import RPi.GPIO as GPIO

def SetAngle1(angle):
	duty = angle / 18 + 2
	GPIO.output(01, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(01, False)
	pwm.ChangeDutyCycle(0)

def SetAngle2(angle):
	duty = angle / 18 + 2
	GPIO.output(02, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(02, False)
	pwm.ChangeDutyCycle(0)

def SetAngle3(angle):
	duty = angle / 18 + 2
	GPIO.output(03, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(03, False)
	pwm.ChangeDutyCycle(0)



# Import the TF graph
with tf.gfile.FastGFile("model.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open("labels.txt", 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

import cv2
import numpy as np

from picamera.array import PiRGBArray
from picamera import PiCamera






# Set up camera constants
IM_WIDTH = 227
IM_HEIGHT = 227


#IM_WIDTH = 1280
#IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)


for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):



    t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    cv2.imshow('Camera display', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    #Actually prediction of image:
    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'


    with tf.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions, = sess.run(prob_tensor, {input_node: [frame] })

    #Viewing the results
        # Print the highest probability label
        highest_probability_index = np.argmax(predictions)
        print('Classified as: ' + labels[highest_probability_index])
        print(highest_probability_index)
        print()




        if (labels[highest_probability_index]) == "can"
            setAngle2(110)
            time.sleep(2)
            setAngle2(20)


        # Or you can print out all of the results mapping labels to probabilities.
        #label_index = 0
        #for p in predictions[0]:
        #    truncated_probablity = np.float64(np.round(p,8))
        #print (labels[label_index]) #truncated_probablity)
        #label_index += 1



    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()

pwm.stop()
GPIO.cleanup()
