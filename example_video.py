"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import time
import numpy as np


gaze = GazeTracking()

# read video
input_path = 'examples/gaze_tracking_demo.mp4'
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('input %s' % input_path)

# write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = input_path.replace('.mp4', '_result.mp4')
writer = cv2.VideoWriter(output_path, fourcc, fps, size)
print('output %s' % output_path)

retval, frame = cap.read()
frame_id = 0
fps_list = []
while retval:
    start_time = time.time()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    # gaze.draw_origins(frame)
    # gaze.draw_eye_landmarks(frame)
    # gaze.draw_all_landmarks(frame)
    # gaze.draw_eye_iris(frame)
    # gaze.draw_eye_iris_ellipse(frame)
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 212, 255), 1)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    # cv2.putText(frame, "Horizontal ratio: %.2f" % gaze.horizontal_ratio(), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 212, 255), 1)
    # cv2.putText(frame, "Blinking ratio: %.2f" % gaze.is_blinking()[1], (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0, 212, 255), 1)

    # fps log
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    print('%03d fps: %.2f' % (frame_id, fps))
    fps_list.append(fps)
    frame_id += 1

    writer.write(frame)
    retval, frame = cap.read()
    
print('mean fps: %.2f' % np.mean(np.array(fps_list)))