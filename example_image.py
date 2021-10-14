"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

for i in range(1,6):
    gaze = GazeTracking()

    # We get a new frame from the webcam
    input_name = 'examples/example%d' % i
    frame = cv2.imread('%s.jpg' % input_name)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    gaze.draw_origins(frame)
    gaze.draw_eye_landmarks(frame)
    # gaze.draw_all_landmarks(frame)
    gaze.draw_eye_iris(frame)
    # gaze.draw_eye_iris_ellipse(frame)
    text = ""

    if gaze.is_blinking()[0]:
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
    cv2.putText(frame, "Horizontal ratio: %.2f" % gaze.horizontal_ratio(), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 212, 255), 1)
    cv2.putText(frame, "Blinking ratio: %.2f" % gaze.is_blinking()[1], (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0, 212, 255), 1)


    cv2.imwrite("%s_result2.jpg" % input_name, frame)
    cv2.destroyAllWindows()
