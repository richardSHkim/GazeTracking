import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        # threshold = calibration.find_best_threshold(self.frame)
        self.pupil = Pupil(self.frame, threshold)

    def draw_origin(self, frame):
        # draw self.origin
        cv2.circle(frame, (self.origin), radius=2, color=(0,0,255), thickness=-1)
    
    def draw_center(self, frame):
        # draw self.center
        center_x = self.origin[0] + int(self.center[0])
        center_y = self.origin[1] + int(self.center[1])
        cv2.circle(frame, (center_x, center_y), radius=2, color=(255,255,255), thickness=-1)
    
    def draw_landmarks(self, frame):
        cv2.polylines(frame, [self.landmark_points], isClosed=True, color=(0, 0, 255), thickness=1)
        
        for pt in self.landmark_points:
            cv2.circle(frame, pt, radius=2, color=(255,255,255), thickness=-1)

    def draw_iris(self, frame):
        pts = np.array([(self.origin[0] + pt[0][0], self.origin[1] + pt[0][1]) for pt in self.pupil.contours])
        pts = pts.astype(np.int32)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

        # for pt in self.pupil.contours:
            # cv2.circle(frame, pt, radius=2, color=(255,255,255), thickness=-1)
    
    def draw_iris_ellipse(self, frame):
        # cv2.circle(frame, (self.origin[0] + self.pupil.c[0], self.origin[1] + self.pupil.c[1]), self.pupil.r, color=(255,0,0), thickness=1)
        self.pupil.e[0] = list(self.pupil.e[0])
        self.pupil.e[0][0] += self.origin[0]
        self.pupil.e[0][1] += self.origin[1]
        cv2.ellipse(frame, self.pupil.e, color=(0,0,255), thickness=1)
    
    def draw_pupil_center(self, frame):
        center = (self.origin[0] + int(self.center[0]), self.origin[1] + int(self.center[1]))
        cv2.circle(frame, center, radius=2, color=(255,255,255), thickness=-1)