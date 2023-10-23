import cv2 as cv
from cvzone import HandTrackingModule
import mediapipe as mp

img = cv.imread('sample.jpg')

detector = HandTrackingModule.HandDetector()

hands, img = detector.findHands(img)

  # img = cv2.flip(img, 1)
cv.imshow("Hands Detected", img)

# if cv.waitKey(1) == ord('x'):
cv.waitKey(0)

cv.destroyAllWindows()