import cv2
from cvzone import HandTrackingModule
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

detector = HandTrackingModule.HandDetector()

point_a = (0, 0)
point_b = (0, 0)
point_c = (0, 0)

while True:
  ret, img = cap.read()
  imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(imageRGB)
  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      for id, lm in enumerate(handLms.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if id == 4:
          point_a = (cx, cy)
          cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        if id == 8:
          point_b = (cx, cy)
          cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)      
        if id == 12:
          point_c = (cx, cy)
          cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    triangle_cnt = np.array( [point_a, point_b, point_c] ) 

    cv2.drawContours(img, [triangle_cnt], 0, (0,255,0), 3)

  # hands, img = detector.findHands(img) # HandTrackingModule of cvzone

  # img = cv2.flip(img, 1) 
  cv2.imshow("Hands Detected", img)

  if cv2.waitKey(1) == ord('x'):
    break

cap.release()
cv2.destroyAllWindows()