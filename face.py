import cv2 as cv


img = cv.imread('Friends.jpg')

# cv.imshow('Friends', img)

# gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Use Cascade to recognize
face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Modify
face = face_classifier.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50)
)

# Draw a rectangle
for (x, y, w, h) in face:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# img_rgb = cv.cvtColor(img, cv.COLOR_RGB2BGR)


cv.imshow('Friends', img)
# print(img.shape)


cv.waitKey(0)
cv.destroyAllWindows()