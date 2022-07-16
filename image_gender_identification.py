# gender detection example
# usage: python image_gender_identification.py <input_image>

# references:
# https://github.com/arunponnusamy/cvlib/blob/master/examples/gender_detection.py
# https://github.com/arunponnusamy/cvlib/blob/master/examples/face_detection.py

# import libraries
import cv2
import cvlib as cv
import sys
import numpy as np

# read input image
image = cv2.imread(sys.argv[1])

# apply face detection
faces, confidences = cv.detect_face(image)
print(faces)
print(confidences)

padding = 20
color = {'male': (0,255,0), 'female':(0,0,255)}

for i in faces:
    (x, y) = max(0, i[0]-padding), max(0, i[1]-padding)
    (x2, y2) = min(image.shape[1]-1, i[2]+padding), min(image.shape[0]-1,i[3]+padding)

    crop = np.copy(image[y:y2, x:x2])
    label, confidences = cv.detect_gender(crop)
    idx = np.argmax(confidences)
    label = label[idx]
    legend = f"{label}: {confidences[idx] * 100:.2f}"

    # draw rectangle around faces
    cv2.rectangle(image, (x, y), (x2, y2), color[label], 2)
    # put legend above rectangular boxes
    Y = y - 10 if y -10 > 10 else y + 10
    cv2.putText(image, legend, (x-5, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[label], 2)

print("detection starting")
cv2.imshow("Gender Detection", image)
cv2.imwrite("face_identification.jpg", image)
cv2.waitKey(1)
cv2.destroyAllWindows()
input("press key")


