import cv2

# Specify the image path for face detection and XML file for the cascade
photo_path = "family.jpg"
cascade_path = "haarcascade_frontalface_default.xml"

# Initialise the Haar CascadeClassifier with the XML file
haar_face_cascade = cv2.CascadeClassifier(cascade_path)

# Read the photo and convert to grayscale
photo = cv2.imread(photo_path)
grayscale = cv2.cvtColor(photo, cv2.COLOR_BGR2grayscale)


# Detect faces in the photo using OpenCV library
faces = haar_face_cascade.detectMultiScale(
        grayscale,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30)
        )

