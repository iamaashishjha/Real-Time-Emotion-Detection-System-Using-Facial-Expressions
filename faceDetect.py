import cv2

# Load the Haar cascade classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input Video
video_capture = cv2.VideoCapture(0)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()