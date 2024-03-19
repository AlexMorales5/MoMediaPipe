import cv2
import numpy as np
from pupil_apriltags import Detector

# Initialize the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize the AprilTag detector
detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.2,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

def draw_3d_spheres(img, corners):
    for corner in corners:
        center = tuple(corner)
        radius = 22  # Adjust the radius as needed
        color = (255, 12, 25)  # Green color, you can customize this

        cv2.circle(img, center, radius, color, -1)  # -1 fills the circle

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    detections = detector.detect(gray)

    # Draw the detected tags on the frame using spheres
    for detection in detections:
        corners = detection.corners.astype(int)
        draw_3d_spheres(frame, corners)

    # Display the frame with detected tags
    cv2.imshow('AprilTag Detection', frame)

    # Break the loop if the 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
