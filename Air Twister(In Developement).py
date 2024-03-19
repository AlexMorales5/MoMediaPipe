import cv2
import mediapipe as mp
import random

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize VideoCapture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Create a fullscreen window
cv2.namedWindow("Pose Landmarks", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pose Landmarks", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Dictionary to store colors for each landmark
landmark_colors = {
    0: (233, 234, 236),   # Nose
    1: (233, 234, 236),   # Left eye inner
    2: (233, 234, 236),   # Left eye
    3: (233, 234, 236),   # Left eye outer
    4: (233, 234, 236),   # Right eye inner
    5: (233, 234, 236),   # Right eye
    6: (233, 234, 236),   # Right eye outer
    7: (233, 234, 236),   # Left ear
    8: (233, 234, 236),   # Right ear
    9: (233, 234, 236),   # Left shoulder
    10: (233, 234, 236),  # Right shoulder
    11: (233, 234, 236),  # Left elbow
    12: (233, 234, 236),  # Right elbow
    13: (233, 234, 236),  # Left wrist
    14: (233, 234, 236),  # Right wrist
    15: (255, 0, 0),      # Left hip
    16: (0, 0, 255),      # Right hip
    17: (255, 0, 0),      # Left knee
    18: (0, 0, 255),      # Right knee
    19: (255, 0, 0),      # Left ankle
    20: (0, 0, 255),      # Right ankle
    21: (255, 0, 0),      # Left eye pupil
    22: (0, 0, 255),      # Right eye pupil
    23: (233, 234, 236),  # Left toe
    24: (233, 234, 236),  # Right toe
    25: (233, 234, 236),  # Left heel
    26: (233, 234, 236),  # Right heel
    27: (0, 255, 0),      # Left foot inner
    28: (0, 255, 255),    # Right foot inner
    29: (0, 255, 0),      # Left foot index
    30: (0, 255, 255),    # Right foot index
    31: (0, 255, 0),      # Head
    32: (0, 255, 255)     # Neck
}

# Function to add text with a custom font
def add_text(image, text, position, color=(255, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Function to check if a point is within a box
def is_point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Generate colored box positions
box_size = 60
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]  # Blue, Red, Green, Yellow,

# Randomize the positions according to the requirement
# Green and yellow squares are placed at the bottom, while blue and red squares are placed at the top
box_positions = [(random.randint(0, 640 - box_size), random.randint(0, 240 - box_size)) for _ in range(2)]  # Green and yellow
box_positions.extend([(random.randint(0, 640 - box_size), random.randint(240, 480 - box_size)) for _ in range(2)])  # Blue and red

# Function to refresh box positions
def refresh_box_positions():
    global box_positions
    # Randomize positions while maintaining the color-wise placement
    box_positions = [(random.randint(0, 640 - box_size), random.randint(0, 240 - box_size)) for _ in range(2)]  # Green and yellow
    box_positions.extend([(random.randint(0, 640 - box_size), random.randint(240, 480 - box_size)) for _ in range(2)])  # Blue and red

# Function to handle mouse click events
def handle_click(event, x, y, flags, param):
    global box_positions
    if event == cv2.EVENT_LBUTTONDOWN:
        button_position = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 200, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 70,
                           int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 20, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20)
        if button_position[0] <= x <= button_position[2] and button_position[1] <= y <= button_position[3]:
            refresh_box_positions()

# Register mouse click event handler
cv2.setMouseCallback("Pose Landmarks", handle_click)

# Function to check if a point from each required group is within the corresponding colored box
def check_requirements(landmark_indices, box_positions):
    if results.pose_landmarks is None:
        return False
    
    for group, (x, y) in zip(landmark_indices, box_positions):
        if not any(is_point_in_box((int(results.pose_landmarks.landmark[i].x * frame.shape[1]), 
                                    int(results.pose_landmarks.landmark[i].y * frame.shape[0])), 
                                    (x, y, x + box_size, y + box_size)) for i in group):
            return False
    return True


# Initialize game variables
points = 0
last_points = 0
refresh_boxes = True

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get landmarks
    results = pose.process(rgb_frame)

    # Draw landmarks on the frame with individual colors
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if landmark.visibility > 0.5:  # Check landmark visibility
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                color = landmark_colors.get(idx, (169, 169, 169))  # Default to grey
                cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)

    # Add text overlay
    add_text(frame, "Air Twister", (int(frame.shape[1]/2)-100, 50), color=(128, 0, 128), font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, font_scale=1.5, thickness=2)

    # Add white text box with cyan text saying "Points:    "
    box_position = (10, 10, 160, 40)
    cv2.rectangle(frame, (box_position[0] - 2, box_position[1] - 2), (box_position[2] + 2, box_position[3] + 2), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (box_position[0], box_position[1]), (box_position[2], box_position[3]), (255, 255, 255), cv2.FILLED)
    add_text(frame, f"Points: {points}", (20, 35), color=(255, 255, 0))

    # Add colored boxes
    for position, color in zip(box_positions, colors):
        x, y = position
        if color == (255, 0, 0):  # Blue box
            if check_requirements([[15, 17, 19, 21]], [position]):  # Right hand requirement
                alpha = 0.5  # Set transparency level to 0.5
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
        elif color == (0, 0, 255):  # Red box
            if check_requirements([[16, 18, 20, 22]], [position]):  # Left hand requirement
                alpha = 0.5
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
        elif color == (0, 255, 0):  # Yellow box
            if check_requirements([[27, 29, 31]], [position]):  # Right leg requirement
                alpha = 0.5
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
        elif color == (0, 255, 255):  # Green box
            if check_requirements([[28, 30, 32]], [position]):  # Left leg requirement
                alpha = 0.5
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
        else:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)

    # Check if all boxes are half transparent
    if all(color == (0, 0, 255, 0) for color in frame[:, :, 2]):
        points += 1  # Give the player a point
        refresh_box_positions()  # Refresh box positions

    # Add button to refresh box positions
    button_position = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 200, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 70,
                       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 20, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20)
    cv2.rectangle(frame, (button_position[0], button_position[1]), (button_position[2], button_position[3]), (0, 255, 0), cv2.FILLED)
    add_text(frame, "Click to refresh", (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 190, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 35), color=(255, 192, 203), font_scale=0.7, thickness=2)

    # Display the frame
    cv2.imshow("Pose Landmarks", frame)

    # Check if the player meets all requirements to get a point
    if refresh_boxes:
        refresh_boxes = False
        last_points = points
        if check_requirements([[16, 18, 20, 22], [15, 17, 19, 21], [28, 30, 32], [27, 29, 31]], box_positions):
            points += 1
            refresh_boxes = True

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()

