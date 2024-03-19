import cv2
import mediapipe as mp
import numpy as np

# Create a white background image
background_height = 768
background_width = 1024

cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=0)
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Set resolution width
cap.set(4, 480)  # Set resolution height

# Initialize counters and previous hand positions
curl_count = 0
prev_left_hand_y = 0
prev_right_hand_y = 0
left_box_touched = False
right_box_touched = False

# Initialize transparency for red/green boxes and hand circles
box_alpha = 0
hand_circle_alpha = 0.5  # Set the alpha value for transparency

# Load barbell images
left_barbell_img = cv2.imread("C:\\Users\\24moralesa\\Pictures\\Screenshots\\Screenshot 2024-01-11 134846.png", cv2.IMREAD_UNCHANGED)
right_barbell_img = cv2.imread("C:\\Users\\24moralesa\\Pictures\\Screenshots\\Screenshot 2024-01-11 134846.png", cv2.IMREAD_UNCHANGED)

# Resize barbell images to match the size where they are overlaid
left_barbell_img = cv2.resize(left_barbell_img, (160, 160))
right_barbell_img = cv2.resize(right_barbell_img, (160, 160))

# Initialize thickness for pose estimation connections
pose_connections_thickness = 1

# Define colors
black = [255, 105, 97]
light_green = [24, 70, 63]
dark_red = [139, 0, 0]
light_blue = [150, 75, 0]

# Function to scale and adjust pose landmarks
def scale_pose_landmarks(landmarks, img_width, img_height, background_width, background_height):
    # Find the original pose center (in x-direction)
    original_center_x = sum(landmark.x for landmark in landmarks.landmark) / len(landmarks.landmark)
   
    # Calculate the scaling factor based on height
    left_hip_y = landmarks.landmark[23].y  # Assuming 23 is the left hip index
    right_hip_y = landmarks.landmark[24].y  # Assuming 24 is the right hip index
    hips = (left_hip_y + right_hip_y) / 2  # Average
    left_knee_y = landmarks.landmark[25].y  # Assuming 23 is the left hip index
    right_knee_y = landmarks.landmark[26].y  # Assuming 24 is the right hip index
    knees = (left_knee_y + right_knee_y) / 2

    min_y = (hips + knees) / 2.2
    max_y = min(landmark.y for landmark in landmarks.landmark)

    pose_height = max_y - min_y
    scale_factor = (background_height * 0.9) / (pose_height * img_height)
   
    # Scale landmarks
    for landmark in landmarks.landmark:
        landmark.x *= img_width * scale_factor / background_width
        landmark.x = 1.0 - landmark.x

        landmark.y = (landmark.y - min_y) * img_height * scale_factor / background_height + 0.05
        landmark.y = 1.0 - landmark.y

    # Find the new scaled pose center (in x-direction)
    scaled_center_x = sum(landmark.x for landmark in landmarks.landmark) / len(landmarks.landmark)

    # Calculate translation in x to align centers
    translation_x = original_center_x - scaled_center_x

    # Apply translation and clamping
    for id, landmark in enumerate(landmarks.landmark):
        # Apply translation in x
        landmark.x += translation_x

        # Clamp x values to be within 0 and 1
        landmark.x = max(0, min(landmark.x, 1))

        # For landmarks 19 and 20, also clamp y values between 0 and 1
        if id in [19, 20]:
            landmark.y = max(0, min(landmark.y, 1))

    return landmarks

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        frame.flags.writeable = True

        display_image = np.ones((background_height, background_width, 4), dtype=np.uint8) * 255  # Use 4 channels for RGBA

        # Add red boxes on the top and bottom
        top_box_height = int(background_height * 0.2)
        bottom_box_height = int(background_height * 0.2)

        left_box_color = black + [box_alpha] if not left_box_touched else light_green + [box_alpha]
        right_box_color = black + [box_alpha] if not right_box_touched else light_green + [box_alpha]

        display_image[:top_box_height, :, :] = left_box_color  # Set top box to black or light green
        display_image[-bottom_box_height:, :, :] = right_box_color  # Set bottom box to black or light green

        if pose_results.pose_landmarks:
            scaled_landmarks = scale_pose_landmarks(pose_results.pose_landmarks, frame.shape[1], frame.shape[0], background_width, background_height)

            # Convert RGBA to BGR
            display_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGBA2BGR)

            mp_drawing.draw_landmarks(
                display_bgr,
                scaled_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=dark_red, thickness=2),
                mp_drawing.DrawingSpec(color=light_blue, thickness=pose_connections_thickness, circle_radius=2)
            )

            # Convert BGR back to RGBA
            display_image = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGBA)

            for id, landmark in enumerate(scaled_landmarks.landmark):
                if id == 19 or id == 20:
                    x = int(landmark.x * background_width)
                    y = int(landmark.y * background_height)

                    # Check for touch in the left box
                    if id == 19 and top_box_height >= y >= 0:
                        left_box_touched = True
                        left_barbell_x = x - left_barbell_img.shape[1] // 2
                        left_barbell_y = y - left_barbell_img.shape[0] // 2
                        left_barbell_end_x = left_barbell_x + left_barbell_img.shape[1]
                        left_barbell_end_y = left_barbell_y + left_barbell_img.shape[0]
                        
                        # Check if the barbell region is within the display image
                        if 0 <= left_barbell_y < background_height and 0 <= left_barbell_end_y < background_height \
                                and 0 <= left_barbell_x < background_width and 0 <= left_barbell_end_x < background_width:
                            display_image[left_barbell_y:left_barbell_end_y, left_barbell_x:left_barbell_end_x] = left_barbell_img

                    # Check for touch in the right box
                    if id == 20 and background_height >= y >= background_height - bottom_box_height:
                        right_box_touched = True
                        right_barbell_x = x - right_barbell_img.shape[1] // 2
                        right_barbell_y = y - right_barbell_img.shape[0] // 2
                        right_barbell_end_x = right_barbell_x + right_barbell_img.shape[1]
                        right_barbell_end_y = right_barbell_y + right_barbell_img.shape[0]

                        # Check if the barbell region is within the display image
                        if 0 <= right_barbell_y < background_height and 0 <= right_barbell_end_y < background_height \
                                and 0 <= right_barbell_x < background_width and 0 <= right_barbell_end_x < background_width:
                            display_image[right_barbell_y:right_barbell_end_y, right_barbell_x:right_barbell_end_x] = right_barbell_img

                    cv2.circle(display_image, (x, y), 35, light_green + [hand_circle_alpha] if id == 19 else black + [hand_circle_alpha], -1)  # Set hand circles with transparency
                    coords_text = f"({x}, {y})"
                    cv2.putText(display_image, coords_text, (x + 60, y), cv2.FONT_HERSHEY_SIMPLEX, 1, light_green if id == 19 else black, 2)

                    # Check for upward movement of the hands
                    if id == 19 and prev_left_hand_y > 700 and y <= 200:
                        curl_count += 1
                    elif id == 20 and prev_right_hand_y > 700 and y <= 200:
                        curl_count += 1

                    if id == 19:
                        prev_left_hand_y = y
                    elif id == 20:
                        prev_right_hand_y = y
            
            if left_box_touched and right_box_touched:
                left_box_touched = False
                right_box_touched = False
                curl_count += 1
                pose_connections_thickness += 1  # Increase thickness for every curl

        # Check if both boxes are touched and reset them
        if left_box_touched and right_box_touched:
            left_box_touched = False
            right_box_touched = False
            curl_count += 1

        # Display the curl counter in the top right corner
        cv2.putText(display_image, f'Curls: {curl_count}', (background_width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, light_blue, 3)

        cv2.imshow('Webcam', display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()