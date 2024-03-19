import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Open a webcam capture
cap = cv2.VideoCapture(0)

# Get the width and height of the video capture
width = int(cap.get(3))
height = int(cap.get(4))

# Flip the screen
flip = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    if flip:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Store thumb landmarks
            thumb_landmarks = [landmarks.landmark[i] for i in range(1, 5)]

            # Store index finger landmarks
            index_finger_landmarks = [landmarks.landmark[i] for i in range(5, 9)]

            # Get thumb CMC point
            thumb_cmc_point = (int(thumb_landmarks[0].x * width), int(thumb_landmarks[0].y * height))

            # Get index PMC point
            index_pmc_point = (int(index_finger_landmarks[0].x * width), int(index_finger_landmarks[0].y * height))

            # Get wrist point (assuming it is the first landmark)
            wrist_point = (int(landmarks.landmark[0].x * width), int(landmarks.landmark[0].y * height))

            # Get thumb tip and index MCP points
            thumb_tip = (int(landmarks.landmark[4].x * width), int(landmarks.landmark[4].y * height))
            index_mcp = (int(landmarks.landmark[5].x * width), int(landmarks.landmark[5].y * height))

            # Calculate the angles
            angle_9_4_5_6 = np.degrees(np.arctan2(thumb_landmarks[3].y - thumb_landmarks[0].y, thumb_landmarks[3].x - thumb_landmarks[0].x) -
                                       np.arctan2(index_finger_landmarks[3].y - index_finger_landmarks[0].y, index_finger_landmarks[3].x - index_finger_landmarks[0].x))
            angle_index_thumb = np.degrees(np.arctan2(thumb_tip[1] - index_mcp[1], thumb_tip[0] - index_mcp[0]))

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw lines for thumb with color change based on straightness and add visual numbers
            for i in range(len(thumb_landmarks) - 1):
                start_point = (int(thumb_landmarks[i].x * width), int(thumb_landmarks[i].y * height))
                end_point = (int(thumb_landmarks[i + 1].x * width), int(thumb_landmarks[i + 1].y * height))

                # Calculate the distance between consecutive points
                distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

                # Change line color based on distance
                color = (0, int(255 * (1 - distance / 200)), int(255 * (distance / 200)))

                # Draw the line
                cv2.line(frame, start_point, end_point, color, 2)

                # Add visual numbers
                number_position = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv2.putText(frame, str(i + 1), number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw lines for index finger with color change based on straightness and add visual numbers
            for i in range(len(index_finger_landmarks) - 1):
                start_point = (int(index_finger_landmarks[i].x * width), int(index_finger_landmarks[i].y * height))
                end_point = (int(index_finger_landmarks[i + 1].x * width), int(index_finger_landmarks[i + 1].y * height))

                # Calculate the distance between consecutive points
                distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

                # Change line color based on distance
                color = (0, int(255 * (1 - distance / 200)), int(255 * (distance / 200)))

                # Draw the line
                cv2.line(frame, start_point, end_point, color, 2)

                # Add visual numbers
                number_position = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv2.putText(frame, str(i + 1 + len(thumb_landmarks) - 1), number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw a green line connecting thumb CMC to index PMC and add a visual number
            cv2.line(frame, thumb_cmc_point, index_pmc_point, (0, 255, 0), 2)
            number_position = ((thumb_cmc_point[0] + index_pmc_point[0]) // 2, (thumb_cmc_point[1] + index_pmc_point[1]) // 2)
            cv2.putText(frame, str(i + 1 + len(thumb_landmarks) + len(index_finger_landmarks) - 2), number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw aquamarine circle around the hand
            radius = int(np.sqrt((thumb_cmc_point[0] - wrist_point[0])**2 + (thumb_cmc_point[1] - wrist_point[1])**2))
            cv2.circle(frame, wrist_point, radius, (127, 255, 212), 2, cv2.LINE_AA)

            # Display angles
            cv2.putText(frame, f"Angle 9-4-5-6: {angle_9_4_5_6:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Angle Index MCP to Thumb Tip: {angle_index_thumb:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2, cv2.LINE_AA)

            # Display "Gun Mode On" or "Gun Mode Off" based on the angle
            if 80 <= angle_index_thumb <= 120:
                cv2.putText(frame, "Gun Mode On", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Gun Mode Off", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame with landmarks
    cv2.imshow('Hand Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
