import cv2
import mediapipe as mp
import random
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Get the screen resolution
screen_width = 800  # Set your desired screen width
screen_height = 450  # Set your desired screen height

# Open a webcam capture with the specified resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, screen_width)
cap.set(4, screen_height)

# Initialize timer and winning player variables
start_time = time.time()
countdown_seconds = 60  # 1 minute
winning_player = None

# Function to flip the frame horizontally
def flip_frame(frame):
    return cv2.flip(frame, 1)

# List of fruits and their positions
fruits = []
fruit_size = 50
fruit_speed = 5
last_fruit_spawn_time = time.time()

# Special fruit constants
SPECIAL_FRUIT_INTERVAL_CIRCLES = 8
SPECIAL_FRUIT_INTERVAL_SQUARE = 4
SPECIAL_FRUIT_INTERVAL_RED_CIRCLE = 5
SPECIAL_FRUIT_INTERVAL_BLACK_CIRCLE = 6  # Adjust as needed
SPECIAL_FRUIT_INTERVAL_GOLD_SQUARE = 10
SPECIAL_FRUIT_SIZE = 20
SPECIAL_FRUIT_SPEED = 5
MAX_CIRCLE_COUNT = 10000  # Adjust as needed

# Initialize scores for Player Blue and Player Green
score_blue = 0
score_green = 0
blue_circles = 0
global black_circle_count
black_circle_count = 0

# Function to create a random fruit at the top of the screen
def create_fruit(circle_count):
    global black_circle_count, blue_circles
    x = random.randint(SPECIAL_FRUIT_SIZE, screen_width - SPECIAL_FRUIT_SIZE)
    y = -SPECIAL_FRUIT_SIZE  # Start fruits at the top of the screen
    is_special_fruit = False

    if circle_count % SPECIAL_FRUIT_INTERVAL_CIRCLES == 0:
        if circle_count % SPECIAL_FRUIT_INTERVAL_SQUARE == 0:
            return x, y, 'square'
        else:
            return x, y, 'circle'

    if circle_count % SPECIAL_FRUIT_INTERVAL_RED_CIRCLE == 0:
        return x, y, 'red_circle'

    if circle_count % SPECIAL_FRUIT_INTERVAL_BLACK_CIRCLE == 0:
        return x, y, 'black_circle'

    if circle_count % 5 == 0:
        blue_circles += 1
        return x, y, 'blue_circle'

    if circle_count % SPECIAL_FRUIT_INTERVAL_GOLD_SQUARE == 0:
        return x, y, 'gold_square'

    return x, y, 'circle'

# Function to create a special fruit at the top of the screen
def create_special_fruit(circle_count):
    x = random.randint(SPECIAL_FRUIT_SIZE, screen_width - SPECIAL_FRUIT_SIZE)
    y = -SPECIAL_FRUIT_SIZE
    return x, y, 'special_fruit'

# Function to determine the player's color based on finger tip position
def get_player_color(x, y):
    if x < screen_width // 2:
        return (255, 0, 0)  # Player Blue
    else:
        return (0, 255, 0)  # Player Green

# Create initial fruits
circle_count = 0
for _ in range(5):
    fruits.append(create_fruit(circle_count))
    circle_count += 1

# Create the window with normal properties
cv2.namedWindow('Fruit Ninja 1v1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Fruit Ninja 1v1', cv2.WINDOW_NORMAL)

# Set the window to fullscreen mode
cv2.setWindowProperty('Fruit Ninja 1v1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Variables for start button
start_button_text = "Pinch to Start"
start_button_x = screen_width // 2 - 130
start_button_y = screen_height // 2 - 25
start_button_width = 260
start_button_height = 50
start_button_color = (0, 255, 0)  # Green color

# Flag to indicate if the game is in the start screen
in_start_screen = True

# Timer and winning player text properties
timer_font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
timer_font_scale = 1
timer_font_thickness = 2
timer_text_color = (255, 255, 255)
timer_border_color = (255, 255, 0)
timer_position = (10, screen_height - 5)
winning_text_color = (100, 243, 125)
winning_text_border_color = (0, 0, 0)
winning_text_position = (10, screen_height // 2)

# Initialize timer and winning player variables
start_time = None
countdown_seconds = 60  # 1 minute
winning_player = None


# Start screen loop
while in_start_screen:
    ret, frame = cap.read()
    if not ret:
        continue

    
    # Add a vertical stationary brown line down the middle of the screen
    cv2.line(frame, (screen_width // 2, 0), (screen_width // 2, screen_height), (87, 41, 206), 10)  # Brown color

    # Draw the start button with inverted text
    cv2.rectangle(frame, (start_button_x, start_button_y),
                  (start_button_x + start_button_width, start_button_y + start_button_height), start_button_color, -1)
    cv2.putText(frame, start_button_text, (start_button_x + 15, start_button_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Flip the frame horizontally to correct the reversed display
    # frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get the position of the pointer (index finger tip)
            x = int(landmarks.landmark[8].x * screen_width)
            y = int(landmarks.landmark[8].y * screen_height)

            # Check if the player hovers over the start button
            if (start_button_x < x < start_button_x + start_button_width) and (start_button_y < y < start_button_y + start_button_height):
                # If the player pinches index finger and thumb finger together, start the game
                thumb_tip_x = int(landmarks.landmark[4].x * screen_width)
                thumb_tip_y = int(landmarks.landmark[4].y * screen_height)
                index_tip_x = int(landmarks.landmark[8].x * screen_width)
                index_tip_y = int(landmarks.landmark[8].y * screen_height)

                thumb_index_distance = ((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2) ** 0.5

                if thumb_index_distance < 30:  # Adjust the distance threshold as needed
                    in_start_screen = False  # Start the game
                    start_time = time.time()  # Start the game timer

    # Display the frame with start button
    cv2.imshow('Fruit Ninja 1v1', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Initialize pointer parameters
pointer_color = (218, 247, 166)  # Cyan color for the pointer
trail_length = 10  # Number of points in the trail
trail_points = []

# Set the window to fullscreen mode before entering the game loop
cv2.setWindowProperty('Fruit Ninja 1v1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Fruit Ninja 1v1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Game loop
while not in_start_screen:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flips the frame for the camera view
    frame = flip_frame(frame)

    # Calculate the remaining time on the countdown timer
    elapsed_time = time.time() - start_time
    remaining_time = max(countdown_seconds - int(elapsed_time), 0)

    # Create a beige-tan colored text box with navy blue background for the timer
    timer_text = f"Time: {remaining_time} seconds"
    timer_text_size, _ = cv2.getTextSize(timer_text, timer_font, timer_font_scale, timer_font_thickness)
    timer_box_width = timer_text_size[0] + 20
    timer_box_height = timer_text_size[1] + 0
    cv2.rectangle(frame, (screen_width - timer_box_width - 10, screen_height - timer_box_height - 20),
                  (screen_width - 10, screen_height - 10), (0, 0, 128), -1)
    cv2.putText(frame, timer_text, (screen_width - timer_box_width + 5, screen_height - timer_box_height + 5),
                timer_font, timer_font_scale, timer_text_color, timer_font_thickness)

    # Add a vertical stationary brown line down the middle of the screen
    cv2.line(frame, (screen_width // 2, 0), (screen_width // 2, screen_height), (139, 69, 19), 10)  # Brown color

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    


    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get the tip of the index finger (landmark 8)
            index_finger_tip = landmarks.landmark[8]
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])

            # Add the finger tip position to the trail
            trail_points.append((x, y))

            # Keep the trail length limited
            if len(trail_points) > trail_length:
                trail_points.pop(0)

            # Draw the trail
            for i, point in enumerate(trail_points):
                alpha = (i + 1) / float(len(trail_points))  # Vary alpha for fading effect
                radius = int(10 * alpha)  # Vary the radius for the fading effect
                cv2.circle(frame, point, radius, pointer_color, -1)

            # Get the position of the pointer (index finger tip)
            x = int(landmarks.landmark[8].x * screen_width)
            y = int(landmarks.landmark[8].y * screen_height)

            # Determine the player's color
            player_color = get_player_color(x, y)

            # Check for fruit slicing (click) by the corresponding player
            for i, (fruit_x, fruit_y, fruit_type) in enumerate(fruits):
                if abs(x - fruit_x) < SPECIAL_FRUIT_SIZE and abs(y - fruit_y) < SPECIAL_FRUIT_SIZE:
                    # Remove sliced fruit, update the player's score, and create a new one
                    fruits[i] = create_fruit(circle_count)
                    circle_count += 1

                # Get the position of the pointer (index finger tip)
                x = int(landmarks.landmark[8].x * screen_width)
                y = int(landmarks.landmark[8].y * screen_height)

                # Determine the player's color
                player_color = get_player_color(x, y)

                # Check for fruit slicing (click) by the corresponding player
                for i, (fruit_x, fruit_y, fruit_type) in enumerate(fruits):
                    if abs(x - fruit_x) < SPECIAL_FRUIT_SIZE and abs(y - fruit_y) < SPECIAL_FRUIT_SIZE:
                        # Remove sliced fruit, update the player's score, and create a new one
                        fruits[i] = create_fruit(circle_count)
                        circle_count += 1

                        if player_color == (255, 0, 0):  # Player Blue
                            if fruit_type == 'circle':
                                score_blue += 1
                                blue_circles += 1
                            elif fruit_type == 'square':
                                score_blue += 4  # Special square gives 4 points
                            elif fruit_type == 'black_circle':
                                score_blue -= 2  # Black circle subtracts 2 points
                        else:  # Player Green
                            if fruit_type == 'circle':
                                score_green += 1
                            elif fruit_type == 'square':
                                score_green += 4  # Special square gives 4 points
                            elif fruit_type == 'black_circle':
                                score_green -= 2  # Black circle subtracts 2 points


    current_time = time.time()
    if current_time - last_fruit_spawn_time >= 1:
        fruits.append(create_fruit(circle_count))
        last_fruit_spawn_time = current_time

    # Update fruit positions
    circle_count += 1
    if circle_count >= MAX_CIRCLE_COUNT:
        circle_count = 0  # Reset circle_count to avoid overflow
    for i, (fruit_x, fruit_y, fruit_type) in enumerate(fruits):
        fruit_y += SPECIAL_FRUIT_SPEED  # Make fruits move downwards
        if fruit_y > screen_height + SPECIAL_FRUIT_SIZE:
            # Remove fruits that are out of the screen
            fruits[i] = create_fruit(circle_count)
            circle_count += 1

        else:
            fruits[i] = (fruit_x, fruit_y, fruit_type)

            # Draw fruits
            if fruit_type == 'circle':
                cv2.circle(frame, (fruit_x, fruit_y), SPECIAL_FRUIT_SIZE, (0, 0, 255), -1)  # Blue circle
            elif fruit_type == 'square':
                cv2.rectangle(frame, (fruit_x - SPECIAL_FRUIT_SIZE, fruit_y - SPECIAL_FRUIT_SIZE),
                              (fruit_x + SPECIAL_FRUIT_SIZE, fruit_y + SPECIAL_FRUIT_SIZE), (0, 223, 255), -1)  # Gold square
            elif fruit_type == 'black_circle':
                cv2.circle(frame, (fruit_x, fruit_y), SPECIAL_FRUIT_SIZE, (0, 0, 0), -1)  # Black circle

    # Check if it's time to create special fruits
    if current_time - last_fruit_spawn_time >= 1:
        special_fruit = create_special_fruit(circle_count)
        if special_fruit:
            fruits.append(special_fruit)
        last_fruit_spawn_time = current_time

    # Display the scores for Player Blue and Player Green
    cv2.putText(frame, f'Player Blue Score: {score_blue}', (20, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Player Green Score: {score_green}', (screen_width // 2 + 20, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the scores for Player Blue and Player Green
    cv2.putText(frame, f'Player Blue Score: {score_blue}', (20, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Player Green Score: {score_green}', (screen_width // 2 + 20, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if the timer has reached 0
    if start_time is not None and current_time - start_time >= countdown_seconds:
        winning_player = "Player Blue has won!" if score_blue > score_green else "Player Green has won!"
        in_start_screen = True  # Return to the start screen

    # Display the frame with fruits, pointers, and scores
    cv2.imshow('Shape Popper Ninja 1v1', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

    # Display the frame with landmarks, the pointer, and fruits
    cv2.imshow('Shape Popper Ninja 1v1', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break
# Display the winning player text
if winning_player is not None:
    # frame = cv2.flip(frame, 1)  # Flip frame horizontally
    cv2.rectangle(frame, (0, screen_height // 4), (screen_width, 3 * screen_height // 4), winning_text_border_color, -1)
    cv2.putText(frame, winning_player, (10, screen_height // 2), timer_font, timer_font_scale, winning_text_color, timer_font_thickness)
    cv2.imshow('Fruit Ninja 1v1', frame)
    cv2.waitKey(50000)  # Display the winning text for 5 seconds


# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()