
import tensorflow as tf
import cv2
import numpy as np
import os
import random

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load the trained model
model_path = os.path.join(project_root, 'model', 'rps_model.h5')
model = tf.keras.models.load_model(model_path)

# Define the labels
labels = ['rock', 'paper', 'scissors']

# Set the image size
img_width, img_height = 300, 300

# Start the webcam
cap = cv2.VideoCapture(0)

# Game state
round_active = False
user_move = ''
computer_move = ''
winner = ''

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI)
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    roi = frame[100:400, 100:400]

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        round_active = True

    if round_active:
        # Preprocess the ROI
        img = cv2.resize(roi, (img_width, img_height))
        img = np.expand_dims(img, axis=0)
        img = img / 255.

        # Make a prediction
        prediction = model.predict(img)
        user_move_index = np.argmax(prediction)
        user_move = labels[user_move_index]

        # Get the computer's move
        computer_move = random.choice(labels)

        # Determine the winner
        if user_move == computer_move:
            winner = 'Tie'
        elif (user_move == 'rock' and computer_move == 'scissors') or \
             (user_move == 'scissors' and computer_move == 'paper') or \
             (user_move == 'paper' and computer_move == 'rock'):
            winner = 'You win!'
        else:
            winner = 'Computer wins!'
        
        round_active = False

    # Display the result
    font = cv2.FONT_HERSHEY_SIMPLEX
    if winner:
        cv2.putText(frame, f'Your move: {user_move}', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Computer move: {computer_move}', (50, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Winner: {winner}', (50, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Press SPACE to play', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # Show the frame
    cv2.imshow('Rock Paper Scissors', frame)


# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

