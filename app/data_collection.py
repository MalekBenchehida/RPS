
import cv2
import os
import time

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create directories to store the images
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(os.path.join(data_dir, "rock")):
    os.makedirs(os.path.join(data_dir, "rock"))
if not os.path.exists(os.path.join(data_dir, "paper")):
    os.makedirs(os.path.join(data_dir, "paper"))
if not os.path.exists(os.path.join(data_dir, "scissors")):
    os.makedirs(os.path.join(data_dir, "scissors"))

# Set the camera
cam = cv2.VideoCapture(0)

# Set the image size
img_width, img_height = 300, 300

# Set the number of images to capture
num_images = 100

# Set the gesture to capture
gesture = "scissors"  # Change this to "paper" or "scissors"

# Start capturing images
count = 0
while count < num_images:
    # Read a frame from the camera
    ret, frame = cam.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Show the frame
    cv2.imshow("Capture", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the 'c' key is pressed, capture the image
    if key == ord('c'):
        # Save the image
        image_path = os.path.join(data_dir, gesture, f"{gesture}_{count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Captured {image_path}")
        count += 1

    # If the 'q' key is pressed, quit
    if key == ord('q'):
        break

# Release the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
