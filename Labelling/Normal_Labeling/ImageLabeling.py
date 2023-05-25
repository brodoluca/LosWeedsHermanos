import cv2
import csv

# Video input path
video_path = '/home/anas/Mechatronics_Project/LabelingImages/firstVideo.mp4'

# Output CSV file path
csv_path = '/home/anas/Mechatronics_Project/LabelingImages/data.csv'

# Open the video file
video = cv2.VideoCapture(video_path)

# Create a CSV file for storing frame names and labels
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frame Name', 'Label'])

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Wait for user input and get the label
        label = input("Enter label for this frame: ")

        # Get the current frame index
        frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        # Generate a unique frame name
        frame_name = f'frame_{frame_index}.jpg'

        # Save the frame name and label to the CSV file
        writer.writerow([frame_name, label])

        # Exit if 'q' is pressed
        if label == 'q':
            break

        # Press any key to proceed to the next frame
        cv2.waitKey(0)

    # Release the video and close CSV file
    video.release()
    cv2.destroyAllWindows()