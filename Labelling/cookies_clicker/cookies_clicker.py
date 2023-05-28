'''
1. open a image from directory /img
2. pop up a window with the image, resize to 25% of the original size
3. click a point in the image
4. save the image name & the coordinate of the point in a .csv file
5. repeat 1-4 until all images are done

Version 1.1 (2023-05-28): accept coninuous mouse press
'''

'''
    Too late, I've already labelled all the videos
'''

'''
    Cool, but did you think of opening the video directly with opencv and analize it frame by frame?

    import cv2
    cap = cv2.VideoCapture("./out.mp4")

    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break

    If I didnt misunderstand, this script wans a list of jpeg. In the snippet above, "frame" is the image and you can use
    it to label
'''
import os
import cv2
import csv

# Configuration
windows_scale = 2 # 50% windows size
image_directory = 'VID_20230517_121808'  # Path to the image directory
csv_file = 'VID_20230517_121808.csv'  # Name of the CSV file to store the image names and coordinates

# Global variable to store the button state
clicked_point = None
button_down = False

def mouse_callback(event, x, y, flags, param):
    global clicked_point, button_down
    clicked_point = (x * windows_scale, y * windows_scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        button_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        button_down = False

def get_next_img(image_file_index):
    image_path = os.path.join(image_directory, image_files[image_file_index])
    print(image_path)

    if image_file_index >= len(image_files) - 1:
        return None, None, None

    # Open the image
    image = cv2.imread(image_path)

    # Calculate the scaled dimensions
    scaled_width = int(image.shape[1] / windows_scale)
    scaled_height = int(image.shape[0] / windows_scale)

    # Resize the image
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))

    image_file_index = image_file_index + 1

    return scaled_image, image_files[image_file_index], image_file_index

def process_images():
    global image_files

    # Get a list of all image filenames in the directory
    image_files = [file for file in os.listdir(image_directory) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
    image_files.sort()  # Sort the image files in ascending order

    # Create or open the CSV file to write image names and coordinates
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'X', 'Y'])  # Write the header row

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)

        scaled_image, image_file, image_file_index = get_next_img(0)

        while True:
            cv2.imshow('Image', scaled_image)

            if button_down:
                # Save the image name and coordinates to the CSV file
                csv_writer.writerow([image_file, clicked_point[0], clicked_point[1]])
                scaled_image, image_file, image_file_index = get_next_img(image_file_index)
                if scaled_image is None:
                    break

            key = cv2.waitKey(1)
            if key == 27:  # Press Esc to exit
                break

    cv2.destroyAllWindows()
    print("Image processing completed. CSV file saved.")

# Run the image processing function
process_images()
