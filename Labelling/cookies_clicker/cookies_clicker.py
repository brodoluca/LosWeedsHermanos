'''
1. open a image from directory /img
2. pop up a window with the image, resize to 25% of the original size
3. click a point in the image
4. save the image name & the coordinate of the point in a .csv file
5. repeat 1-4 until all images are done
'''
import os
import cv2
import csv

windows_scale = 2 # 50% windows size

def click_event(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x * windows_scale, y * windows_scale)

def process_images():
    global clicked_point

    image_directory = 'img'  # Path to the image directory
    csv_file = 'image_coordinates.csv'  # Name of the CSV file to store the image names and coordinates

    # Get a list of all image filenames in the directory
    image_files = [file for file in os.listdir(image_directory) if file.endswith('.jpeg') or file.endswith('.png')]
    image_files.sort()  # Sort the image files in ascending order

    # Create or open the CSV file to write image names and coordinates
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'X', 'Y'])  # Write the header row

        # Iterate through each image in the directory
        for image_file in image_files:
            image_path = os.path.join(image_directory, image_file)

            # Open the image
            image = cv2.imread(image_path)

            # Calculate the scaled dimensions
            scaled_width = int(image.shape[1] / windows_scale)
            scaled_height = int(image.shape[0] / windows_scale)

            # Resize the image
            scaled_image = cv2.resize(image, (scaled_width, scaled_height))

            cv2.imshow('Image', scaled_image)
            clicked_point = None
            cv2.setMouseCallback('Image', click_event)

            while True:
                if clicked_point is not None:
                    # Save the image name and coordinates to the CSV file
                    csv_writer.writerow([image_file, clicked_point[0], clicked_point[1]])
                    clicked_point = None
                    break

                key = cv2.waitKey(1)
                if key == 27:  # Press Esc to exit
                    break

    cv2.destroyAllWindows()
    print("Image processing completed. CSV file saved.")

# Run the image processing function
process_images()
