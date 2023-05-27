import tkinter as tk
import csv
from PIL import Image, ImageTk
import random
import cv2 


moves = [
    [0, -1], # move n. 0
    [0, -0.7], # move n. 1
    [0, -0.5], # move n. 2
    [0, -0.3], # move n. 3
    [0, 0], # move n. 4
    [0, 0.3], # move n. 5
    [0, 0.5], # move n. 6
    [0, 0.7], # move n. 7
    [0, 1], # move n. 8
    [0.3, -1], # move n. 9
    [0.3, -0.7], # move n. 10
    [0.3, -0.5], # move n. 11
    [0.3, -0.3], # move n. 12
    [0.3, 0], # move n. 13
    [0.3, 0.3], # move n. 14
    [0.3, 0.5], # move n. 15
    [0.3, 0.7], # move n. 16
    [0.3, 1], # move n. 17
    [0.5, -1], # move n. 18
    [0.5, -0.7], # move n. 19
    [0.5, -0.5], # move n. 20
    [0.5, -0.3], # move n. 21
    [0.5, 0], # move n. 22
    [0.5, 0.3], # move n. 23
    [0.5, 0.5], # move n. 24
    [0.5, 0.7], # move n. 25
    [0.5, 1], # move n. 26
    [0.7, -1], # move n. 27
    [0.7, -0.7], # move n. 28
    [0.7, -0.5], # move n. 29
    [0.7, -0.3], # move n. 30
    [0.7, 0], # move n. 31
    [0.7, 0.3], # move n. 32
    [0.7, 0.5], # move n. 33
    [0.7, 0.7], # move n. 34
    [0.7, 1], # move n. 35
    [1, -1], # move n. 36
    [1, -0.7], # move n. 37
    [1, -0.5], # move n. 38
    [1, -0.3], # move n. 39
    [1, 0], # move n. 40
    [1, 0.3], # move n. 41
    [1, 0.5], # move n. 42
    [1, 0.7], # move n. 43
    [1, 1], # move n. 44
]

video_path = "/home/anas/LosWeedsHermanos/Labelling/Normal_Labeling/firstVideo.mp4"
cvs_path = "/home/anas/LosWeedsHermanos/Labelling/Normal_Labeling/data.csv"
images_path = "/home/anas/LosWeedsHermanos/Labelling/Normal_Labeling/Images/"

class VideoPlayer:
    def __init__(self):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_name = ""
        self.selected_move_index = -1
        self.root = tk.Tk()
        self.root.title("Video Player")

        self.video_frame = tk.Frame(self.root)
        self.video_frame.grid(row=0, column=0)

        self.move_selection_frame = tk.Frame(self.root)
        self.move_selection_frame.grid(row=0, column=1)

        self.canvas = tk.Canvas(self.video_frame, width=600, height=400)
        self.canvas.pack()
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.create_move_buttons()

        self.display_video_frame()

        self.root.mainloop()

    def save_selected_move(self):
        with open(cvs_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.frame_name, self.selected_move_index])
        print(f"Move n. {self.selected_move_index} is selected and saved for frame {self.frame_name}.")

        frame_save_name = images_path + f"{self.frame_name}.png"
        cv2.imwrite(frame_save_name, self.current_frame)
        print(f"Frame saved as {frame_save_name}.")

    def move_clicked(self, move_index):
        self.selected_move_index = move_index
        self.save_selected_move()
        self.display_video_frame()

    def create_move_buttons(self):
        num_rows = 9
        num_columns = 5
        for row in range(num_rows):
            for column in range(num_columns):
                index = row * num_columns + column
                if index < len(moves):
                    move = moves[index]
                    button = tk.Button(self.move_selection_frame, text=f"Move: {move}", command=lambda idx=index: self.move_clicked(idx))
                    button.grid(row=row, column=column, padx=5, pady=5)

    def display_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.root.quit()
            return

        self.current_frame = frame.copy()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((600, 400))  # Adjust the size of the displayed frame as needed
        photo = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.image_item, image=photo)
        self.canvas.image = photo

        self.frame_name = f"frame_{random.randint(1, 1000000):06d}"

player = VideoPlayer()
