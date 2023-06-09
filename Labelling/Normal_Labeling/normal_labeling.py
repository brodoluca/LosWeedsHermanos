import tkinter as tk
import csv
from PIL import Image, ImageTk
import random
import cv2 


moves = [
    [-1, 0], # move n. 0
    [-1, 0.2], # move n. 1
    [-1, 0.4], # move n. 2
    [-1, 0.6], # move n. 3
    [-1, 0.8], # move n. 4
    [-1, 1], # move n. 5
    [-0.8, 0], # move n. 6
    [-0.8, 0.2], # move n. 7
    [-0.8, 0.4], # move n. 8
    [-0.8, 0.6], # move n. 9
    [-0.8, 0.8], # move n. 10
    [-0.8, 1], # move n. 11
    [-0.6, 0], # move n. 12
    [-0.6, 0.2], # move n. 13
    [-0.6, 0.4], # move n. 14
    [-0.6, 0.6], # move n. 15
    [-0.6, 0.8], # move n. 16
    [-0.6, 1], # move n. 17
    [-0.4, 0], # move n. 18
    [-0.4, 0.2], # move n. 19
    [-0.4, 0.4], # move n. 20
    [-0.4, 0.6], # move n. 21
    [-0.4, 0.8], # move n. 22
    [-0.4, 1], # move n. 23
    [-0.2, 0], # move n. 24
    [-0.2, 0.2], # move n. 25
    [-0.2, 0.4], # move n. 26
    [-0.2, 0.6], # move n. 27
    [-0.2, 0.8], # move n. 28
    [-0.2, 1], # move n. 29
    [0, 0], # move n. 30
    [0, 0.2], # move n. 31
    [0, 0.4], # move n. 32
    [0, 0.6], # move n. 33
    [0, 0.8], # move n. 34
    [0, 1], # move n. 35
    [0.2, 0], # move n. 36
    [0.2, 0.2], # move n. 37
    [0.2, 0.4], # move n. 38
    [0.2, 0.6], # move n. 39
    [0.2, 0.8], # move n. 40
    [0.2, 1], # move n. 41
    [0.4, 0], # move n. 42
    [0.4, 0.2], # move n. 43
    [0.4, 0.4], # move n. 44
    [0.4, 0.6], # move n. 45
    [0.4, 0.8], # move n. 46
    [0.4, 1], # move n. 47
    [0.6, 0], # move n. 48
    [0.6, 0.2], # move n. 49
    [0.6, 0.4], # move n. 50
    [0.6, 0.6], # move n. 51
    [0.6, 0.8], # move n. 52
    [0.6, 1], # move n. 53
    [0.8, 0], # move n. 54
    [0.8, 0.2], # move n. 55
    [0.8, 0.4], # move n. 56
    [0.8, 0.6], # move n. 57
    [0.8, 0.8], # move n. 58
    [0.8, 1], # move n. 59
    [1, 0], # move n. 60
    [1, 0.2], # move n. 61
    [1, 0.4], # move n. 62
    [1, 0.6], # move n. 63
    [1, 0.8], # move n. 64
    [1, 1], # move n. 65
]

video_path = "/home/anas/LosWeedsHermanos/Labelling/Normal_Labeling/firstVideo.mp4"
cvs_path = "/home/anas/LosWeedsHermanos/Labelling/Normal_Labeling/data.csv"
images_path = "/home/anas/LosWeedsHermanos/Labelling/Normal_Labeling/Images/"

class VideoPlayer:
    def __init__(self):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_name = ""
        self.selected_move = [999,999]
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
            writer.writerow([self.frame_name, self.selected_move])
        print(f"Move n. {self.selected_move} is selected and saved for frame {self.frame_name}.")

        frame_save_name = images_path + f"{self.frame_name}.png"
        cv2.imwrite(frame_save_name, self.current_frame)
        print(f"Frame saved as {frame_save_name}.")

    def move_clicked(self, move):
        self.selected_move = move
        self.save_selected_move()
        self.display_video_frame()

    def create_move_buttons(self):
        num_rows = 11
        num_columns = 6
        for row in range(num_rows):
            for column in range(num_columns):
                index = row * num_columns + column
                if index < len(moves):
                    move = moves[index]
                    button = tk.Button(self.move_selection_frame, text=f"Move: {move}", command=lambda mv=move: self.move_clicked(mv))
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
