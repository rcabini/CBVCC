import cv2
import os, sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#-------------------------------------------------------------------------

WINDOW_SIZE = (50, 50, 20)

#---------------------------------------------------------------------------

def read_video(path_video):
    #leggo il video
    video_reader = cv2.VideoCapture(path_video)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video = []
    while True:
        ret, frame = video_reader.read()
        if frame is None: break
        else: video.append(frame) #RGB
    video = np.array(video)
    return video, fps
    
#---------------------------------------------------------------------------

class VideoPlayer:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Video Player")
        
        self.base = video_path.split('/')[-1].split('.avi')[0]
        
        self.video_array, self.fps = read_video(video_path)
        self.total_frames = self.video_array.shape[0]

        self.frame_index = 0

        self.video_width = self.video_array.shape[2]
        self.video_height = self.video_array.shape[1]

        # Impostare una dimensione fissa per la visualizzazione del video
        if self.video_width>=self.video_height:
            self.canvas_height = 580
            self.canvas_width = int(self.video_width * self.canvas_height / self.video_height)
        else:
            self.canvas_width = 580
            self.canvas_height = int(self.video_height * self.canvas_width / self.video_width)

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.scale = tk.Scale(root, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.update_frame)
        self.scale.pack(fill=tk.X)

        self.update_frame(0)

        self.current_class = None

    def update_frame(self, frame_index):
        self.frame_index = int(frame_index)
        frame = cv2.cvtColor(self.video_array[self.frame_index].copy(), cv2.COLOR_BGR2RGB)
        # Ridimensionare l'immagine al formato desiderato
        frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("850x850")

    # Controlla se è stato passato un file_path dalla linea di comando
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Altrimenti, usa l'interfaccia grafica per selezionare il file
        file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", "*.mp4"), ("Video files", "*.avi")])
    if file_path:
        player = VideoPlayer(root, file_path)
        root.mainloop()
