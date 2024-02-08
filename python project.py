import cv2
import os
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


class Facer:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("600x480")
        self.window.title("Face Recognition")
        self.f1 = Frame(self.window, width=600, height=500,bg="red")
        self.f1.place(x=0, y=0)
        l1=Label(self.f1,text="        Face Detection Lock      ",font=("Honk",15,"bold"),bg="yellow").place(x=175,y=175)
        l2=Label(self.f1,text="        Use the Button to get details      ",font=("Honk",15,"italic"),bg="yellow").place(x=150,y=200)
        

        # Button to start webcam feed
        self.start_button = Button(self.f1, text="   Start Webcam   ",font=("bold"),fg="black",bg="sky blue", command=self.start_webcam).place(x=230,y=240)

        
        # Load cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces

    def start_webcam(self):
        self.f2 = Frame(self.window, width=600, height=480)
        self.f2.place(x=0, y=50)

        self.label = Label(self.f2)
        self.label.place(x=0, y=0)

        # Face detection from webcam
        cap = cv2.VideoCapture(0)
        face_detected_once = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame")
                break

            faces_found = self.detect_faces(frame)

            for (x, y, w, h) in faces_found:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if not face_detected_once:
                file_to_open = 'details.txt'
                if os.path.exists(file_to_open):
                    with open(file_to_open, 'r') as f:
                        details = f.read()
                        print("The Details of the detected face:", details)
                face_detected_once = True

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)

            # Update the label with the new image
            self.label.configure(image=img)
            self.label.img = img

            cv2.imshow('Detected Faces', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


v = Facer()
v.window.mainloop()







