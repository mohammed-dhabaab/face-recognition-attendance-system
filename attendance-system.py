import os
import cv2
import numpy as np
from datetime import datetime
import datetime
from tensorflow.keras.models import load_model
import openpyxl

# Define the path to the dataset
data_path = 'data'

# Get the list of classes (folders) in the dataset
classes = ['Abdulaziz', 'Abdulrahman Khalid', 'Abdulrhman Younis', 'Afnan', 'Ahmed', 'Aishah', 'Anas', 'Hala', 'Hassan', 'Hazem', 'Mariyyah', 'Marwah', 'Muhammad Alhudari', 'Muhammed Dhabaab', 'Omar', 'Shahad Alnami', 'Shoog', 'Snd', 'Tariq']

# Create a dictionary to store the class and label mapping
class_labels = {}
for i, c in enumerate(classes):
    class_labels[c] = i



# Create a new workbook object (cxcel)
attendance_workbook = openpyxl.Workbook()
tody_date = str(datetime.datetime.now().date())
root_directory = os.getcwd()
attendance_folder = f"{root_directory}\\attendance"
attendance_file_name = f"{attendance_folder}\\attenders\\attendance_{tody_date}.xlsx"

# Select the active worksheet
attendance_sheet = attendance_workbook.active

# Add some data to the worksheet
attendance_sheet["A1"] = "Photo"
attendance_sheet["B1"] = "Name"
attendance_sheet["C1"] = "Time"
excel_row = 2
handle_Repetition = {}


# Load the best model
model = load_model('model.h5')

# Initialize the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a image from the video capture device
    ret, image = cap.read()

    image_copy = image

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the image
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Make a prediction on the face using the loaded model
        pred_label = np.argmax(model.predict(face), axis=1)[0]
        pred_name = classes[pred_label]
        print(pred_label)
        print(pred_name)
        # Draw a rectangle around the detected face with the predicted label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the attendance information in an Excel qfile
        if pred_name not in handle_Repetition:
            # Get the current time
            time = datetime.datetime.now().time()
            attender_photo_path = f'{attendance_folder}\\attenders_photos\\attenders_photos_{tody_date}\\{pred_name}.jpg'
            cv2.imwrite(attender_photo_path, image_copy)
            print(attender_photo_path)
            attendance_sheet["A" + str(excel_row)] = attender_photo_path
            attendance_sheet["B" + str(excel_row)] = pred_name
            attendance_sheet["C" + str(excel_row)] = time

            # handle_Repetition[str(class_name[2:])] += 1
            handle_Repetition[pred_name] = 1
            excel_row += 1

    # Display the image
    cv2.imshow('Face Recognition Attendance System', image)

    # Check if the 'q' or 'Esc' key is pressed to quit the program
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

attendance_workbook.save(attendance_file_name)

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from datetime import datetime
import datetime
import tensorflow as tf
import openpyxl

print(f"opencv-python=={cv2.__version__}")
print(f"numpy=={np.__version__}")
print(f"tensorflow=={tf.__version__}")
print(f"openpyxl=={openpyxl.__version__}")