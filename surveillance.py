import cv2
import pandas as pd
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

# Load the pretrained YOLO model
model = YOLO('yolov8n.pt')

# Assuming `class_list` is provided and correctly maps indices to class names, including 'person'
class_list = model.names  # Update according to how class names are accessed in your setup

def send_email(image):
    """Sends an email with the detected image attached."""
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("your_email@gmail.com", "your_password")
    msg = MIMEMultipart()
    msg['From'] = "your_email@gmail.com"
    msg['To'] = "receiver_email@gmail.com"
    msg['Subject'] = "Intruder Alert"
    text = MIMEText("An intruder was detected. See the attached image.")
    msg.attach(text)
    img_data = cv2.imencode('.jpg', image)[1].tobytes()
    image = MIMEImage(img_data, name='intruder.jpg')
    msg.attach(image)
    server.send_message(msg)
    server.quit()
    print("Email sent with image!")

def get_person_coordinates(frame):
    """
    Extracts the coordinates of the person bounding boxes from the YOLO model predictions.
    """
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.detach().cpu()
    px = pd.DataFrame(a, columns=['x1', 'y1', 'x2', 'y2', 'conf', 'cls_id']).astype("float")

    list_corr = []
    for index, row in px.iterrows():
        if class_list[int(row['cls_id'])] == 'person' and row['conf'] > 0.5:
            list_corr.append([row['x1'], row['y1'], row['x2'], row['y2']])
    return list_corr

def webcam_detect():
    cap = cv2.VideoCapture("http://192.168.1.17/live")  # Ensure the correct stream URL
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect persons in the frame
        person_coordinates = get_person_coordinates(frame)
        if person_coordinates:
            send_email(frame)  # Send the frame where the person was detected

        cv2.imshow('YOLOv8 Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam_detect()
