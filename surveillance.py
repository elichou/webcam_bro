import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from ultralytics import YOLO
import pandas as pd
import cv2
import asyncio
import telegram
from telegram import Bot
import time


# Load the pretrained model
model = YOLO("yolov8n.pt")

# Replace 'YOUR_TOKEN' with your actual Bot token received from BotFather
TOKEN = "7003877059:AAHuGD_HzyBrAtlL5Tc8_6Bh3PkZ68WSW6k"

# Replace 'YOUR_CHAT_ID' with your actual chat ID
CHAT_ID = '650813102'


async def send_telegram_message(image):
    """Sends a Telegram message with the detected image attached."""
    bot = Bot(TOKEN)
    async with bot:
        # Convert image from OpenCV to a file-like byte array to send as photo
        ret, buffer = cv2.imencode('.jpg', image)
        buffer = buffer.tobytes()
        await bot.send_photo(chat_id=CHAT_ID, photo=buffer, caption="Intruder detected!")

def get_person_coordinates(frame):
    """Extracts coordinates of 'person' from model predictions."""
    results = model(frame)
    person_boxes = []
    if results:
        for result in results:
            boxes = result.boxes.data.detach().cpu().numpy()
            # Retrieve the index for 'person' from names dictionary
            person_index = [k for k, v in result.names.items() if v == 'person'][0]  # Assuming 'person' is a key
            # Filter out boxes detected as 'person'
            for box in boxes:
                if int(box[5]) == person_index and box[4] > 0.5:
                    person_boxes.append([box[0], box[1], box[2], box[3]])
    return person_boxes

async def webcam_detect():
    cap = cv2.VideoCapture("http://192.168.1.17/live")
    email_sent = False  # Flag to control the email sending
    pause_time = 10  # Pause for 60 seconds after sending an email

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if not email_sent:
            person_coordinates = get_person_coordinates(frame)
            print(person_coordinates)
            if person_coordinates:
                print("Person detected")
                await send_telegram_message(frame) # Send the frame where the person was detected
                email_sent = True  # Set the flag to True after sending an email
                last_email_time = time.time()  # Record the time when the email was sent

        else:
            # Check if the pause time has elapsed
            if (time.time() - last_email_time) > pause_time:
                email_sent = False  # Reset the flag to start detecting again

        cv2.imshow('YOLOv8 Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # As the main function is asynchronous, we need to run it in an asyncio event loop
    asyncio.run(webcam_detect())