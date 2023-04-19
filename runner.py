import cv2
from ultralytics import YOLO
import pyautogui

# Load the YOLOv8 model
model = YOLO('evioModel.pt')


while True:
    # Read a frame from the video
    frame = pyautogui.screenshot()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cv2.destroyAllWindows()
