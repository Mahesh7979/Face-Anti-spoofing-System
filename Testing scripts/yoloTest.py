# import math
# import time
#
# import cv2
# import cvzone
# from ultralytics import YOLO
#
#
# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 640)
# cap.set(4, 480)
# # cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
#
# model = YOLO("../models/yolov8n.pt")
#
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake" "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush", "pen", "knife"
#               ]
#
# prev_frame_time = 0
# new_frame_time = 0
#
# while True:
#     new_frame_time = time.time()
#     success, img = cap.read()
#     results = model(img, stream=True, verbose=False)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
#                                (max(0, x1), max(35, y1)), scale=1, thickness=4)
#
#     fps = 1 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time
#     print(fps)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
import math
import time
import cv2
import cvzone
from ultralytics import YOLO

# Load a higher-capacity YOLO model for better accuracy
model = YOLO("../models/yolov8n.pt")  # Try a larger model for better accuracy

# Class names for COCO dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "pen", "knife"]

# Initialize video capture
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        break

    # Resize and preprocess the image
    img_resized = cv2.resize(img, (640, 640))  # Adjust based on the model's input size

    # Run object detection
    results = model(img_resized, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf < 0.5:  # Adjust confidence threshold as needed
                continue

            # Class Name
            cls = int(box.cls[0])
            if cls < 0 or cls >= len(classNames):
                print(f"Warning: Detected class index {cls} out of range.")
                continue
            label = f'{classNames[cls]} {int(conf * 100)}%'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=4)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
