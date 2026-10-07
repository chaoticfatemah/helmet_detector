import cv2
from ultralytics import YOLO

def start_helmet_detection():
    print("Model load ho raha hai, thoda wait karein...")
    
    # Base YOLOv8 model (already downloaded & cached)
    model = YOLO('yolov8n.pt') 
    
    # Laptop Camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera open nahi ho saka!")
        return

    print("Detection start ho gayi hai! Band karne ke liye 'q' press karein")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("Helmet Detection System", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_helmet_detection()
