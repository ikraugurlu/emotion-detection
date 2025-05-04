import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import re
import json

model_path = """C:/Users/LENOVO/Desktop/PROJEM-aty/face_landmarker.task"""

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Düzenli ifade (regex) ile x, y, z değerlerini ayıklama

# Create a callback function for the face landmarker
# Global veri yapısı
landmark_history = {}

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global landmark_history

    for face_landmarks in result.face_landmarks:
        for i, landmark in enumerate(face_landmarks):
            point = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "timestamp_ms": timestamp_ms
            }

            if i not in landmark_history:
                landmark_history[i] = []

            landmark_history[i].append(point)

    # JSON çıktısını oluştur
    json_result = json.dumps(landmark_history, indent=4)

    # Konsola kısa bilgi yaz (isteğe bağlı)
    print(f"Frame işlendi. Landmark 1 toplam {len(landmark_history.get(1, []))} değer içeriyor.")

    # Dosyaya kaydet (her frame'de güncellenmiş haliyle)
    with open("uzgun.json", "w") as f:
        f.write(json_result)



# print result içinde hata var!!
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    
    # Used to calculate FPS
    previous_frame_time = 0
    current_frame_time = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

            
        # Convert the frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Send the frame to the face landmarker
        # MediaPipe requires timestamp in ms
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
        
       
       
        # Calculate FPS
        current_frame_time = time.time()
        fps = 1 / (current_frame_time - previous_frame_time) if previous_frame_time > 0 else 0
        previous_frame_time = current_frame_time
        
        # Display the frame
        mirror=cv2.flip(image,2)
        # Display FPS on the frame
        cv2.putText(mirror, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Landmarks", mirror)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()