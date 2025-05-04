import mediapipe as mp
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe ayarları
model_path = "C:/Users/LENOVO/Desktop/PROJEM-aty/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Eğitilmiş duygu tanıma modelini yükleme
def load_model(file_path="emotion_classifier_model.pkl"):
    """Kaydedilmiş modeli ve duygu etiketlerini yükleme"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Model başarıyla yüklendi: {file_path}")
        return data['model'], data['emotion_mapping']
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {str(e)}")
        print("Lütfen önce modeli eğittiğinizden emin olun.")
        return None, None

# Tanımlanmış belirli landmark noktaları (yüz ifadesi için önemli noktalar)
# MediaPipe Face Mesh'te toplam 468 nokta var, bunlardan duygular için önemli olanları seçiyoruz
EXPRESSION_LANDMARKS = [
    # Kaşlar
    67, 69, 66, 65, 107, 55, 56, 
    # Göz kenarları
    362, 382, 381, 380, 374, 373, 390, 249, 263, 
    # Burun
    5, 6, 197, 195, 197, 4, 
    # Ağız kenarları
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Dudaklar
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    # Çene
    172, 136, 150, 149, 176, 148, 152
]

# Global değişkenler
model = None
emotion_mapping = None
recent_predictions = []
MAX_PREDICTIONS = 10  # Son tahminlerin tutulacağı maksimum sayı
landmark_points = {}  # Son alınan landmark noktaları
current_emotion = "Tanımlanıyor..."
emotion_confidence = 0.0

# Landmark verilerinden özellik çıkarma
def extract_features(face_landmarks, timestamp_ms):
    features = {}
    
    # Önceki zaman damgasını almak için landmark_points kontrolü
    prev_timestamp = 0
    if landmark_points and 'timestamp_ms' in landmark_points:
        prev_timestamp = landmark_points['timestamp_ms']
    
    # Zaman farkını hesaplama
    time_diff = timestamp_ms - prev_timestamp if prev_timestamp > 0 else 0
    
    # Koordinat farklarını hesaplama için önceki değerleri alma
    prev_x = prev_y = prev_z = 0
    if landmark_points:
        for idx in EXPRESSION_LANDMARKS:
            key = f"landmark_{idx}"
            if key in landmark_points:
                prev_x = landmark_points[key]['x']
                prev_y = landmark_points[key]['y']
                prev_z = landmark_points[key]['z']
                break
    
    # Seçili landmark noktalarının koordinatlarını kaydetme
    for idx in EXPRESSION_LANDMARKS:
        if idx < len(face_landmarks):
            landmark = face_landmarks[idx]
            key = f"landmark_{idx}"
            
            # Koordinatları kaydet
            if key not in landmark_points:
                landmark_points[key] = {}
            
            landmark_points[key]['x'] = landmark.x
            landmark_points[key]['y'] = landmark.y
            landmark_points[key]['z'] = landmark.z
            
            # Özellikler için tek bir landmark kullanıyoruz (basitlik için)
            if idx == EXPRESSION_LANDMARKS[0]:  # İlk landmark için
                features['x'] = landmark.x
                features['y'] = landmark.y
                features['z'] = landmark.z
                
                # Koordinat farklarını hesaplama
                features['x_diff'] = landmark.x - prev_x if prev_x != 0 else 0
                features['y_diff'] = landmark.y - prev_y if prev_y != 0 else 0
                features['z_diff'] = landmark.z - prev_z if prev_z != 0 else 0
    
    # Zaman bilgisini kaydet
    landmark_points['timestamp_ms'] = timestamp_ms
    features['time_diff'] = time_diff
    
    return features

# Duygu tahmini yapma
def predict_emotion(features):
    global model, emotion_mapping, recent_predictions, current_emotion, emotion_confidence
    
    if model is None or emotion_mapping is None:
        return "Model yüklenmedi", {}
    
    try:
        # Özellikleri DataFrame'e dönüştürme
        df = pd.DataFrame([features])
        
        # Eksik sütunları kontrol etme ve ekleme
        required_columns = ['x', 'y', 'z', 'time_diff', 'x_diff', 'y_diff', 'z_diff']
        
        # Modelin beklediği sütunları doğru sırayla içeren boş bir DataFrame oluştur
        empty_df = pd.DataFrame(columns=required_columns)
        
        # Özellik DataFrame'ini bu sütunlarla birleştir (eksik olanları NaN ile doldurur)
        df = pd.concat([empty_df, df], ignore_index=True)
        
        # NaN değerleri 0 ile doldur
        df = df.fillna(0)
        
        # Sadece gerekli sütunları seç ve modelin beklediği sırayla düzenle
        df = df[required_columns]
        
        # Tahmin yapma
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Sayısal tahmin değerini duygu ismine çevirme
        emotion_names = list(emotion_mapping.keys())
        emotion_codes = list(emotion_mapping.values())
        
        # Tahmin edilen duyguyu bulma
        predicted_code = int(prediction[0])
        predicted_emotion = emotion_names[emotion_codes.index(predicted_code)]
        
        # Olasılıkları duygularla eşleştirme
        probability_dict = {}
        for i, prob in enumerate(probabilities[0]):
            for emotion, code in emotion_mapping.items():
                if code == i:
                    probability_dict[emotion] = float(prob)
                    break
        
        # Son tahminleri kaydet (stabilite için)
        recent_predictions.append(predicted_emotion)
        if len(recent_predictions) > MAX_PREDICTIONS:
            recent_predictions.pop(0)
        
        # En sık tahmin edilen duyguyu belirle
        from collections import Counter
        emotion_counts = Counter(recent_predictions)
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        confidence = emotion_counts[most_common_emotion] / len(recent_predictions)
        
        # Güncel duygu ve güven değerini güncelle
        current_emotion = most_common_emotion
        emotion_confidence = confidence
        
        return most_common_emotion, probability_dict
    
    except Exception as e:
        print(f"Tahmin yapılırken hata: {str(e)}")
        return "Hata", {}
    
# MediaPipe için callback fonksiyonu
def process_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_emotion, emotion_confidence
    
    if result.face_landmarks:
        # İlk yüzün işaretleyicilerini al (birden fazla yüz tespit edilebilir)
        face_landmarks = result.face_landmarks[0]
        
        # Özellik çıkarma
        features = extract_features(face_landmarks, timestamp_ms)
        
        # Duygu tahmini
        emotion, probabilities = predict_emotion(features)

# Ana program
def main():
    global model, emotion_mapping
    
    # Modeli yükle
    print("Duygu tanima modeli yukleniyor...")
    model, emotion_mapping = load_model()
    
    if model is None:
        print("Model yüklenemedi, çıkılıyor...")
        return
    
    # MediaPipe ayarları
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_result,
        num_faces=1  # Sadece bir yüz takip et
    )
    
    # Kamera bağlantısı
    cap = cv2.VideoCapture(0)
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # FPS hesaplaması için değişkenler
        previous_frame_time = 0
        current_frame_time = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Boş kamera karesi. Devam ediliyor...")
                continue
            
            # Görüntüyü RGB'ye dönüştür (MediaPipe için)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # MediaPipe işlemesi için kareyi gönder
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            # FPS hesapla
            current_frame_time = time.time()
            fps = 1 / (current_frame_time - previous_frame_time) if previous_frame_time > 0 else 0
            previous_frame_time = current_frame_time
            
            # Görüntüyü aynala
            mirror = cv2.flip(image, 1)  # 1 yatay aynalama
            
            # FPS ve duygu bilgisini ekranda göster
            cv2.putText(mirror, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mirror, f"Duygu: {current_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(mirror, f"Güven: {emotion_confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow("Duygu Tanıma", mirror)
            
            # 'q' tuşuna basılırsa döngüyü sonlandırq
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()