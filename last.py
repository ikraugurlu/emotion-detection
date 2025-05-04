import mediapipe as mp
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from collections import Counter

# MediaPipe setup
model_path = "face_landmarker.task"  # Update with your model path

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Facial landmark groups for feature extraction
LANDMARK_GROUPS = {
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [336, 296, 334, 293, 300],
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    'nose': [1, 2, 3, 4, 5, 6, 197, 195, 5, 4, 45, 220, 115, 49, 131, 134, 51, 5, 281, 275, 45],
    'mouth_outline': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    'mouth_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    'jaw': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
}

# Load trained emotion recognition model
def load_model(file_path="emotion_classifier_model.pkl"):
    """Load the saved model and emotion labels"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Model successfully loaded: {file_path}")
        return data['model'], data['emotion_mapping']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure you've trained the model first.")
        return None, None

# Data collection mode to gather training data
class DataCollector:
    def __init__(self, emotion_name, output_dir="./data"):
        self.emotion_name = emotion_name
        self.output_dir = output_dir
        self.data = {}
        self.counter = 0
        self.is_recording = False
        self.max_frames = 100
        self.current_frame = 0
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self):
        """Start recording facial landmarks for the specified emotion"""
        self.is_recording = True
        self.current_frame = 0
        self.data = {}
        print(f"Started recording '{self.emotion_name}' emotion data...")
        
    def stop_recording(self):
        """Stop recording and save the collected data"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Save data to file
        if self.data:
            file_path = os.path.join(self.output_dir, f"{self.emotion_name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(file_path, 'w') as f:
                import json
                json.dump(self.data, f)
            print(f"Saved {len(self.data)} frames of '{self.emotion_name}' emotion data to {file_path}")
        
    def process_landmarks(self, face_landmarks, timestamp_ms):
        """Process and save facial landmarks"""
        if not self.is_recording:
            return
            
        if self.current_frame >= self.max_frames:
            self.stop_recording()
            return
            
        # Create data structure for this frame
        frame_data = {}
        
        # Save all 468 landmarks
        for i, landmark in enumerate(face_landmarks):
            frame_data[i] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "timestamp_ms": timestamp_ms
            }
            
        # Add to data collection
        self.data[str(self.current_frame)] = frame_data
        self.current_frame += 1
        
        # Display progress
        progress = (self.current_frame / self.max_frames) * 100
        print(f"\rRecording: {progress:.1f}% complete", end="")

# Feature extraction for emotion recognition
def extract_features(face_landmarks, timestamp_ms):
    """Extract meaningful features from facial landmarks for emotion recognition"""
    features = {}
    
    # Distance metrics between key points
    
    # 1. Eye aspect ratio (EAR) - measures eye openness
    def calculate_ear(eye_points):
        """Calculate eye aspect ratio"""
        # Vertical distances
        v1 = np.sqrt((face_landmarks[eye_points[1]].x - face_landmarks[eye_points[5]].x)**2 + 
                     (face_landmarks[eye_points[1]].y - face_landmarks[eye_points[5]].y)**2)
        v2 = np.sqrt((face_landmarks[eye_points[2]].x - face_landmarks[eye_points[4]].x)**2 + 
                     (face_landmarks[eye_points[2]].y - face_landmarks[eye_points[4]].y)**2)
        
        # Horizontal distance
        h = np.sqrt((face_landmarks[eye_points[0]].x - face_landmarks[eye_points[3]].x)**2 + 
                    (face_landmarks[eye_points[0]].y - face_landmarks[eye_points[3]].y)**2)
        
        # EAR calculation
        return (v1 + v2) / (2.0 * h) if h > 0 else 0
    
    # Calculate EAR for both eyes
    left_eye_points = [33, 160, 158, 133, 153, 144]  # Simplified key points
    right_eye_points = [362, 385, 387, 263, 373, 380]  # Simplified key points
    
    features['left_ear'] = calculate_ear(left_eye_points)
    features['right_ear'] = calculate_ear(right_eye_points)
    features['avg_ear'] = (features['left_ear'] + features['right_ear']) / 2.0
    
    # 2. Mouth aspect ratio (MAR) - measures mouth openness
    mouth_top = face_landmarks[13]  # Top lip
    mouth_bottom = face_landmarks[14]  # Bottom lip
    mouth_left = face_landmarks[78]  # Left corner
    mouth_right = face_landmarks[308]  # Right corner
    
    # Vertical distance
    v_mouth = np.sqrt((mouth_top.x - mouth_bottom.x)**2 + (mouth_top.y - mouth_bottom.y)**2)
    
    # Horizontal distance
    h_mouth = np.sqrt((mouth_left.x - mouth_right.x)**2 + (mouth_left.y - mouth_right.y)**2)
    
    features['mar'] = v_mouth / h_mouth if h_mouth > 0 else 0
    
    # 3. Eyebrow position relative to eyes
    left_eyebrow_center = face_landmarks[107]  # Center of left eyebrow
    left_eye_center = face_landmarks[159]  # Center of left eye
    
    right_eyebrow_center = face_landmarks[336]  # Center of right eyebrow
    right_eye_center = face_landmarks[386]  # Center of right eye
    
    # Calculate vertical distances between eyebrow and eye
    features['left_eyebrow_eye_dist'] = left_eye_center.y - left_eyebrow_center.y
    features['right_eyebrow_eye_dist'] = right_eye_center.y - right_eyebrow_center.y
    features['avg_eyebrow_eye_dist'] = (features['left_eyebrow_eye_dist'] + features['right_eyebrow_eye_dist']) / 2.0
    
    # 4. Mouth corner position - detect smile/frown
    mouth_left = face_landmarks[61]  # Left corner of mouth
    mouth_right = face_landmarks[291]  # Right corner of mouth
    mouth_center_top = face_landmarks[13]  # Top center of mouth
    
    # Calculate relative heights of mouth corners
    features['left_mouth_corner_height'] = mouth_center_top.y - mouth_left.y
    features['right_mouth_corner_height'] = mouth_center_top.y - mouth_right.y
    features['mouth_corner_diff'] = features['left_mouth_corner_height'] - features['right_mouth_corner_height']
    
    # 5. Calculate center points for each feature group
    for group_name, landmark_indices in LANDMARK_GROUPS.items():
        x_sum = y_sum = z_sum = 0
        for idx in landmark_indices:
            landmark = face_landmarks[idx]
            x_sum += landmark.x
            y_sum += landmark.y
            z_sum += landmark.z
        
        n = len(landmark_indices)
        features[f'{group_name}_center_x'] = x_sum / n if n > 0 else 0
        features[f'{group_name}_center_y'] = y_sum / n if n > 0 else 0
        features[f'{group_name}_center_z'] = z_sum / n if n > 0 else 0
    
    # 6. Calculate distances between feature groups
    feature_groups = list(LANDMARK_GROUPS.keys())
    for i in range(len(feature_groups)):
        for j in range(i+1, len(feature_groups)):
            group1 = feature_groups[i]
            group2 = feature_groups[j]
            
            # Calculate Euclidean distance between group centers
            x1 = features[f'{group1}_center_x']
            y1 = features[f'{group1}_center_y']
            z1 = features[f'{group1}_center_z']
            
            x2 = features[f'{group2}_center_x']
            y2 = features[f'{group2}_center_y']
            z2 = features[f'{group2}_center_z']
            
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            features[f'{group1}_{group2}_dist'] = dist
    
    # 7. Add timestamp for temporal features
    features['timestamp_ms'] = timestamp_ms
    
    return features

# Global variables
model = None
emotion_mapping = None
recent_predictions = []
MAX_PREDICTIONS = 10  # Maximum number of predictions to keep for stability
current_emotion = "Analyzing..."
emotion_confidence = 0.0
data_collector = None

# Emotion prediction function
def predict_emotion(features):
    """Predict emotion from extracted features"""
    global model, emotion_mapping, recent_predictions, current_emotion, emotion_confidence
    
    if model is None or emotion_mapping is None:
        return "Model not loaded", {}
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Get all required columns that the model expects
        required_columns = model.feature_names_in_
        
        # Create empty DataFrame with required columns
        empty_df = pd.DataFrame(columns=required_columns)
        
        # Merge feature DataFrame (fills missing ones with NaN)
        df = pd.concat([empty_df, df], ignore_index=True)
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Select only the required columns in the expected order
        df = df[required_columns]
        
        # Make prediction
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Convert numeric prediction to emotion name
        emotion_names = list(emotion_mapping.keys())
        emotion_codes = list(emotion_mapping.values())
        
        predicted_code = int(prediction[0])
        predicted_emotion = emotion_names[emotion_codes.index(predicted_code)]
        
        # Match probabilities with emotions
        probability_dict = {}
        for i, prob in enumerate(probabilities[0]):
            for emotion, code in emotion_mapping.items():
                if code == i:
                    probability_dict[emotion] = float(prob)
                    break
        
        # Save recent predictions (for stability)
        recent_predictions.append(predicted_emotion)
        if len(recent_predictions) > MAX_PREDICTIONS:
            recent_predictions.pop(0)
        
        # Determine most common emotion
        emotion_counts = Counter(recent_predictions)
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        confidence = emotion_counts[most_common_emotion] / len(recent_predictions)
        
        # Update current emotion and confidence
        current_emotion = most_common_emotion
        emotion_confidence = confidence
        
        return most_common_emotion, probability_dict
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return "Error", {}

# MediaPipe callback function
def process_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_emotion, emotion_confidence, data_collector
    
    if result.face_landmarks:
        # Get landmarks of the first detected face
        face_landmarks = result.face_landmarks[0]
        
        # For data collection mode
        if data_collector and data_collector.is_recording:
            data_collector.process_landmarks(face_landmarks, timestamp_ms)
        else:
            # Extract features for emotion recognition
            features = extract_features(face_landmarks, timestamp_ms)
            
            # Predict emotion
            emotion, probabilities = predict_emotion(features)

# Process JSON data for training
def process_json_for_training(json_file_path):
    """Process a JSON file with facial landmark data for training"""
    try:
        with open(json_file_path, 'r') as f:
            import json
            data = json.load(f)
        
        all_features = []
        
        for frame_key, frame_data in data.items():
            # Check if we have landmarks in this frame
            if not frame_data:
                continue
                
            # Get timestamp from first landmark
            timestamp_ms = frame_data.get("0", {}).get("timestamp_ms", 0)
            if not timestamp_ms:
                first_key = list(frame_data.keys())[0]
                timestamp_ms = frame_data[first_key].get("timestamp_ms", 0)
            
            # Convert to list of landmarks for processing
            landmarks = [None] * 478  # Initialize with None for all possible landmarks
            
            for landmark_idx, landmark_data in frame_data.items():
                # Convert string index to integer
                idx = int(landmark_idx)
                
                # Create a simple object with x, y, z properties
                class LandmarkPoint:
                    def __init__(self, x, y, z):
                        self.x = x
                        self.y = y
                        self.z = z
                
                # Create landmark object
                landmarks[idx] = LandmarkPoint(
                    landmark_data.get("x", 0),
                    landmark_data.get("y", 0),
                    landmark_data.get("z", 0)
                )
            
            # Skip if we don't have enough valid landmarks
            if landmarks.count(None) > 100:  # Allow some missing landmarks
                continue
                
            # Fill any None values with default points
            for i in range(len(landmarks)):
                if landmarks[i] is None:
                    landmarks[i] = LandmarkPoint(0, 0, 0)
                    
            # Extract features
            features = extract_features(landmarks, timestamp_ms)
            all_features.append(features)
        
        return all_features
    
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        return []

# Train emotion classifier from collected data
def train_emotion_classifier(emotion_files, output_path="./"):
    """
    Train model using facial landmark data for different emotions
    emotion_files: {"emotion_name": "file_path"} dictionary
    """
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Process all emotion files
    all_features = []
    for emotion_name, file_path in emotion_files.items():
        features = process_json_for_training(file_path)
        for feature in features:
            feature['emotion'] = emotion_name
        all_features.extend(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    print(f"Total data size: {df.shape}")
    
    # Show emotion distribution
    emotion_counts = df['emotion'].value_counts()
    print("\nEmotion distribution:")
    print(emotion_counts)
    
    # Convert categorical emotions to numeric codes
    label_encoder = LabelEncoder()
    df['emotion_code'] = label_encoder.fit_transform(df['emotion'])
    
    # Save emotion mapping
    emotion_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nEmotion label mapping:")
    for emotion, code in emotion_mapping.items():
        print(f"  {emotion}: {code}")
    
    # Add temporal features
    df = df.sort_values(['emotion', 'timestamp_ms'])
    df['time_diff'] = df.groupby('emotion')['timestamp_ms'].diff().fillna(0)
    
    # Remove unnecessary columns
    df = df.drop('timestamp_ms', axis=1)
    
    # Split features and labels
    X = df.drop(['emotion', 'emotion_code'], axis=1)
    y = df['emotion_code']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and emotion mapping
    model_file = os.path.join(output_path, 'emotion_classifier_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'emotion_mapping': emotion_mapping}, f)
    
    print(f"\nModel successfully saved: {model_file}")
    return model, emotion_mapping

# Main program
def main():
    global model, emotion_mapping, data_collector
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'collect', 'train'],
                       help='Operation mode: detect, collect, or train')
    parser.add_argument('--emotion', type=str, help='Emotion name for data collection')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for saving/loading data')
    parser.add_argument('--model_path', type=str, default='emotion_classifier_model.pkl', 
                       help='Path to save/load the model')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Mode: collect training data
    if args.mode == 'collect':
        if not args.emotion:
            print("Error: --emotion parameter is required for collection mode")
            return
            
        print(f"Data collection mode for emotion: {args.emotion}")
        data_collector = DataCollector(args.emotion, args.data_dir)
        
    # Mode: train model
    elif args.mode == 'train':
        print("Training mode")
        # Find all emotion data files
        emotion_files = {}
        for file in os.listdir(args.data_dir):
            if file.endswith('.json'):
                emotion_name = file.split('_')[0]  # Extract emotion name from filename
                emotion_files[emotion_name] = os.path.join(args.data_dir, file)
        
        if not emotion_files:
            print("No emotion data files found in the specified directory.")
            return
            
        print(f"Found emotion files: {emotion_files}")
        train_emotion_classifier(emotion_files, os.path.dirname(args.model_path))
        return
        
    # Mode: detect emotion
    else:
        print("Emotion detection mode")
        # Load model
        print("Loading emotion recognition model...")
        model, emotion_mapping = load_model(args.model_path)
        
        if model is None:
            print("Model could not be loaded, exiting...")
            return
    
    # MediaPipe setup
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_result,
        num_faces=1  # Track only one face
    )
    
    # Camera connection
    cap = cv2.VideoCapture(0)
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Variables for FPS calculation
        previous_frame_time = 0
        current_frame_time = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Empty camera frame. Continuing...")
                continue
            
            # Convert image to RGB (for MediaPipe)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Send frame for MediaPipe processing
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            # Calculate FPS
            current_frame_time = time.time()
            fps = 1 / (current_frame_time - previous_frame_time) if previous_frame_time > 0 else 0
            previous_frame_time = current_frame_time
            
            # Mirror image for display
            mirror = cv2.flip(image, 1)  # 1 for horizontal mirroring
            
            # Display FPS and emotion info
            cv2.putText(mirror, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if args.mode == 'collect':
                # Show recording status
                status = "RECORDING" if data_collector.is_recording else "STANDBY"
                cv2.putText(mirror, f"Mode: COLLECT - {args.emotion} ({status})", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                if data_collector.is_recording:
                    progress = (data_collector.current_frame / data_collector.max_frames) * 100
                    cv2.putText(mirror, f"Progress: {progress:.1f}%", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Show emotion recognition results
                cv2.putText(mirror, f"Emotion: {current_emotion}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(mirror, f"Confidence: {emotion_confidence:.2f}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Display image
            cv2.imshow("Emotion Recognition", mirror)
            
            # Key controls
            key = cv2.waitKey(5) & 0xFF
            
            # Q: Quit
            if key == ord('q'):
                break
                
            # R: Start/Stop recording (in collection mode)
            elif key == ord('r') and args.mode == 'collect':
                if data_collector.is_recording:
                    data_collector.stop_recording()
                else:
                    data_collector.start_recording()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()