import json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Birden fazla duygu dosyasını yükleme fonksiyonu
def load_emotion_data(emotion_files):
    """
    Her duygu için ayrı JSON dosyalarını yükler.
    emotion_files: {"duygu_adı": "dosya_yolu"} şeklinde bir sözlük
    """
    all_samples = []
    
    for emotion_name, file_path in emotion_files.items():
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # JSON veri yapısını kontrol et
            if isinstance(data, dict):
                # Her bir numara anahtarı için örnekleri işle
                for _, samples in data.items():
                    for sample in samples:
                        if all(key in sample for key in ['x', 'y', 'z', 'timestamp_ms']):
                            features = {
                                'x': sample['x'],
                                'y': sample['y'],
                                'z': sample['z'],
                                'timestamp_ms': sample['timestamp_ms']
                            }
                            features['emotion'] = emotion_name
                            all_samples.append(features)
            else:
                # Direkt liste olarak yüklenmiş veriler için
                for sample in data:
                    if all(key in sample for key in ['x', 'y', 'z', 'timestamp_ms']):
                        features = {
                            'x': sample['x'],
                            'y': sample['y'],
                            'z': sample['z'],
                            'timestamp_ms': sample['timestamp_ms']
                        }
                        features['emotion'] = emotion_name
                        all_samples.append(features)
                
            print(f"{emotion_name} duygusundan {len(all_samples)} örnek yüklendi.")
        except Exception as e:
            print(f"Hata: {file_path} dosyası yüklenirken bir sorun oluştu: {str(e)}")
    
    return pd.DataFrame(all_samples)

# Veri önişleme fonksiyonu
def preprocess_data(df):
    # Kategorik duygu isimlerini sayısal değerlere dönüştürme
    label_encoder = LabelEncoder()
    df['emotion_code'] = label_encoder.fit_transform(df['emotion'])
    
    # Etiket kodlarını isimlere eşleştirme bilgisini kaydet
    emotion_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nDuygu etiketleri eşleştirmesi:")
    for emotion, code in emotion_mapping.items():
        print(f"  {emotion}: {code}")
    
    # Zaman farkı özellikleri oluşturma
    df = df.sort_values(['emotion', 'timestamp_ms'])
    df['time_diff'] = df.groupby('emotion')['timestamp_ms'].diff().fillna(0)
    
    # Hız ve ivme gibi özellikler ekleyebiliriz (opsiyonel)
    # X, Y ve Z değerlerindeki değişimleri hesaplayalım
    for coord in ['x', 'y', 'z']:
        df[f'{coord}_diff'] = df.groupby('emotion')[coord].diff().fillna(0)
    
    # Gereksiz sütunları kaldırma
    df = df.drop('timestamp_ms', axis=1)
    
    # Özellikler ve etiketleri ayırma
    X = df.drop(['emotion', 'emotion_code'], axis=1)
    y = df['emotion_code']  # Sayısal kodları kullan
    
    return X, y, emotion_mapping

# Ana fonksiyon
def train_emotion_classifier(emotion_files, test_size=0.3, random_state=42):
    """
    Farklı duygu durumlarına ait verilerle model eğitir
    emotion_files: {"duygu_adı": "dosya_yolu"} şeklinde bir sözlük
    """
    # Veriyi yükleme
    print("Veriler yükleniyor...")
    df = load_emotion_data(emotion_files)
    print(f"Toplam yüklenen veri boyutu: {df.shape}")
    
    # Veri dağılımını gösterme
    emotion_counts = df['emotion'].value_counts()
    print("\nDuygu dağılımı:")
    print(emotion_counts)
    
    # Duygu sınıflarını görselleştirme
    plt.figure(figsize=(10, 6))
    sns.countplot(x='emotion', data=df)
    plt.title('Duygu Sınıflarının Dağılımı')
    plt.xlabel('Duygu Etiketi')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('emotion_distribution.png')
    
    # Veriyi önişleme
    print("\nVeri önişleniyor...")
    X, y, emotion_mapping = preprocess_data(df)
    
    # Eğitim ve test verilerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Eğitim seti: {X_train.shape[0]} örnek")
    print(f"Test seti: {X_test.shape[0]} örnek")
    
    # Özellik korelasyonlarını görselleştirme
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=.5)
    plt.title('Özellik Korelasyonları')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    
    # Model pipeline oluşturma
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    
    # Modeli eğitme
    print("\nModel eğitiliyor...")
    model.fit(X_train, y_train)
    
    # Modeli değerlendirme
    print("\nModel değerlendiriliyor...")
    y_pred = model.predict(X_test)
    
    # Performans metriklerini hesaplama
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Karmaşıklık matrisini görselleştirme
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Etiket isimleri ile karmaşıklık matrisini görselleştirme
    emotion_names = list(emotion_mapping.keys())
    emotion_codes = list(emotion_mapping.values())
    sorted_indices = np.argsort(emotion_codes)
    sorted_names = [emotion_names[i] for i in sorted_indices]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted_names, 
                yticklabels=sorted_names)
    plt.title('Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Duygu')
    plt.ylabel('Gerçek Duygu')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Özellik önemini görselleştirme
    feature_importances = model.named_steps['classifier'].feature_importances_
    features = X.columns
    
    # Özellik önemlerine göre sıralama
    indices = np.argsort(feature_importances)[::-1]
    top_features = [features[i] for i in indices]
    top_importances = [feature_importances[i] for i in indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features)
    plt.title('Özellik Önemi')
    plt.xlabel('Önem Skoru')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("\nModel eğitimi tamamlandı ve değerlendirildi.")
    return model, emotion_mapping

# Yeni bir örnek için duygu tahmini yapma fonksiyonu
def predict_emotion(model, emotion_mapping, x, y, z, time_diff=0):
    """
    Tek bir koordinat için duygu tahmini yapar
    
    Parameters:
    -----------
    model: Eğitilmiş sınıflandırıcı modeli
    emotion_mapping: Duygu etiketi eşleştirmesi sözlüğü
    x, y, z: Yüz koordinatları
    time_diff: Zaman farkı (ms)
    
    Returns:
    --------
    Tahmin edilen duygu ve olasılıklar
    """
    # Eksik özellikler için varsayılan değerler
    x_diff, y_diff, z_diff = 0, 0, 0
    
    # Yeni örneği bir DataFrame'e dönüştürme
    new_sample = pd.DataFrame({
        'x': [x],
        'y': [y],
        'z': [z],
        'time_diff': [time_diff],
        'x_diff': [x_diff],
        'y_diff': [y_diff],
        'z_diff': [z_diff]
    })
    
    # Duygu tahmini yapma
    prediction = model.predict(new_sample)
    probabilities = model.predict_proba(new_sample)
    
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
    
    # Sonucu döndürme
    return {
        'predicted_emotion': predicted_emotion,
        'predicted_code': predicted_code,
        'probabilities': probability_dict
    }

# Birden fazla koordinat için duygu tahmini yapma
def predict_emotion_sequence(model, emotion_mapping, coordinate_list):
    """
    Bir koordinat listesi için duygu tahmini yapar
    
    Parameters:
    -----------
    model: Eğitilmiş sınıflandırıcı modeli
    emotion_mapping: Duygu etiketi eşleştirmesi sözlüğü
    coordinate_list: X, Y, Z koordinatlarını ve zaman bilgisini içeren sözlük veya liste
    
    Returns:
    --------
    Her koordinat için tahmin edilen duygular ve olasılıklar
    """
    results = []
    
    # Koordinatları DataFrame'e dönüştürme
    df = pd.DataFrame(coordinate_list)
    
    # Zaman farkı hesaplama
    df['time_diff'] = df['timestamp_ms'].diff().fillna(0)
    
    # Koordinat farklarını hesaplama
    for coord in ['x', 'y', 'z']:
        df[f'{coord}_diff'] = df[coord].diff().fillna(0)
    
    # Her örnek için tahmin yapma
    for i, row in df.iterrows():
        sample = pd.DataFrame({
            'x': [row['x']],
            'y': [row['y']],
            'z': [row['z']],
            'time_diff': [row['time_diff']],
            'x_diff': [row['x_diff']],
            'y_diff': [row['y_diff']],
            'z_diff': [row['z_diff']]
        })
        
        # Duygu tahmini yapma
        prediction = model.predict(sample)
        probabilities = model.predict_proba(sample)
        
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
        
        # Sonucu listeye ekleme
        results.append({
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'timestamp_ms': row['timestamp_ms'],
            'predicted_emotion': predicted_emotion,
            'probabilities': probability_dict
        })
    
    return results

# Modeli kaydetme ve yükleme fonksiyonları
def save_model(model, emotion_mapping, file_path="emotion_classifier_model.pkl"):
    """Eğitilmiş modeli ve duygu etiketlerini kaydetme"""
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model, 'emotion_mapping': emotion_mapping}, f)
    print(f"Model başarıyla kaydedildi: {file_path}")

def load_model(file_path="emotion_classifier_model.pkl"):
    """Kaydedilmiş modeli ve duygu etiketlerini yükleme"""
    import pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['emotion_mapping']

# Örnek kullanım
if __name__ == "__main__":
    # Duygu verilerinin yolları
    emotion_files = {
        "mutlu": "mutlu.json",
        "uzgun": "uzgun.json",
        "heyecanli": "heyecanli.json"
        # Diğer duygu dosyalarını buraya ekleyin
    }
    
    # Modeli eğitme
    trained_model, emotion_mapping = train_emotion_classifier(emotion_files)
    
    # Modeli kaydetme
    save_model(trained_model, emotion_mapping)
    
    # Örnek bir test
    test_result = predict_emotion(
        model=trained_model,
        emotion_mapping=emotion_mapping,
        x=0.568,  # Örnek x koordinatı
        y=0.761,  # Örnek y koordinatı
        z=-0.039, # Örnek z koordinatı
        time_diff=100  # Örnek zaman farkı (ms)
    )
    
    print("\nÖrnek tahmin sonucu:")
    print(f"Tahmin edilen duygu: {test_result['predicted_emotion']}")
    print("Duygu olasılıkları:")
    for emotion, prob in test_result['probabilities'].items():
        print(f"  {emotion}: {prob:.4f}")
    
    # Örnek bir dizi koordinat için tahmin
    print("\nRealtime koordinat dizisi için tahmin örneği:")
    # Örnek koordinat dizisi oluşturma
    sample_sequence = [
        {'x': 0.568, 'y': 0.761, 'z': -0.039, 'timestamp_ms': 1000},
        {'x': 0.570, 'y': 0.762, 'z': -0.040, 'timestamp_ms': 1100},
        {'x': 0.572, 'y': 0.764, 'z': -0.042, 'timestamp_ms': 1200}
    ]
    
    sequence_results = predict_emotion_sequence(trained_model, emotion_mapping, sample_sequence)
    
    for i, result in enumerate(sequence_results):
        print(f"\nKoordinat {i+1} için tahmin:")
        print(f"X: {result['x']:.4f}, Y: {result['y']:.4f}, Z: {result['z']:.4f}")
        print(f"Duygu: {result['predicted_emotion']}")
        print("Olasılıklar:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")