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
import pickle
import argparse

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
def train_emotion_classifier(emotion_files, output_path="./", test_size=0.3, random_state=42):
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
    plt.figure(figsize=(12, 6))
    sns.countplot(x='emotion', data=df)
    plt.title('Duygu Sınıflarının Dağılımı')
    plt.xlabel('Duygu Etiketi')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'emotion_distribution.png'))
    
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
    plt.figure(figsize=(14, 12))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=.5)
    plt.title('Özellik Korelasyonları')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_correlations.png'))
    
    # Model pipeline oluşturma
    print("\nModel oluşturuluyor...")
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
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    
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
    plt.savefig(os.path.join(output_path, 'feature_importance.png'))
    
    # Modeli ve etiket eşlemesini kaydet
    model_file = os.path.join(output_path, 'emotion_classifier_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'emotion_mapping': emotion_mapping}, f)
    
    print(f"\nModel başarıyla kaydedildi: {model_file}")
    return model, emotion_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Duygu tanıma modeli eğitimi')
    
    parser.add_argument('--mutlu', type=str, required=True, help='Mutlu yüz ifadeleri JSON dosyası')
    parser.add_argument('--uzgun', type=str, required=True, help='Üzgün yüz ifadeleri JSON dosyası')
    parser.add_argument('--heyecanli', type=str, required=True, help='Heyecanlı yüz ifadeleri JSON dosyası')
    parser.add_argument('--output', type=str, default='./', help='Çıktı dosyalarının kaydedileceği dizin')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test seti boyutu (0-1 arası)')
    
    args = parser.parse_args()
    
    # Duygu dosyaları sözlüğü oluştur
    emotion_files = {
        "mutlu": args.mutlu,
        "uzgun": args.uzgun,
        "heyecanli": args.heyecanli
    }
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output, exist_ok=True)
    
    # Modeli eğit
    train_emotion_classifier(emotion_files, args.output, args.test_size)