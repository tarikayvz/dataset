import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model('plant_classifier.h5')

# Sınıf etiketlerini yükle
class_names = np.load('class_names.npy', allow_pickle=True)

# Kamera başlat
cap = cv2.VideoCapture(0)  # 0, bilgisayarınızdaki ana kamerayı temsil eder

##sa ben onur
while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break
    
    # Görüntüyü yeniden boyutlandır (224x224), modelin beklentisi bu şekilde
    img = cv2.resize(frame, (224, 224))
    
    # Görüntüyü 4 boyutlu bir array'e dönüştür
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Tahmin yap
    prediction = model.predict(img_array)
    
    # En yüksek olasılığı bul
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = prediction[0][class_index]  # Güven seviyesi
    
    # Eğer güven seviyesi düşükse, tahmin yapma
    if confidence < 0.4:  # 90% güven seviyesinin altındaki tahminleri dikkate alma
        class_label = "Bilinmeyen"  # Düşük güvenle yapılan tahminler için
        confidence = 0  # Güven seviyesini sıfırla
    
    # Görüntü üzerinde tahmin etiketini göster
    cv2.putText(frame, f"Tahmin: {class_label} ({confidence*100:.2f}%)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Görüntüyü göster
    cv2.imshow('Kamera Görüntüsü', frame)
    
    # 'q' tuşuna basarak çıkabilirsiniz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
