from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = load_model('plant_classifier.h5')

# Sınıf etiketlerini yükle
class_names = np.load('class_names.npy', allow_pickle=True)
print(f"Sınıf etiketleri: {class_names}")

# Test görseli ile tahmin yapma
img_path = 'train/orchid_yellow/177968_jpg.rf.6e57176d6d8a98ffc79ec8bc2595bf99.jpg'  # Test görselinin yolu
img = image.load_img(img_path, target_size=(224, 224))  # Görseli yükle ve yeniden boyutlandır
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizasyon

# Tahmin yap
prediction = model.predict(img_array)
class_index = np.argmax(prediction)
predicted_class = class_names[class_index]

# Sonuçları yazdır
print(f"Tahmin edilen sınıf: {predicted_class}")
