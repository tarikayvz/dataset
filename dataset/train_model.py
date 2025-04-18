import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Veri yolları
train_dir = 'train'
val_dir = 'val'

# Parametreler
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Veri hazırlama
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model oluşturma
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Önceden öğrenilmiş katmanları dondur

# Üzerine kendi sınıflandırma katmanlarımızı ekleyelim
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')  # Çıkış sınıf sayısı kadar
])

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Eğitim
checkpoint = ModelCheckpoint(
    'plant_classifier.h5',  # Model dosyasının ismi
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

# Sınıf etiketlerini kaydet (dönüştürme için gerekli)
class_indices = train_gen.class_indices
class_names = list(class_indices.keys())
np.save('class_names.npy', class_names)
print("Sınıf etiketleri kaydedildi: class_names.npy")

# Accuracy ve Loss grafikleri
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')

plt.tight_layout()
plt.show()

# Test görüntüsü ile modeli dene
try:
    from tensorflow.keras.preprocessing import image
    
    # Görseli yükle
    img_path = 'test_ornekleri/monstera1.jpg'  # Test görseli
    if os.path.exists(img_path):
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Tahmin yap
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_labels = list(train_gen.class_indices.keys())  # Sınıf adlarını al

        print(f"Tahmin edilen sınıf: {class_labels[class_index]}")
    else:
        print(f"Test görseli bulunamadı: {img_path}")
except Exception as e:
    print(f"Test sırasında hata oluştu: {e}")

print("\nModel eğitimi tamamlandı ve kaydedildi: plant_classifier.h5")
print("Şimdi convert_model.py dosyasını çalıştırabilirsiniz.")