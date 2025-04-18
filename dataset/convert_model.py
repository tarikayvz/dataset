import tensorflow as tf
import numpy as np

try:
    # Eğitilmiş modeli yükle
    print("Model yükleniyor...")
    model = tf.keras.models.load_model('plant_classifier.h5')
    print("Model başarıyla yüklendi.")
    
    # Sınıf isimlerini yükle
    class_names = np.load('class_names.npy', allow_pickle=True)
    print(f"Sınıf etiketleri yüklendi: {class_names}")
    
    # TFLite dönüştürücüsünü oluştur
    print("TensorFlow Lite'a dönüştürülüyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizasyon seçenekleri (isteğe bağlı)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Dönüştürme işlemi
    tflite_model = converter.convert()
    
    # TFLite modelini kaydet
    with open('plant_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model başarıyla TensorFlow Lite'a dönüştürüldü: plant_classifier.tflite")
    
    # Metadata ekle (isteğe bağlı)
    try:
        from tflite_support import metadata as md
        from tflite_support import metadata_schema_py_generated as md_schema
        
        # Metadata oluştur
        model_meta = md.MetadataPopulator.with_model_file('plant_classifier.tflite')
        model_meta.load_metadata_buffer(md.MetadataPopulator.create_metadata_buffer(
            model_description="Bitki sağlığı sınıflandırma modeli",
            input_names=["input_image"],
            output_names=["class_probabilities"],
            associated_files=[md.LabelFile(file_path="labels.txt", locale="tr")]
        ))
        
        # Etiketleri dosyaya kaydet
        with open("labels.txt", "w") as f:
            for label in class_names:
                f.write(f"{label}\n")
        
        # Metadata'yı kaydet
        model_meta.populate()
        print("Model metadata'sı eklendi.")
    except ImportError:
        print("Metadata eklemek için tflite-support paketi gerekli.")
        print("Yüklemek için: pip install tflite-support")
    
except Exception as e:
    print(f"Hata oluştu: {e}")