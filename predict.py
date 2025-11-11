import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
import numpy as np
import sys
import os

# Bu import, modelin mimarisini (EfficientNet)
# tanıyabilmesi için gereklidir.
import efficientnet.tfkeras

# --- Gerekli Kütüphaneler ---
# Kodu çalıştırmadan önce bu kütüphaneleri yüklemeniz gerekir:
# pip install tensorflow numpy efficientnet

# --- Model Ayarları (Notebook'unuzdan Alındı) ---

# 1. Görüntü Boyutları (snippet 2)
IMG_HEIGHT = 192
IMG_WIDTH = 256

# 2. Özel 'mish' Aktivasyon Fonksiyonu (snippet 5, 10)
# Modeliniz bu özel fonksiyonla eğitildiği için yüklerken gereklidir.
def mish(x):
    """Mish Aktivasyon Fonksiyonu (Lambda implementasyonu)"""
    return tf.keras.layers.Lambda(lambda x: x * K.tanh(K.softplus(x)))(x)

# Aktivasyon fonksiyonunu Keras'ın tanıması için kaydet
get_custom_objects().update({'mish': Activation(mish)})


MODEL_PATH = 'model_0.h5'

def predict_image(model, img_path):
    """
    Verilen model ve görüntü yolu ile katarakt tahmini yapar.
    (Notebook'taki 'predict_cataract' fonksiyonuna dayanarak - snippet 10)
    """
    try:
        # Görüntüyü yükle ve yeniden boyutlandır
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

        # Görüntüyü numpy array'e dönüştür
        img_array = image.img_to_array(img)

        # Modeli batch (yığın) olarak beslemek için boyutu genişlet (1, 192, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize et (Eğitimdeki gibi, 0-1 arasına getir - snippet 4)
        img_array /= 255.

        # Tahmin yap
        prediction = model.predict(img_array)

        # Sonucu yorumla (Notebook'taki mantığa göre - snippet 10)
        # prediction[0][0] -> Katarakt Yok olasılığı
        # prediction[0][1] -> Katarakt Var olasılığı
        if prediction[0][0] > prediction[0][1]:
            return "Tahmin: Katarakt Yok"
        else:
            return "Tahmin: Katarakt Var"

    except FileNotFoundError:
        return f"Hata: '{img_path}' görüntü dosyası bulunamadı."
    except Exception as e:
        return f"Görüntü işlenirken bir hata oluştu: {e}"

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # Komut satırından görüntü yolu alınıp alınmadığını kontrol et
    if len(sys.argv) != 2:
        print("\n---------------------------------------------------------")
        print(f"Kullanım: python {os.path.basename(__file__)} <görüntü_dosyasının_yolu>")
        print("Örnek: python tahmin_yap.py C:/resimler/goz.png")
        print("---------------------------------------------------------")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Modeli yükle (snippet 10)
        print(f"'{MODEL_PATH}' modeli yükleniyor...")
        # 'mish' aktivasyonunu custom_objects olarak belirtmek şart
        model = load_model(MODEL_PATH, custom_objects={'mish': Activation(mish)})
        print("Model başarıyla yüklendi.")

        # Tahmin yap
        print(f"'{image_path}' için tahmin yapılıyor...")
        result = predict_image(model, image_path)

        # Sonucu yazdır
        print("\n------------------- SONUÇ -------------------")
        print(result)
        print("-----------------------------------------------")

    except IOError:
        print(f"\nHata: Model dosyası '{MODEL_PATH}' bulunamadı.")
        print("Lütfen 'model_0.h5' dosyasının bu script ile aynı dizinde olduğundan emin olun.")
    except Exception as e:
        print(f"\nGenel bir hata oluştu: {e}")
