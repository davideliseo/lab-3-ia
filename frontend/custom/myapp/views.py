from django.shortcuts import render
from django.http import JsonResponse
from urllib.request import urlopen
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Configurar TensorFlow para usar solo la CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Obtén la ruta completa al archivo model.h5
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')

# Definir la clase del optimizador personalizado
class CustomAdam(tf.keras.optimizers.Adam):
    pass

# Registrar el optimizador personalizado
tf.keras.utils.get_custom_objects()['CustomAdam'] = CustomAdam

def index(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == 'POST':
        # Cargar el modelo entrenado
        model = tf.keras.models.load_model(model_path, compile=False)  # Deshabilitar la compilación

        if 'image' in request.FILES:
            # Se subió un archivo
            image_file = request.FILES['image']
            img = Image.open(image_file).convert('L')  # Convertir a escala de grises
        elif 'image_url' in request.POST:
            # Se proporcionó una URL de imagen
            image_url = request.POST['image_url']
            try:
                with urlopen(image_url) as url_response:
                    img = Image.open(url_response).convert('L')
            except:
                return JsonResponse({'error': 'No se pudo abrir la imagen desde la URL proporcionada.'})
        else:
            return JsonResponse({'error': 'No se proporcionó ninguna imagen.'})

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)  # Agregar una dimensión de canal
        predictions = model.predict(img_array)
        labels = ['Axe', 'Claw Hammer', 'Drill', 'Handsaw', 'Measuring tape', 'Paint brush', 'Pliers', 'Screwdriver', 'Shovel', 'Square tool']  # Reemplaza con tus etiquetas
        predicted_label = labels[np.argmax(predictions)]
        confidence = np.max(predictions)
        response = {
            'predicted_label': predicted_label,
            'confidence': float(confidence)
        }
        return JsonResponse(response)
    return JsonResponse({'error': 'Invalid request'})
