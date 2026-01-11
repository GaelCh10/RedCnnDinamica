import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model # type: ignore

# --- 1. CARGAR EL MODELO ENTRENADO ---
model = load_model('lsm_modelo.keras')
actions = np.array(['hola', 'gracias', 'te_quiero']) # El mismo orden que en el entrenamiento
colors = [(245,117,16), (117,245,16), (16,117,245)] # Colores para cada seña

# --- 2. CONFIGURACIÓN DE MEDIAPIPE (Tasks API) ---
# Asegúrate de que los archivos .task sigan en la carpeta 'modelos'
base_options_hand = python.BaseOptions(model_asset_path='modelos/hand_landmarker.task')
base_options_pose = python.BaseOptions(model_asset_path='modelos/pose_landmarker.task')

options_hand = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO)

options_pose = vision.PoseLandmarkerOptions(
    base_options=base_options_pose,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO)

# --- 3. FUNCIONES DE APOYO ---

def prob_viz(res, actions, input_frame, colors):
    """Función para dibujar las barras de probabilidad en pantalla"""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def extract_keypoints(hand_result, pose_result):
    """La misma función exacta que usaste para recolectar datos"""
    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33*4)

    lh = np.zeros(21*3)
    rh = np.zeros(21*3)

    if hand_result.hand_landmarks and hand_result.handedness:
        for idx, classification in enumerate(hand_result.handedness):
            label = classification[0].category_name 
            landmarks = hand_result.hand_landmarks[idx]
            flat_lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            if label == 'Left': lh = flat_lms
            else: rh = flat_lms

    return np.concatenate([pose, lh, rh])

# --- 4. BUCLE DE PREDICCIÓN ---
detector_hand = vision.HandLandmarker.create_from_options(options_hand)
detector_pose = vision.PoseLandmarker.create_from_options(options_pose)

sequence = [] # Aquí acumularemos los frames
sentence = [] # Historial de palabras detectadas
threshold = 0.75 # Confianza mínima (80%) para aceptar la seña

cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Procesamiento de imagen
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # Detección
    hand_result = detector_hand.detect_for_video(mp_image, frame_timestamp_ms)
    pose_result = detector_pose.detect_for_video(mp_image, frame_timestamp_ms)

    # Extracción de puntos
    keypoints = extract_keypoints(hand_result, pose_result)
    sequence.append(keypoints)
    
    # Mantener solo los últimos 30 frames
    sequence = sequence[-30:]

    if len(sequence) == 30:
        # PREDICCIÓN
        # Expandimos dimensiones para que coincida con lo que espera el modelo: (1, 30, 258)
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        # Lógica de visualización
        # Si la confianza es mayor al threshold
        if res[np.argmax(res)] > threshold: 
            current_action = actions[np.argmax(res)]
            
            # Solo agregamos la palabra si es distinta a la anterior (para no parpadear)
            if len(sentence) > 0: 
                if current_action != sentence[-1]:
                    sentence.append(current_action)
            else:
                sentence.append(current_action)

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        # Visualizar barras de probabilidad
        frame = prob_viz(res, actions, frame, colors)

    # Dibujar cuadro de texto con la palabra actual
    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Traductor LSM en Tiempo Real', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
detector_hand.close()
detector_pose.close()
cv2.destroyAllWindows()