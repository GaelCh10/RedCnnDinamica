import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIGURACIÓN DE MODELOS (TASKS API) ---
base_options_hand = python.BaseOptions(model_asset_path='modelos/hand_landmarker.task')
base_options_pose = python.BaseOptions(model_asset_path='modelos/pose_landmarker.task')

# Configuramos para MODO VIDEO (procesamiento frame a frame con timestamp)
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

# --- 2. FUNCIONES DE DIBUJO Y EXTRACCIÓN ---

def draw_landmarks(image, hand_result, pose_result):
    """Dibuja esqueletos usando OpenCV puro (compatible con la nueva API)"""
    
    # Dibujar Pose (Cuerpo)
    if pose_result.pose_landmarks:
        for landmarks in pose_result.pose_landmarks:
            # Dibujar puntos
            for landmark in landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 4, (80,22,10), -1)
    
    # Dibujar Manos
    if hand_result.hand_landmarks:
        for landmarks in hand_result.hand_landmarks:
            for landmark in landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 4, (121,22,76), -1)

def extract_keypoints(hand_result, pose_result):
    """
    Aplana los resultados en un solo array.
    Orden: [Pose (33*4), Mano Izq (21*3), Mano Der (21*3)]
    Nota: La API Tasks no siempre distingue izq/der por orden, así que usaremos
    lógica basada en coordenadas o simplemente llenaremos slots de manos detectadas.
    Para simplificar entrenamiento, si hay 1 mano, llenamos el primer slot, etc.
    """
    
    # 1. Extraer Pose (33 puntos x 4 val: x,y,z,visibility)
    if pose_result.pose_landmarks:
        # Tomamos la primera detección de pose (usualmente solo hay 1 persona)
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33*4)

    # 2. Extraer Manos
    # La nueva API devuelve una lista de listas de landmarks.
    # Hand Landmarker puede devolver las manos en cualquier orden.
    # Para ser consistentes, vamos a intentar identificar izq/der basándonos en la etiqueta si está disponible,
    # o simplificar guardando Hand1 y Hand2.
    
    # Inicializamos vacíos
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)

    if hand_result.hand_landmarks and hand_result.handedness:
        for idx, classification in enumerate(hand_result.handedness):
            # classification es una lista de categorías, tomamos la primera
            label = classification[0].category_name # "Left" o "Right"
            landmarks = hand_result.hand_landmarks[idx]
            
            # Convertir a array plano (x, y, z)
            flat_lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            if label == 'Left':
                lh = flat_lms
            else:
                rh = flat_lms

    return np.concatenate([pose, lh, rh])

# --- 3. CONFIGURACIÓN DE DATOS ---
DATA_PATH = os.path.join('datos_MP_LSM') 
actions = np.array(['hola', 'gracias', 'te_quiero']) # CAMBIA ESTO POR TUS SEÑAS
no_sequences = 90
sequence_length = 30

# Crear carpetas
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- 4. BUCLE PRINCIPAL ---
detector_hand = vision.HandLandmarker.create_from_options(options_hand)
detector_pose = vision.PoseLandmarker.create_from_options(options_pose)

cap = cv2.VideoCapture(0) # Cambia a 1 si no abre

# Tiempo de inicio para calcular timestamps
start_time = time.time()

for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            if not ret: break

            # Preparar imagen para MediaPipe (RGB + Objeto mp.Image)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calcular timestamp actual en ms (necesario para modo VIDEO)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)

            # --- DETECCIONES ---
            # Ejecutamos ambos detectores
            hand_result = detector_hand.detect_for_video(mp_image, frame_timestamp_ms)
            pose_result = detector_pose.detect_for_video(mp_image, frame_timestamp_ms)

            # --- VISUALIZACIÓN ---
            draw_landmarks(frame, hand_result, pose_result)
            
            # Mensajes en pantalla
            if frame_num == 0: 
                cv2.putText(frame, 'COMENZANDO RECOLECCION', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, 'Accion: {} Video #{}'.format(action, sequence), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Recolector Tasks API', frame)
                cv2.waitKey(2000)
            else: 
                cv2.putText(frame, 'Accion: {} Video #{}'.format(action, sequence), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Recolector Tasks API', frame)

            # --- GUARDAR DATOS ---
            keypoints = extract_keypoints(hand_result, pose_result)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
detector_hand.close()
detector_pose.close()
cv2.destroyAllWindows()