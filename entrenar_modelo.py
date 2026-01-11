import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore

# --- 1. CARGAR Y PROCESAR DATOS ---
DATA_PATH = os.path.join('datos_MP_LSM') 
actions = np.array(['hola', 'gracias', 'te_quiero']) # Deben ser las mismas que recolectaste
no_sequences = 30 # 30 videos por seña
sequence_length = 30 # 30 frames por video

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Cargando datos...")

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Cargar cada frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(f"Datos cargados exitosamente. Total secuencias: {len(sequences)}")

X = np.array(sequences)
y = to_categorical(labels).astype(int) # Convertir etiquetas a formato binario [1,0,0], [0,1,0]...

# Separar datos: 95% para entrenar, 5% para testear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 2. ARQUITECTURA DE LA RED NEURONAL (LSTM) ---
model = Sequential()

# Capa 1: LSTM que devuelve secuencias (pasa información a la siguiente capa LSTM)
# input_shape es (30 frames, X características)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))

# Capa 2: Otra capa LSTM para refinar patrones
model.add(LSTM(128, return_sequences=True, activation='relu'))

# Capa 3: Última capa LSTM, ya no devuelve secuencias, sino el resultado final del análisis temporal
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Capas Densas (Brain): Para tomar decisiones basadas en lo que aprendió la LSTM
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Capa de Salida: Tiene tantas neuronas como acciones (3). 
# 'softmax' nos da la probabilidad (ej: 90% Hola, 5% Gracias, 5% Te quiero)
model.add(Dense(actions.shape[0], activation='softmax'))

# --- 3. COMPILACIÓN Y ENTRENAMIENTO ---
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenamos por 200 épocas (vueltas completas a los datos)
model.fit(X_train, y_train, epochs=200, callbacks=[TensorBoard(log_dir='logs')])

# --- 4. GUARDAR EL MODELO ---
model.summary()
model.save('lsm_modelo.h5') # Guardamos el cerebro entrenado
print("¡Modelo entrenado y guardado como 'lsm_modelo.h5'!")