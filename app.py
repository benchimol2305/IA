import cv2
import numpy as np
import tensorflow as tf

# cargar el modelo entrenado
print("cargando modelo...")
modelo = tf.keras.models.load_model("modelo_residuos.h5")
print("modelo cargado correctamente")

# clases del dataset
clases = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# abrir la camara
cam = cv2.VideoCapture(0)

print("presiona la tecla q para salir")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # preprocesar imagen
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # prediccion
    pred = modelo.predict(img, verbose=0)
    clase_idx = np.argmax(pred)
    prob = np.max(pred)

    texto = f"clase: {clases[clase_idx]}  prob: {prob:.2f}"

    # mostrar texto en pantalla
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("clasificador de residuos en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
