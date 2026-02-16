#
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

ruta_dataset = r"C:\Users\Cesar\Downloads\dataset-resized\dataset-resized"


clases = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

gen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

train_data = gen.flow_from_directory(
    ruta_dataset,
    target_size=(224, 224),
    batch_size=32,
    classes=clases,
    class_mode="categorical",
    subset="training"
)

val_data = gen.flow_from_directory(
    ruta_dataset,
    target_size=(224, 224),
    batch_size=32,
    classes=clases,
    class_mode="categorical",
    subset="validation"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation="relu")(x)
salida = Dense(len(clases), activation="softmax")(x)

modelo = Model(inputs=base.input, outputs=salida)

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("entrenando modelo real...")
modelo.fit(train_data, validation_data=val_data, epochs=5)

modelo.save("modelo_residuos.h5")
print("modelo_residuos.h5 generado correctamente")
