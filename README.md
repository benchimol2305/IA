# clasificador de residuos en tiempo real

este proyecto implementa un clasificador de residuos en tiempo real usando vision por computadora y redes neuronales convolucionales.

## estructura del proyecto

```text
.
├── app.py
├── train.py
├── modelo_residuos.h5
├── requirements.txt
├── dataset-resized/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── ejemplos/
    ├── ejemplo_cardboard.png
    ├── ejemplo_glass.png
    ├── ejemplo_metal.png
    ├── ejemplo_paper.png
    ├── ejemplo_plastic.png
    └── ejemplo_trash.png
dataset
se utilizo un dataset publico de residuos solidos:

trashnet dataset  
link: https://github.com/garythung/trashnet

para este proyecto se uso la version redimensionada:

carpeta: dataset-resized

clases utilizadas:

cardboard

glass

metal

paper

plastic

trash

coloca la carpeta dataset-resized en la raiz del proyecto, con las subcarpetas anteriores y las imagenes dentro de cada una.

requisitos
instalar dependencias:

bash
pip install -r requirements.txt
contenido sugerido de requirements.txt:

text
tensorflow
opencv-python
numpy
entrenamiento del modelo
para entrenar el modelo desde cero:

bash
python train.py
esto:

carga las imagenes desde dataset-resized/

divide automaticamente en entrenamiento y validacion (validation_split=0.2)

entrena un modelo basado en mobilenetv2

guarda el modelo entrenado en modelo_residuos.h5

ejecucion del clasificador en tiempo real
una vez entrenado el modelo (y generado modelo_residuos.h5), ejecutar:

bash
python app.py
esto abrira la camara y mostrara:

la imagen en tiempo real

la clase predicha

la probabilidad de la prediccion

para cerrar la ventana, presiona la tecla q
