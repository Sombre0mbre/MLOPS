import cv2
import onnxruntime
import tensorflow as tf
import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from save_model import save_onnx

height, width = 224, 224
batch_size=64

CLASSES = ['Normal', 'Viral Pneumonia', 'Covid']

model_path = 'model/vgg16_best.onnx'
session = onnxruntime.InferenceSession(model_path)
input_name = session.get_inputs()[0].name


def to_rgb(image):
    """Converts the image to RGB if it is not already.
    Args:
        image: The image to convert.
    Returns:
        The image converted to RGB.
    """
    # conversion de l'image en couleurs selon le nombre de canaux
    if image.ndim == 2:
        # image en niveaux de gris : conversion en couleurs avec cv2.COLOR_GRAY2BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # image avec 4 canaux : suppression du canal alpha avec cv2.COLOR_RGBA2BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def get_prediction(image_file):
    """Returns the prediction of the model for the image located at image_path.
    Args:
        image_file: The image to predict.
    Returns:
        str: The prediction of the model.
    """
    image = Image.open(image_file)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = cv2.resize(image, (224, 224))
    # conversion de l'image en couleur si nécessaire
    image = to_rgb(image)
    # reshape de l'image pour qu'elle ait la forme attendue par le modèle (None, 1000, 1000, 3)
    image = image[np.newaxis, :, :, :]
    # normalisation des valeurs de l'image
    image = image / 255.0

    prediction = session.run(None, {input_name: image})[0]
    prediction = np.argmax(prediction, axis=1)
    return CLASSES[prediction[0]]


def generate_data(DIR):
    datagen = ImageDataGenerator(rescale=1. / 255.)

    generator = datagen.flow_from_directory(
        DIR,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        class_mode='binary',
        target_size=(height, width),
        classes={'Normal': 0, 'Viral Pneumonia': 1, 'Covid': 2}
    )
    return generator

def train(data_path):
    """Trains the model.
    Args:
        data_path: The path to the data.
    """
    TRAINING_DIR: str = f"{data_path}/train"
    TESTING_DIR: str = f"{data_path}/test"

    train_generator = generate_data(TRAINING_DIR)
    test_generator = generate_data(TESTING_DIR)

    model = tf.keras.models.load_model('model/vgg16_best.h5')

    model.compile(loss='SparseCategoricalCrossentropy',
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        metrics=['acc'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('model/vgg16_best.h5', monitor='acc', verbose=1, mode='max',
                                                    save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience=5)


    callbacks_list = [checkpoint, early]

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        # steps_per_epoch=10,
        epochs=50,
        shuffle=False,
        verbose=True,
        callbacks=callbacks_list)

    save_onnx('model/vgg16_best.h5', 'model/vgg16_best.onnx')


    return True
