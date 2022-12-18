import os
import subprocess
import logging
import keras

PATH = "../saved_models/covid_classification"

def get_model_dir():
    """Returns the path where to save the current model.
    The path is as follows: './saved_models/{incremental_integer}
    Note: all sub-directories inside 'saved_models' must have an incremental integer as name

    Returns:
      - Path to save a new model (str)
    """
    # create 'saved_models' directory if it does not exist
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # check for model directories inside 'saved_models'
    model_dirs = [int(i) for i in os.listdir(PATH)]

    # if there is no prior model, current model has version 1
    # otherwise, current model version is highest available version incremented by 1
    current_model = "1" if len(model_dirs) == 0 else str(max(model_dirs) + 1)

    return PATH + "/" + current_model + "/"


def save_model(model):
    """Saves the model in the directory returned by get_model_dir().

    Args:
      - model: model to save
    """
    model_dir = get_model_dir()
    model.save(model_dir)
    return model_dir


def save_onnx(model, model_name):
    """Saves the model in the directory returned by get_model_dir().

    Args:
      - model: model to save
    """
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # load model
    model = keras.models.load_model(model)

    # save model to tf format
    model_dir = save_model(model)

    subprocess.run(
        ['python', '-m', 'tf2onnx.convert', '--saved-model', model_dir, '--output', model_name])
