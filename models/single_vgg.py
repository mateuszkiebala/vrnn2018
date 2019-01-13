import keras, argparse, os
import tensorflow as tf
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.applications import VGG16
from common.constants import DEFAULT_IMAGE_SIZE, GAMES_ARR_PATH, SINGLE_MODEL_NAME
from train import train_and_evaluate

# constants
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE*2, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--batches', type=int, default=64, help='Number of batches')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--extlabels', action='store_true', help='Determines if generator should generate extended labels')
parser.add_argument('--plot-model', action='store_true', help='Determines if structure of the model should be plotted')
parser.add_argument('--plot-history', action='store_true', help='Determines if history of loss and accuracy should be plotted')
args = parser.parse_args()

if args.extlabels:
    num_classes = 18
    loss = keras.losses.binary_crossentropy
    last_activation = 'sigmoid'
else:
    num_classes = 2
    loss = keras.losses.categorical_crossentropy
    last_activation = 'softmax'

os.environ["CUDA_VISIBLE_DEVICES"]="3"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def compiled_single_model(model_input_shape):
    input = Input(shape=model_input_shape)

    vgg = VGG16(weights='imagenet', input_shape=model_input_shape, include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    output_vgg = vgg(input)

    model = Flatten()(output_vgg)

    model = Dense(512, activation='relu')(model)
    model = Dropout(.5)(model)

    model = Dense(512, activation='relu')(model)
    model = Dropout(.5)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(.5)(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(.5)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(.5)(model)
    print(num_classes)
    model = Dense(num_classes, activation=last_activation)(model)
    model = Model(inputs=input, outputs=model)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'],
    )

    return model

# Model
try:
    model = load_model(SINGLE_MODEL_NAME)
    print("Model loaded from disk")
    create_model = False
except Exception:
    create_model = True

if create_model:
    print("Creating new single vgg model")
    model = compiled_single_model(input_shape)

train_and_evaluate(model, args.epochs, args.batches, plot_history=args.plot_history, plot_model=args.plot_model)