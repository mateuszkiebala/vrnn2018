import keras, argparse, os
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.applications import VGG16
from common.constants import DEFAULT_IMAGE_SIZE, GAMES_ARR_PATH, SINGLE_MODEL_NAME
from train import parse_args, train_and_evaluate

# constants
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE*2, 3)

args = parse_args()

def compiled_single_model(model_input_shape):
    input = Input(shape=model_input_shape)

    vgg = VGG16(weights='imagenet', input_shape=model_input_shape, include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    output_vgg = vgg(input)

    model = Flatten()(output_vgg)
    model = Dense(256, activation='relu')(model)
    model = Dropout(.2)(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(.2)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(.2)(model)
    model = Dense(args.num_classes, activation=args.last_activation)(model)
    model = Model(inputs=input, outputs=model)

    if len(args.gpus) > 1:
        model = keras.utils.multi_gpu_model(model, len(args.gpus), cpu_merge=False)

    model.compile(
        loss=args.loss,
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

train_and_evaluate(model, args.epochs, args.batches, gpus=args.gpus, plot_history=args.plot_history, plot_model=args.plot_model)
