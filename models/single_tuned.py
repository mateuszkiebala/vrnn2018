import keras, argparse, os
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from common.constants import DEFAULT_IMAGE_SIZE, SINGLE_MODEL_NAME, GAMES_ARR_PATH
from train import parse_args, train_and_evaluate

# constants
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 2) # stacked gray scale

args = parse_args()

if args.extlabels:
    num_classes = 18
    loss = keras.losses.binary_crossentropy
    last_activation = 'sigmoid'
else:
    num_classes = 2
    loss = keras.losses.categorical_crossentropy
    last_activation = 'softmax'

def compiled_single_model(model_input_shape):
    input = Input(shape=model_input_shape)
    model = ZeroPadding2D((3, 3))(input)

    model = Conv2D(32, (3, 3), activation='relu')(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3))(model)

    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D((3, 3))(model)


    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3))(model)

    model = Conv2D(256, (3, 3), activation='relu')(model)
    model = MaxPooling2D((3, 3))(model)

    model = Dropout(.25)(model)
    model = Flatten()(model)

    model = Dense(128, activation='relu')(model)
    model = Dropout(.25)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(.25)(model)

    model = Dense(num_classes, activation=last_activation)(model)
    model = Model(inputs=input, outputs=model)

    if len(args.gpus) > 1:
        model = keras.utils.multi_gpu_model(model, len(args.gpus), cpu_merge=False)

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
    print("Creating new single tuned model")
    model = compiled_single_model(input_shape)

train_and_evaluate(model, args.epochs, args.batches, gpus=args.gpus, plot_history=args.plot_history, plot_model=args.plot_model)
