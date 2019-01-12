import keras, argparse
from keras.utils import plot_model
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE, DUAL_MODEL_NAME
from train import train_and_evaluate

# constants
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)

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

def half_model():
    input = Input(shape=input_shape)
    model = ZeroPadding2D((3, 3))(input)
    model = Conv2D(64, (3, 3), activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.00005, maxval=0.00005, seed=None))(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3))(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = MaxPooling2D((3, 3))(model)
    model = Conv2D(256, (3, 3), activation='relu')(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3))(model)
    model = Conv2D(512, (3, 3), activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.00005, maxval=0.00005, seed=None))(model)
    model = MaxPooling2D((3, 3))(model)
    # model = AveragePooling2D((6, 6), name='avg_pool')(model)
    model = Dropout(.5)(model)
    model = Flatten()(model)
    return Model(inputs=input, outputs=model)

def compiled_dual_model():
    before_model = half_model()
    after_model = half_model()

    merged_model = concatenate([before_model.output, after_model.output],axis=-1)
    merged_model = Dense(512, activation='relu')(merged_model)
    merged_model = Dropout(.1)(merged_model)
    merged_model = Dense(256, activation='relu')(merged_model)
    merged_model = Dropout(.1)(merged_model)
    merged_model = Dense(128, activation='relu')(merged_model)
    merged_model = Dropout(.1)(merged_model)
    merged_model = Dense(64, activation='relu')(merged_model)
    merged_model = Dropout(.1)(merged_model)
    merged_model = Dense(num_classes, activation=last_activation)(merged_model)
    whole_model = Model([before_model.input, after_model.input], merged_model)

    if args.plot_model:
        plot_model(whole_model, to_file='model.png')

    whole_model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'],
    )

    return whole_model

# Model
try:
    model = load_model(DUAL_MODEL_NAME)
    print("Model loaded from disk")
    create_model = False
except Exception:
    create_model = True

if create_model:
    print("Creating new dual v1 model")
    model = compiled_dual_model()

train_and_evaluate(model, args.epochs, args.batches, dual=True, plot_history=args.plot_history, plot_model=args.plot_model)
