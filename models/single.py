import keras, argparse, os
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras import backend as K
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE, SINGLE_MODEL_NAME, GAMES_ARR_PATH

# constants
num_classes = 2 # 0 or 1
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE*2, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--batches', type=int, default=64, help='Number of batches')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--plot-model', action='store_true', help='Determines if structure of the model should be plotted')
parser.add_argument('--plot-history', action='store_true', help='Determines if history of loss and accuracy should be plotted')
args = parser.parse_args()

class ChessGenerator(Sequence):

    def __init__(self, train=True):
        self.train = train
        self.dirs = [dir for dir in os.listdir(GAMES_ARR_PATH) if os.path.isdir(os.path.join(GAMES_ARR_PATH, directory))]

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        dir = self.dirs[idx]

        dataset = Dataset()
        dataset.load(number=dir)

        (x_train, y_train), (x_test, y_test) = dataset.data(type='concat')

        if self.train:
            yield (x_train, y_train)
        else:
            yield (x_test, y_test)

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

    model = Dropout(.5)(model)
    model = Flatten()(model)


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

    model = Dense(num_classes, activation='softmax')(model)
    model = Model(inputs=input, outputs=model)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
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
    print("Creating new single model")
    model = compiled_single_model(input_shape)


if args.plot_model:
    plot_model(model, to_file='model.png')

tr_gen = ChessGenerator(train=True)
ts_gen = ChessGenerator(train=False)

history = model.fit_generator(
    generator=tr_gen,
    # steps_per_epoch=1,
    # batch_size=args.batches,
    epochs=args.epochs,
    verbose=1,
    validation_data=ts_gen,
    validation_steps=1,
    max_queue_size=2,
    callbacks=[ModelCheckpoint(SINGLE_MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
)


if args.plot_history:
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

dataset = Dataset()
dataset.load(number=0)

(x_train, y_train), (x_test, y_test) = dataset.data(type='concat')
score = model.evaluate(x_test, y_test, verbose=0)

model.save(SINGLE_MODEL_NAME)
