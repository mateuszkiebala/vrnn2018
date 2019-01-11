import keras, argparse, os
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras.applications import VGG16
from keras import backend as K
from preprocess import Dataset, DataFetcher
from common.constants import DEFAULT_IMAGE_SIZE, SINGLE_MODEL_NAME, GAMES_ARR_PATH, EPOCHS_BATCH

# constants
num_classes = 2 # 0 or 1
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE*2, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--batches', type=int, default=64, help='Number of batches')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--plot-model', action='store_true', help='Determines if structure of the model should be plotted')
parser.add_argument('--plot-history', action='store_true', help='Determines if history of loss and accuracy should be plotted')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="3"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def compiled_single_model(model_input_shape):
    input = Input(shape=model_input_shape)

    vgg = VGG16(weights='imagenet', input_shape=model_input_shape)

    for layer in vgg.layers:
        layer.trainable = False

    output_vgg = VGG16(weights='imagenet')(input)

    model = Flatten()(output_vgg)
    # model = Dense(512, activation='relu')(model)

    # model = ZeroPadding2D((3, 3))(model)

    # model = Conv2D(32, (3, 3), activation='relu')(model)
    # model = BatchNormalization(axis=3)(model)
    # model = Activation('relu')(model)
    # model = MaxPooling2D((3, 3))(model)

    # model = Conv2D(64, (3, 3), activation='relu')(model)
    # model = MaxPooling2D((3, 3))(model)


    # model = Conv2D(128, (3, 3), activation='relu')(model)
    # model = BatchNormalization(axis=3)(model)
    # model = Activation('relu')(model)
    # model = MaxPooling2D((3, 3))(model)

    # model = Conv2D(256, (3, 3), activation='relu')(model)
    # model = MaxPooling2D((3, 3))(model)

    # model = Dropout(.5)(model)
    # model = Flatten()(model)


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

fetcher = DataFetcher()
current_epochs = 0
history = None

for samples in fetcher.fetch_inf():
    if current_epochs >= args.epochs:
        break

    (x_train, y_train), (x_test, y_test) = samples

    history = model.fit(
        x_train, y_train,
        batch_size=args.batches,
        epochs=EPOCHS_BATCH + current_epochs,
        initial_epoch=current_epochs,
        verbose=1,
        validation_data=(x_test, y_test),
    )


    current_epochs += EPOCHS_BATCH
    model.save(SINGLE_MODEL_NAME)

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
