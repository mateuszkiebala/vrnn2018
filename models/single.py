import keras, argparse
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras import backend as K
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE

# constants
num_classes = 2 # 0 or 1
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE*2, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--batches', type=int, default=64, help='Number of batches')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--plot-model', action='store_true', help='Determines if structure of the model should be plotted')
parser.add_argument('--plot-history', action='store_true', help='Determines if history of loss and accuracy should be plotted')
args = parser.parse_args()

dataset = Dataset()
dataset.load()
(x_train, y_train), (x_test, y_test) = dataset.data(type='concat')

print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

print(x_train.shape)
print(x_train.shape)

# Model

input = Input(shape=input_shape)
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

# model = Dense(512, activation='relu')(model)
# model = Dropout(.25)(model)
model = Dense(256, activation='relu')(model)
model = Dropout(.25)(model)
model = Dense(128, activation='relu')(model)
model = Dropout(.25)(model)
model = Dense(64, activation='relu')(model)
model = Dropout(.25)(model)

model = Dense(num_classes, activation='softmax')(model)
model = Model(inputs=input, outputs=model)

if args.plot_model:
    plot_model(model, to_file='model.png')

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'],
)

history = model.fit(
    x_train, y_train,
    batch_size=args.batches,
    epochs=args.epochs,
    verbose=1,
    validation_data=(x_test, y_test),
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

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("simple_model.h5")
