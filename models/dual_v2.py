import keras, argparse
from keras.utils import plot_model
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.regularizers import l1, l2
from keras import backend as K
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE

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

dataset = Dataset()
dataset.load(split=False, shuffle=True)
(x_1, x_2, y) = dataset.data(type='split')

print(y.shape[0], '# samples')

print(x_1.shape)
print(x_2.shape)

def half_model():
    input = Input(shape=input_shape)
    model = ZeroPadding2D((3, 3))(input)
    model = Conv2D(64, (6, 6))(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3))(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = MaxPooling2D((3, 3))(model)
    return Model(inputs=input, outputs=model)

before_model = half_model()
after_model = half_model()

merged_model = concatenate([before_model.output, after_model.output], axis=1)
merged_model = Conv2D(256, (3, 3))(merged_model)
merged_model = BatchNormalization(axis=3)(merged_model)
merged_model = Activation('relu')(merged_model)
merged_model = MaxPooling2D((3, 3))(merged_model)
merged_model = Conv2D(512, (3, 3), activation='relu')(merged_model)
merged_model = MaxPooling2D((3, 3))(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Flatten()(merged_model)
merged_model = Dense(512, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(256, activation='relu')(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(128, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(64, activation='relu')(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(32, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(num_classes, activation=last_activation)(merged_model)
whole_model = Model([before_model.input, after_model.input], merged_model)

if args.plot_model:
    plot_model(whole_model, to_file='model.png')

whole_model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

history = whole_model.fit(
    [x_1, x_2], y,
    validation_split=0.3,
    shuffle=True,
    batch_size=args.batches,
    epochs=args.epochs,
    verbose=1,
)

if args.plot_history:
    import matplotlib.pyplot as plt

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
