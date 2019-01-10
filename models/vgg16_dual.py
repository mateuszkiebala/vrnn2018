import keras, argparse
from keras.utils import plot_model
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras import backend as K
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE
from keras.applications.vgg16 import VGG16

# constants
num_classes = 2 # 0 or 1
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--batches', type=int, default=64, help='Number of batches')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--plot-model', action='store_true', help='Determines if structure of the model should be plotted')
args = parser.parse_args()

dataset = Dataset()
dataset.load(0)
(x_train_1, x_train_2, y_train), (x_test_1, x_test_2, y_test) = dataset.data(type='split')

print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

print(x_train_1.shape)
print(x_train_2.shape)


def half_model():
    inputs = Input(shape=input_shape)
    model_vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    model_vgg16.trainable = False
    model = Dropout(.5)(model_vgg16.output)
    model = Flatten()(model)
    return Model(inputs=inputs, outputs=model)


before_model = half_model()
after_model = half_model()
for layer in after_model.layers:
    layer.name += str("_2")

merged_model = concatenate([before_model.output, after_model.output], axis=-1)
merged_model = Dense(512, activation='relu')(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(256, activation='relu')(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(128, activation='relu')(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(64, activation='relu')(merged_model)
merged_model = Dropout(.5)(merged_model)
merged_model = Dense(num_classes, activation='softmax')(merged_model)

whole_model = Model([before_model.input, after_model.input], merged_model)

if args.plot_model:
    plot_model(whole_model, to_file='model.png')

whole_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'],
)

history = whole_model.fit(
    [x_train_1, x_train_2], y_train,
    batch_size=args.batches,
    epochs=args.epochs,
    verbose=1,
    shuffle=True,
    validation_data=([x_test_1, x_test_2], y_test),
)

score = whole_model.evaluate([x_test_1, x_test_2], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
