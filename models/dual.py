import keras, argparse
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Add
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE

# constants
num_classes = 2 # 0 or 1
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--batches', type=int, default=128, help='Number of batches')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
args = parser.parse_args()

dataset = Dataset()
dataset.load()
(x_train_1, x_train_2, y_train), (x_test_1, x_test_2, y_test) = dataset.data(type='split')

print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

print(x_train_1.shape)
print(x_train_2.shape)

def half_model():
    input = Input(shape=input_shape)
    model = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(.25)(model)
    model = Flatten()(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(.5)(model)
    return Model(inputs=input, outputs=model)

before_model = half_model()
after_model = half_model()

merged_model = Add()([before_model.output, after_model.output])
merged_model = Dense(64, activation='relu')(merged_model)
merged_model = Dropout(.25)(merged_model)
merged_model = Dense(32, activation='relu')(merged_model)
merged_model = Dropout(.25)(merged_model)
merged_model = Dense(num_classes, activation='softmax')(merged_model)

whole_model = Model([before_model.input, after_model.input], merged_model)

whole_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'],
)

whole_model.fit(
    [x_train_1, x_train_2], y_train,
    batch_size=args.batches,
    epochs=args.epochs,
    verbose=1,
    validation_data=([x_test_1, x_test_2], y_test),
)

score = whole_model.evaluate([x_test_1, x_test_2], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
