import keras
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.regularizers import l1, l2
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE, DUAL_MODEL_NAME
from train import parse_args, train_and_evaluate

# constants
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 1)

args = parse_args()

def half_model():
    input = Input(shape=input_shape)
    model = Conv2D(64, (3, 3))(input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, (3, 3), strides=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, (3, 3), strides=(3, 3), activation='relu')(model)
    model = MaxPooling2D((3, 3))(model)
    model = Dropout(.5)(model)
    model = Flatten()(model)
    return Model(inputs=input, outputs=model)

def compiled_dual_model():
    before_model = half_model()
    after_model = half_model()

    merged_model = concatenate([before_model.output, after_model.output], axis=-1)
    merged_model = Dense(400, activation='relu')(merged_model)
    merged_model = Dropout(.25)(merged_model)
    merged_model = Dense(200, activation='relu')(merged_model)
    merged_model = Dropout(.25)(merged_model)
    merged_model = Dense(100, activation='relu')(merged_model)
    merged_model = Dropout(.25)(merged_model)
    #merged_model = Dense(32, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))(merged_model)
    merged_model = Dense(args.num_classes, activation=args.last_activation)(merged_model)
    whole_model = Model([before_model.input, after_model.input], merged_model)

    if len(args.gpus) > 1:
        model = keras.utils.multi_gpu_model(model, len(args.gpus), cpu_merge=False)

    print(whole_model.summary())
    whole_model.compile(
        loss=args.loss,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'],
    )

    return whole_model

print("Creating new dual v2 model")
model = compiled_dual_model()

train_and_evaluate(model, args.epochs, args.batches, gpus=args.gpus, dual=True, plot_history=args.plot_history, plot_model=args.plot_model)
