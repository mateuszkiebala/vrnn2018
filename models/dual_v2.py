import keras
from keras.models import Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Add, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, concatenate
from keras.regularizers import l1, l2
from preprocess import Dataset
from common.constants import DEFAULT_IMAGE_SIZE, DUAL_MODEL_NAME
from train import parse_args, train_and_evaluate

# constants
input_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)

args = parse_args()

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

def compiled_dual_model():
    before_model = half_model()
    after_model = half_model()

    merged_model = concatenate([before_model.output, after_model.output], axis=1)
    merged_model = Conv2D(256, (3, 3))(merged_model)
    merged_model = BatchNormalization(axis=3)(merged_model)
    merged_model = Activation('relu')(merged_model)
    merged_model = MaxPooling2D((3, 3))(merged_model)
    merged_model = Conv2D(512, (3, 3), activation='relu')(merged_model)
    merged_model = MaxPooling2D((3, 3))(merged_model)
    merged_model = Dropout(.05)(merged_model)
    merged_model = Flatten()(merged_model)
    merged_model = Dense(128, activation='relu')(merged_model)
    merged_model = Dropout(.05)(merged_model)
    merged_model = Dense(64, activation='relu')(merged_model)
    merged_model = Dropout(.05)(merged_model)
    merged_model = Dense(32, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))(merged_model)
    merged_model = Dropout(.05)(merged_model)
    merged_model = Dense(args.num_classes, activation=args.last_activation)(merged_model)
    whole_model = Model([before_model.input, after_model.input], merged_model)

    if len(args.gpus) > 1:
        model = keras.utils.multi_gpu_model(model, len(args.gpus), cpu_merge=False)

    whole_model.compile(
        loss=args.loss,
        optimizer=keras.optimizers.Adam(),
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
    print("Creating new dual v2 model")
    model = compiled_dual_model()

train_and_evaluate(model, args.epochs, args.batches, gpus=args.gpus, dual=True, plot_history=args.plot_history, plot_model=args.plot_model)
