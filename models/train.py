import os, argparse
from preprocess import Dataset, DataFetcher
from common.constants import SINGLE_MODEL_NAME, DUAL_MODEL_NAME, EPOCHS_BATCH

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=64, help='Number of batches')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpus', type=str, default='', help='GPUs numbers separated by commas')
    parser.add_argument('--extlabels', action='store_true', help='Determines if generator should generate extended labels')
    parser.add_argument('--plot-model', action='store_true', help='Determines if structure of the model should be plotted')
    parser.add_argument('--plot-history', action='store_true', help='Determines if history of loss and accuracy should be plotted')
    args = parser.parse_args()
    args.gpus = [item for item in args.gpus.split(',') if item != '']

    import keras
    if args.extlabels:
        args.num_classes = 18
        args.loss = keras.losses.binary_crossentropy
        args.last_activation = 'sigmoid'
    else:
        args.num_classes = 2
        args.loss = keras.losses.categorical_crossentropy
        args.last_activation = 'softmax'

    return args

def train_and_evaluate(model, epochs, batches, gpus=[], dual=False, plot_history=False, plot_model=False):
    import keras, tensorflow as tf
    from keras import utils

    if len(gpus) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=','.join(gpus)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
        keras.backend.get_session().run(tf.global_variables_initializer())

    if plot_model:
        if dual:
            utils.plot_model(model, to_file='dual_model.png', show_shapes=True)
        else:
            utils.plot_model(model, to_file='single_model.png', show_shapes=True)

    fetcher = DataFetcher()
    current_epochs = 0
    history = None

    if dual:
        data_type = 'split'
    else:
        data_type = 'stack'

    for samples in fetcher.fetch_inf(type=data_type):
        if current_epochs >= epochs:
            break

        if dual:
            (x_train1, x_train2, y_train), (x_test1, x_test2, y_test) = samples

            history = model.fit(
                [x_train1, x_train2], y_train,
                batch_size=batches,
                epochs=EPOCHS_BATCH + current_epochs,
                initial_epoch=current_epochs,
                verbose=1,
                validation_data=([x_test1, x_test2], y_test),
            )
            model.save(DUAL_MODEL_NAME)
        else:
            (x_train, y_train), (x_test, y_test) = samples

            history = model.fit(
                x_train, y_train,
                batch_size=batches,
                epochs=EPOCHS_BATCH + current_epochs,
                initial_epoch=current_epochs,
                verbose=1,
                validation_data=(x_test, y_test),
            )
            model.save(SINGLE_MODEL_NAME)

        current_epochs += EPOCHS_BATCH

    if plot_history:
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

    dataset = Dataset()
    dataset.load(number=0)

    if dual:
        (x_train1, x_train2, y_train), (x_test1, x_test2, y_test) = dataset.data(type='split')
        score = model.evaluate([x_test1, x_test2], y_test, verbose=0)
        model.save(DUAL_MODEL_NAME)
    else:
        (x_train, y_train), (x_test, y_test) = dataset.data(type='stack')
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save(SINGLE_MODEL_NAME)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
