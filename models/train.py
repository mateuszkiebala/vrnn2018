from keras import utils
from preprocess import Dataset, DataFetcher
from common.constants import SINGLE_MODEL_NAME, DUAL_MODEL_NAME, EPOCHS_BATCH

def train_and_evaluate(model, epochs, batches, dual=False, plot_history=False, plot_model=False):
    if plot_model:
        utils.plot_model(model, to_file='model.png')

    fetcher = DataFetcher()
    current_epochs = 0
    history = None

    if dual:
        data_type = 'split'
    else:
        data_type = 'concat'

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
        (x_train, y_train), (x_test, y_test) = dataset.data(type='concat')
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save(SINGLE_MODEL_NAME)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
