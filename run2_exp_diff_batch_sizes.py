import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import pickle
import numpy as np
import keras
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score, accuracy_score
import cnn
import utils


def main(debug):
    n_times = 10
    weights_dir = '_weights'
    models_dir = '_models'
    plots_dir = '_plots'
    results_filename = 'results.pkl'

    x_train, y_train, x_test, y_test, num_classes = prepare_data(debug)
    batches = [32, 128, 512, 20000, len(x_train)]
    WEIGHT_UPDATES = len(x_train) / 32 * 100

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    model = cnn.create_model(x_train.shape, num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])
    model.summary()

    results = {}
    # recover results
    if os.path.isfile(results_filename):
        results = pickle.load(open(results_filename, 'rb'))

    for weights_filename in sorted(glob.glob(weights_dir + '/*')):
        w_num = int(weights_filename[-3:])
        if w_num not in results:
            results[w_num] = {}

        for batch_size in batches:
            if batch_size not in results[w_num]:
                results[w_num][batch_size] = {}

            for exp_num in range(n_times):
                name = '{}_{}_{}'.format(weights_filename[-3:],
                                         batch_size, exp_num)

                if exp_num not in results[w_num][batch_size]:
                    print('Start {}'.format(name))
                else:
                    print('Skip: {} - already calculated'.format(name))
                    continue

                model.load_weights(weights_filename)
                updates_per_epochs = len(x_train) // batch_size
                epochs = WEIGHT_UPDATES // updates_per_epochs

                history = model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    shuffle=True)

                y_scores = model.predict_proba(x_test)

                results[w_num][batch_size][exp_num] = save_results(
                    model, name, history, y_test, y_scores,
                    models_dir, plots_dir,
                    debug)

                pickle.dump(results, open(results_filename, 'wb'))


def prepare_data(debug=True):
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10

    if debug:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, num_classes


def save_results(model, name, history, y_test, y_scores, models_dir, plots_dir,
                 debug=True):
    res = {}

    res['auc'] = roc_auc_score(y_test, y_scores)
    res['acc'] = accuracy_score(np.argmax(y_test, axis=1),
                                np.argmax(y_scores, axis=1))

    res['model'] = os.path.join(models_dir, name + '.h5')
    model.save(res['model'])

    res['plot'] = os.path.join(plots_dir, name + '.png')
    utils.save_acc_history(history, res['plot'])

    if debug:
        print('auc: {:.2f}\nacc: {:.2f}'.format(res['auc']*100, res['acc']*100))

    return res


if __name__ == '__main__':
    main(debug=True)
