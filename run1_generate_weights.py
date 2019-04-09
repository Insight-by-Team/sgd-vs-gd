import keras
from keras import backend as K
from keras.datasets import cifar10
import os
import cnn


def main(weights_dir, n=100):
    os.makedirs(weights_dir, exist_ok=True)

    (x_train, y_train), _ = cifar10.load_data()
    model = cnn.create_model(x_train.shape, 10)

    for i in range(n):
        reset_weights(model)
        model.save_weights(os.path.join(weights_dir, '{:03d}'.format(i)))


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, keras.engine.network.Network):
            reset_weights(layer)
            continue
        for v in layer.__dict__.values():
            if hasattr(v, 'initializer'):
                v.initializer.run(session=session)


if __name__ == '__main__':
    main('_weights', n=100)
