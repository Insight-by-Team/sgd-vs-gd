import matplotlib
# Force matplotlib to not use any Xwindows backend (for SSH computing)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_acc_history(history, name, title='model accuracy'):
    # summarize history for accuracy
    fig = plt.figure(figsize=(12, 6))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim(0., 1.05)
    plt.savefig(name)
    plt.close(fig)

