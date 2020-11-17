import matplotlib.pyplot as plt
import os

def plot(plot_dir,history):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.savefig(plot_dir+'/acc.png')
