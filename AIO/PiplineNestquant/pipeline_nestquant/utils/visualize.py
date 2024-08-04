import os
import matplotlib.pyplot as plt

def save_plot(filename, data, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    for datum in data: ax.plot(*datum['data'], color=datum['color'], label=datum['label'])
    ax.legend()
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    plt.title(os.path.basename(filename).split('.')[0])
    fig.savefig(filename)
    plt.close('all')