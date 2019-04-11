
"""
        _
    .__(.)<  (KWAK)
     \___)    
~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_covariance_matrix(x1, x2, channels, title = "Covariance matrix"):

    x = [x1, x2]
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(x[i], cmap="OrRd")
        ticks = range(len(channels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(channels)
        ax.set_yticklabels(channels)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()

    # fig1, ax1 = plt.subplot(1, 2, 1)
    # plt.imshow(x1, cmap="OrRd")
   
    # ticks = range(len(channels))
    # ax1.set_xticks(ticks)
    # ax1.set_yticks(ticks)
    # ax1.set_xticklabels(channels)
    # ax1.set_yticklabels(channels)

    # fig2, ax2 = plt.subplot(1, 2, 2)
    # plt.imshow(x2, cmap="OrRd")
   
    # ticks = range(len(channels))
    # ax2.set_xticks(ticks)
    # ax2.set_yticks(ticks)
    # ax2.set_xticklabels(channels)
    # ax2.set_yticklabels(channels)


    # plt.colorbar() 
    # # plt.colorbar(boundaries=np.linspace(0,4e-10,1000)) 
    
    # plt.title(title)

    # plt.show()
