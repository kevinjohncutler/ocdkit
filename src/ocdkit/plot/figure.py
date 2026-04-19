from .imports import *


def figure(nrow=None, ncol=None, aspect=1, **kwargs):
    figsize = kwargs.get('figsize', 2)
    if not isinstance(figsize, (list, tuple, np.ndarray)) and figsize is not None:
        figsize = (figsize*aspect, figsize)

    kwargs['figsize'] = figsize
    fig = Figure(**kwargs)
    # fig = plt.figure(**kwargs)
    if nrow is not None and ncol is not None:
        axs = []
        for i in range(nrow * ncol):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            axs.append(ax)
        return fig, axs
    else:
        ax = fig.add_subplot(111)
        return fig, ax


def split_list(lst, N):
    return [lst[i:i + N] for i in range(0, len(lst), N)]
