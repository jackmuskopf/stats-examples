import numpy as np
import scipy.stats

def gaussian_filter(x, y, nx, sigma=1, nsigma=2):
    '''
    x: x-axis array of original signal
    y: y-axis array of original signal
    nx: number of evenly-spaced samples in resampled x axis
    sigma: determines the width of the gaussian filter used to compute filtered y values from original signal
    nsigma: number of standard deviations to include in the computation of the filtered y values
    '''
    
    if x.shape != y.shape:
        raise ValueError(f'Input arrays have different shapes: {x.shape}, {y.shape}')

    # sort input arrays
    argsort = np.argsort(x)
    x = x[argsort]
    y = y[argsort]

    # generate new x axis from x and nx
    xh = np.linspace(x[0], x[-1], nx)

    # for each _xh, compute the corresponding yh and variance in that window
    yh, errs = np.zeros(nx), np.zeros(nx)
    for i, _xh in enumerate(xh):

        # get indices in window
        _ix = np.argwhere(abs(x - _xh) < nsigma*sigma)

        # get original data in window
        _x = x[_ix]
        _y = y[_ix]

        # get gaussian env for this window
        envelope = scipy.stats.norm(_xh, sigma).pdf

        # compute weights for original data in this window
        w = envelope(_x)
        w = w/w.sum()

        # dot with orignal y to get estimated point
        yh[i] = w.T.dot(_y)

        # get weighted std for window
        errs[i] = np.sqrt(np.average((_y - yh[i])**2, weights=w))
    
    return xh, yh, errs
