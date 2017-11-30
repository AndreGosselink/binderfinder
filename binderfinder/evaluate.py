from functionlib import euklid, theta
import numpy as np

# def evaluate(a, c, a_ref, c_ref):
#     """
#     a: value for antibody
#     c: value for carexpression
# 
#     a_ref: reference value for antibody
#     c_ref: reference value for carexpression
# 
#     returns r, g, b as in RGB-Values
#     """
#     e = euklid([a, c], [a_ref, c_ref])
#     t = theta([a, c], [a_ref, c_ref])
#     if t in (0, 1):
#         a_sig = 1
#     else:
#         a_sig = -1
# 
#     if t in (0, 4):
#         c_sig = 1
#     else:
#         c_sig = -1
#     if a_ref == 0: a_ref = 1
#     if c_ref == 0: c_ref = 1
#     a = e * np.exp( (a_sig * a) / a_ref )
#     c = e * np.exp( (c_sig * c) / c_ref )
#     return 0, a, c
            
def evaluate(params, weights, refs):
    """Calculates RGB mapping.

    Functions that gets all measured parameters as stored in the
    csv file including weights and refs. This function is calld for
    each sample. Can be replaced.
    
    Parameters
    ----------
    params : ndarray
        An array containing all parameters for each sample
    weights : iterable
        Weights as definded by Matrix(weights)
    refs : iterable
        Reference values as definded by Matrix(refs)
    Returns
    -------
    rgb : tuple
        Tuple of tree floats (r, g, b), where the first index is the
        intensitiy of red, second intensitiy of green and third
        intensity of blue.
    Notes
    -----
    The calculated rgb value is mapped accordingly to the sample form the
    csv file, the parameters where taken from. A short example would be

    ``rgb = params[0], params[1], params[2]``
    
    Here for each sample in the csv file, the first parameter per row is
    mapped to the red channel, the second is mpped to the green and the
    third to the ble channel

    ``rgb = np.median(params), np.mean(params), 0``

    Here in the read channel the median of all parameters is colorcoded
    red and the mean is colorcoded green. Blue is always 0.
    """
    rgb = 0, params[0], params[1]

    return rgb


def stats_calculation(datapoints):
    """Calculationg of row/colum statistics

    The default calculates the mean value of all RGB values in
    each row/colum
    
    Parameters
    ----------
    datapoints : ndarray
        RGB data as used in the Matrix. Thus, datapoints
        are the rows/columns of the Matrix. Is a ndarray
        with the shape (n, 3). It contains the 3 RGB vlaus
        (axis 1) where n (axis 0) is the number of tiles/fields
        in the respective row/column.
    Returns
    -------
    rgb : tuple
        Statistical RGB value representing the row/column datapoints
        was taken from.
    """
    return np.sum(datapoints, 0) / len(datapoints)


def sort_reduction(datapoints):
    """Reduces RGB Value to scalar.

    Sortig relays on ordering scalar values. As the statistics
    return RGB values (vector with three scalars). The data must
    be reduced. Per default, the mean value of all channels is
    calculated -> (r + g + b) / 3
        
    Parameters
    ----------
    datapoints : ndarray
        RGB data as calculated for the statistics rows/columns
        Is a ndarray with the shape (n, 3). It contains the
        3 RGB vlaus (axis 1) where n (axis 0) is the number of
        tiles/fields statistics row/column
    Returns : iterable
        Iterable with the length of the row/column entries. If
        (n, 3) datapoints are given, an iterable with (n, 1)=(n, ) 
        must be returned.
    """
    return (datapoints[:,1] + datapoints[:,2])/2.0


def rgb_to_illumination(rgb):
    factors = (0.299, 0.587, 0.114)
    return np.sum(np.asarray([f*c for f, c in zip(factors, rgb)], float))


