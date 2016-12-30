from functionlib import euklid, theta
import numpy as np

def evaluate(a, c, a_ref, c_ref):
    """
    a: value for antibody
    c: value for carexpression

    a_ref: reference value for antibody
    c_ref: reference value for carexpression

    returns r, g, b as in RGB-Values
    """
    e = euklid([a, c], [a_ref, c_ref])
    t = theta([a, c], [a_ref, c_ref])
    if t in (0, 1):
        a_sig = 1
    else:
        a_sig = -1

    if t in (0, 4):
        c_sig = 1
    else:
        c_sig = -1
    if a_ref == 0: a_ref = 1
    if c_ref == 0: c_ref = 1
    a = e * np.exp( (a_sig * a) / a_ref )
    c = e * np.exp( (c_sig * c) / c_ref )
    return 0, a, c
            
def evaluate(a, c, a_ref, c_ref):
    """
    a: value for antibody
    c: value for carexpression

    a_ref: reference value for antibody
    c_ref: reference value for carexpression

    returns r, g, b as in RGB-Values
    """
    e = euklid([a, c], [a_ref, c_ref])
    t = theta([a, c], [a_ref, c_ref])

    return 0, a, c


def rgb_to_illumination(rgb):
    factors = (0.299, 0.587, 0.114)
    return np.asarray(map(f*c for f, c in zip(factors, rgb)), float)


def stats_calculation(datapoints):
    """
    datapoints: evaluated data, as in the rows/cols
                of the matrix. is a np.array with the
                dimension (n, 3), where n is the number 
                of datapoints in the matrix row/col and
                3 are the rgb values
    """
    return np.sum(datapoints, 0) / len(datapoints)


def sort_reduction(datapoints):
    """
    datapoints: reduces an rgb colour to a scalar value
                wicht then is used to order the data in
                a sort
    """
    return datapoints[:,1]

def
