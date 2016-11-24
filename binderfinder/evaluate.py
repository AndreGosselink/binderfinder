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

    return a, c, 0


def stats_calculation(datapoints):
    """
    datapoints: evaluated data, as in the rows/cols
                of the matrix. is a np.array with the
                dimension (n, 3), where n is the number 
                of datapoints in the matrix row/col and
                3 are the rgb values

    """
    return np.sum(datapoints, 0) / len(datapoints)
