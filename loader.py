import csv
import numpy as np


def load_iris(path='iris.data'):
    dset = {'sepal_length': [],
             'sepal_width': [],
            'petal_length': [],
             'petal_width': [],
                   'class': [],
           }
    
    with open(path, 'r') as dfile:
        irisreader = csv.reader(dfile, delimiter=';')
        irisreader.next()
        for row in irisreader:
            sl, sw, pl, pw = map(float, row[:-1])
            cl = float(row[-1])
            dset['sepal_length'].append(sl)
            dset['sepal_width'].append(sw)
            dset['petal_length'].append(pl)
            dset['petal_width'].append(pw)
            dset['class'].append(cl)
    
    dset['sepal_length'] = np.array(dset['sepal_length'])
    dset['sepal_width'] = np.array(dset['sepal_width'])
    dset['petal_length'] = np.array(dset['petal_length'])
    dset['petal_width'] = np.array(dset['petal_width'])
    dset['class'] = np.array(dset['class'])

    return dset


if __name__ == '__main__':
    for k, v in load_iris().items():
        print k, v[:5]



