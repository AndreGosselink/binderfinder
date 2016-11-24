import csv
import numpy as np


def parse_csv(filename):

    binder = []
    linker = []
    data = [[], []]
    
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:

            b = row[0]
            l = row[1]

            if not b in binder:
                binder.append(b)

            if not l in linker:
                linker.append(l)

            data[0].append(float(row[2].replace(',', '.')))
            data[1].append(float(row[3].replace(',', '.')))

    return binder, linker, np.asarray(data, float).T

