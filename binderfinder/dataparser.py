import csv
import numpy as np
from errors import InvalidHeader, InconsistentData


#TODO relational data model?
class Parser(object):
    PROP_KEY = 'properties'
    PARA_KEY = 'parameters'

    def __init__(self, filename, csv_setup={'delimiter': ';'}):
        self.data_layout = {self.PROP_KEY: -1,
                            self.PARA_KEY: -1,}

        header_error = filename + ", Line {}: first two lines of file must be " + "'{};UINT' and '{};UINT'".format(self.PROP_KEY, self.PARA_KEY)
        data_error = filename + ", Line {}: data seems to be inconsistent..."


        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, **csv_setup)
            try:
                self.parse_header(reader)
            except InvalidHeader:
                raise InvalidHeader(header_error.format(reader.line_num))
            
            try:
                data = self.parse_data(reader)
                print data
                seems to parse not deep enough, wrong last prop
                self.check_consistency(self, data)
                self.data = data
            except InconsistentData:
                raise InconsistentData(data_error.format(reader.line_num))

    def parse_header(self, reader):
        for layout in [reader.next() for i in xrange(2)]:
            # check if header lines is valid
            try:
                name, val = layout
                name = name.lower()
                val = int(val)
                if not self.data_layout.has_key(name) or val <= 0:
                    raise ValueError
                # and set key
                self.data_layout[name] = val
            except ValueError:
                raise InvalidHeader

    def parse_data(self, reader):
        prop_num = self.data_layout[self.PROP_KEY]
        para_num = self.data_layout[self.PARA_KEY]

        param_template = lambda str_vals: np.array([float(x.replace(',', '.')) for x in str_vals], float)
        data = {}
        for dset in reader:
            curdict = data
            for prop in dset[:prop_num-1]:

                # check data consistency
                if len(dset[prop_num:]) != para_num:
                    raise InconsistentData
                
                curdict = curdict.setdefault(prop, {})

            prop = dset[prop_num]
            values = dset[prop_num:]
            curdict[prop] = param_template(values)
        return data

    def check_consistency(self, data):

        try:
            keys = data.keys()
        except AttributeError:
            return True

        ref_sub = data[keys[0]].keys()

        for k in keys:
            if not all([sub in ref_sub for sub in data[k].keys()]):
                raise InconsistentData
            self.check_consistency(data[k])

        return True


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
