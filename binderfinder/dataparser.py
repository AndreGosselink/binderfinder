import csv
import numpy as np
import warnings
from errors import InvalidHeader, InconsistentData, InvalidLabels


#TODO relational data model?
class Parser(object):
    PROP_KEY = 'properties'
    PARA_KEY = 'parameters'
    PARA_LABEL = 'labels'

    def __init__(self, filename, csv_setup={'delimiter': ';'}):
        self.data_layout = {  self.PROP_KEY: -1,
                              self.PARA_KEY: -1,
                            self.PARA_LABEL: -1,}

        header_error_msg = filename + ", Line {}: first two lines of file must be " + "'{};UINT' and '{};UINT'".format(self.PROP_KEY, self.PARA_KEY)
        data_error_msg = filename + ", Line {}: inconsistency while parsing found"
        incons_error_msg = filename + ", inconistency in level of '{}' while checking found"

        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, **csv_setup)
            try:
                self.parse_header(reader)
            except InvalidHeader as IH:
                print IH
                raise InvalidHeader(header_error_msg.format(reader.line_num))
            
            try:
                data = self.parse_data(reader)
            except InconsistentData:
                raise InconsistentData(data_error_msg.format(reader.line_num))
            
            incons_dset = self.find_inconsisty(data)
            if incons_dset != '':
                raise InconsistentData(incons_error_msg.format(incons_dset))
            
            self.data = data


    def parse_header(self, reader):
        header_lines = []
        while len(header_lines) < 2:
            line = reader.next()
            try:
                if line[0].startswith('#'):
                    continue
                else:
                    header_lines.append(line)
            except:
                line = []
        for layout in header_lines:
            # check if header lines is valid
            try:
                name, val = layout[:2]
                name = name.lower()
                val = int(val)
                if not self.data_layout.has_key(name):
                    raise ValueError
                # and set key
                self.data_layout[name] = val
                # check labels
                if name == self.PARA_KEY:
                    labels = layout[2:2+val]
                    if labels == [] or len(labels) != val:
                        warnings.warn('No labesl defined or labesl dont match parameter count!', InvalidLabels)
                        # raise Warning('No labesl defined or labesl dont match parameter count!')
                        labels = ['Parameter_{}'.format(i) for i in xrange(val)]
                    self.data_layout[self.PARA_LABEL] = labels

            except ValueError:
                raise InvalidHeader
        
        # final testing of header
        propcount = self.data_layout[self.PROP_KEY]
        paramcount = self.data_layout[self.PARA_KEY]
        if  propcount == 1 or propcount < 0:
            raise InvalidHeader('Properties need to be =0 or >=2')

        if  paramcount <= 0:
            raise InvalidHeader('Parameter need to be =0 or >=2')


    def parse_data(self, reader):
        prop_num = self.data_layout[self.PROP_KEY]
        para_num = self.data_layout[self.PARA_KEY]

        param_template = lambda str_vals: np.array([float(x.replace(',', '.')) for x in str_vals], float)
        data = {}
        for n, dset in enumerate(reader):
            curdict = data
            if prop_num >= 2:
                props = dset[:prop_num]
            elif prop_num == 0:
                props = [str(n), 'DATA']
            values = dset[prop_num:]
            for prop in props[:-1]:
                # check data consistency
                if len(dset[prop_num:prop_num+para_num]) != para_num:
                    raise InconsistentData
                curdict = curdict.setdefault(prop, {})
            curdict[props[-1]] = param_template(values)
        return data

    def find_inconsisty(self, data):
        keys = data.keys()
        try:
            ref_sub = data[keys[0]].keys()
        except AttributeError:
            return ''

        for k in keys:
            if not all([sub in ref_sub for sub in data[k].keys()]):
                return k
            res = self.find_inconsisty(data[k])
            if res != '':
                return '{}/{}'.format(k, res)

        return ''

    def get_matrix_formatted(self):
        #TODO implement correct access w/o using old data format -> need some hacking in the matrix

        binder = self.data.keys()
        linker = self.data[binder[0]].keys()
        data = [[], []]

        for b in binder:
            for l in linker:
                data[0].append(self.data[b][l][0])
                data[1].append(self.data[b][l][1])

        return binder, linker, np.asarray(data, float).T

    def get_pca_formatted(self):

        binder = self.data.keys()
        linker = self.data[binder[0]].keys()
        data = [[] for i in xrange(self.data_layout[self.PARA_KEY])]
        
        dset = linker[0]
        for b in xrange(len(binder)):
            b = str(b)
            for i in xrange(self.data_layout[self.PARA_KEY]):
                data[i].append(self.data[b][dset][i])

        return binder, linker, np.asarray(data, float).T


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
