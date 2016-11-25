import numpy as np
import matplotlib.pyplot as plt
from dataparser import parse_csv
from evaluate import evaluate, stats_calculation
import warnings
import os


class Matrix(object):

    def __init__(self, filename, reference=[], annotate='none', stats=False, sort='none', legend='', ceil=False, normalize='total'):
        if not annotate in ('none', 'data', 'all'):
            raise ValueError("annotate must be 'none', 'data' or 'all'")

        if not sort in ('none', 'row', 'col', 'both'):
            raise ValueError("sort must be 'none', 'row', 'col' or 'both'")
        
        if len(legend) != 0 and len(legend) != 2:
            raise ValueError("legend must be empty string of any combination of two 'r', 'g' and 'b'")
        else:
            if sum(['r' in legend, 'g' in legend, 'b' in legend]) < 2:
                raise ValueError("legend must only have 'r', 'g' or 'b'")
        if not normalize in ('total', 'channels'):
                raise ValueError("normalize must be 'total' or 'channels'")
        
        self._normalizeflag = normalize
        self._annotateflag = annotate
        self._sortflag = sort
        self._statsflag = stats
        self._legendflag = legend.lower()
        self._ceilflag = ceil
        self._doneflag = False

        self.typnames, self.subnames, self.data = parse_csv(filename)

        self.ref = np.asarray(reference, float)

        self.types = len(self.typnames)
        self.subtypes = len(self.subnames)

        self._matrix = None
        self._dat_sc = None
        self._dat_eu = None
        self._dat_th = None

        if self._statsflag:
            self._matrix = np.zeros((self.types+1, self.subtypes+1, 3))
        else:
            self._matrix = np.zeros((self.types, self.subtypes, 3))
        
        if annotate == 'all':
            self.typnames += ['stats']
            self.subnames += ['stats']

        self.filename = filename

    def __str__(self):
        return '{} types with {} subtypes in {} datapoints'.format(self.types, self.subtypes, len(self.data))
    
    def _evaluate(self):
        a_ref, c_ref = self.ref
        self.antigen = self.data[:,:1].squeeze()
        self.carexpr = self.data[:,1:2].squeeze()
        for i, (a, c) in enumerate(zip(self.antigen, self.carexpr)):
            r, g, b = evaluate(a, c, a_ref, c_ref)
            if any(map(lambda x: x < 0, (r, g, b))):
                raise ValueError('Evaluation function can not negative color value!')
            self._matrix[i/self.subtypes, i%self.subtypes] = [r, g, b]

    def _normalize(self):
        if self._normalizeflag == 'total':
            self._matrix /= np.max(self._matrix)
        else:
            for i in xrange(3):
                max_val = np.max(self._matrix[:,:,i])
                if max_val == 0:
                    continue
                self._matrix[:,:,i] /= max_val

        if self._ceilflag:
            self._matrix *= 10
            self._matrix = np.ceil(self._matrix)
            self._matrix /= 10

    def _annotate(self):
        # if self._debug:
        #     for i, sn in enumerate(self.subnames):
        #         for j, tn in enumerate(self.typnames):
        #             c = [(3 - np.sum(self._matrix[j, i])) / 4] * 3
        #             self.ax.text(i-0.25, j+0.1, '{} {}'.format(tn, sn), color=c)

        for i, sn in enumerate(self.subnames):
            for j, tn in enumerate(self.typnames):
                c = [(3 - np.sum(self._matrix[j, i])) / 4] * 3
                ci0, ci1 = ['rgb'.index(lc) for lc in self._legendflag]
                self.ax.text(i-0.5, j+0.15, '{:.1f}'.format(self._matrix[j, i, ci1]), color=c)
                self.ax.text(i-0.25, j-0.25, '{:.1f}'.format(self._matrix[j, i, ci0]), color=c)

    def _plot_legend(self):
        axis = np.linspace(0.1, 1.0, 10)
        names = [str(v) for v in axis]
        leg_matrix = np.zeros((10, 10, 3))
        grid = np.meshgrid(axis, axis)
        for lc, g in zip(self._legendflag, grid):
            i = 'rgb'.index(lc)
            leg_matrix[:,:,i] = g
        self.leg.set_xlabel(self._legendflag[0].upper())
        self.leg.set_ylabel(self._legendflag[1].upper())
        self.leg.imshow(leg_matrix, interpolation='none')
        self.leg.set_xticks(range(10))
        self.leg.set_xticklabels(names)
        self.leg.xaxis.tick_top()
        self.leg.set_yticks(range(10))
        self.leg.set_yticklabels(names)

    def _plotit(self):
        if not self._legendflag:
            self.fig, self.ax = plt.subplots(1)
        else:
            self.fig, (self.ax, self.leg) = plt.subplots(1, 2)
            self._plot_legend()

        self.ax.imshow(self._matrix, interpolation='none')

        self.ax.set_xticks(range(len(self.subnames)))
        self.ax.set_xticklabels(self.subnames)
        self.ax.xaxis.tick_top()

        self.ax.set_yticks(range(len(self.typnames)))
        self.ax.set_yticklabels(self.typnames)

    def save_last_run(self):

        if not self._doneflag:
            warnings.warn('run() was not called before. Next time please call run() first (and double check the output)')
            self.run()
        
        path, filename = os.path.split(self.filename)

        try:
            fname, _ = filename.split('.')
        except ValueError:
            fname, = filename.split('.')

        self.fig.savefig(os.path.join(path, fname) + '_matrix.png')
        self._dump2csv(os.path.join(path, fname) + '_matrixdata.csv')

    def _dump2csv(self, fname):
        with open(fname, 'w') as df:
            # make a header
            h0 = ''
            h1 = ''
            for i in xrange(2 * self.subtypes):
                h0 += ';{}'.format(self.subnames[i/2])
                h1 += ';{}'.format(['a', 'c'][i%2])

            df.write(h0 + '\n' + h1 + '\n')

            for i, (a, c) in enumerate(zip(self.antigen, self.carexpr)):
                if i%self.subtypes == 0:
                    df.write('{}'.format(self.typnames[i/self.subtypes]))

                df.write(';{};{}'.format(a, c))

                if i%self.subtypes == self.subtypes-1:
                    df.write('\n')

    def _calc_stats(self):
        for c in xrange(self.subtypes):
            col = self._matrix[:-1,c,:].squeeze()
            self._matrix[-1,c] = stats_calculation(col)

        for r in xrange(self.types):
            row = self._matrix[r,:-1,:].squeeze()
            self._matrix[r,-1] = stats_calculation(row)

    def _sort_matrix(self):
        new_matrix = self._matrix.copy()

        if self._sortflag in ('both', 'col'):
            new_subnames = []
            for n, i in enumerate(np.argsort(np.mean(self._matrix[-1,:-1], 1))):
                new_matrix[:,n] = self._matrix[:,i]
                new_subnames.append(self.subnames[i])
            self.subnames = new_subnames
            if self._annotateflag == 'all':
                self.subnames += ['stats']

        if self._sortflag in ('both', 'row'):
            new_typnames = []
            for n, i in enumerate(np.argsort(np.mean(self._matrix[:-1,-1], 1))):
                new_matrix[n,:] = self._matrix[i,:]
                new_typnames.append(self.typnames[i])
            self.typnames = new_typnames
            if self._annotateflag == 'all':
                self.typnames += ['stats']

        self._matrix = new_matrix

    def run(self, show=False):
        self._evaluate()
        self._normalize()

        if self._statsflag:
            self._calc_stats()

        if self._sortflag != 'none':
            self._sort_matrix()
        
        self._plotit()

        if self._annotateflag != 'none':
            self._annotate()


        self._doneflag = True

        if show:
            plt.show()

    def run_forrest_run(self, *args, **kwargs):
        kwargs['show'] = True
        return self.run(*args, **kwargs)

    def show_me_where_the_white_rabbit_goes(self, *args, **kwargs):
        kwargs['show'] = True
        return self.run(*args, **kwargs)
