import matplotlib
import sys
import os
from . import __version__

if os.name != 'nt' or sys.platform != 'win32':
    print 'falling back to TkAgg'
    matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from dataparser import parse_csv
from evaluate import evaluate, stats_calculation, sort_reduction
import warnings
from eventhandler import EventHandler
from matplotlib.widgets import RadioButtons
import matplotlib.gridspec as gridspec


class Matrix(object):
    """
Input:
------------------------
 filename: path to csv file, containing parsable data

reference: refernce values as iterable which are passed to the evaluate
           function (default=[0.0, 0.0])

   weight: weights as iterable which are passed to the evaluate function
           (default=[1.0, 1.0])

 annotate: specifies how the matrix is labled. If 'none' the tiles in the
           matrix aren't labels at all, if 'data' only the tile representing
           data are labeled and if 'all' the data and statistics tiles are
           labeld. The labels show the used RGB values per row and per
           col (default='none').

    stats: Show per row/col statistics. The calculations are defined in
           elvaluate.py/stats_calculation() (default=False).

     sort: Defines the sorting behaviour. Can be sorted by 'row', 'col' or
           'both' or can be not sorted at all with 'none'. In order for
           sorting to work, stats needs to be True (default='none').

   legend: Defines the legend behaviour. Needs to be a string of two chars,
           the chars need to be 'r', 'g' or 'b'. The first char defines the
           color plotted along the x-axis, the second char defines the
           color along the y-axis (default='bg').

     ceil: Round the values for the matrix. The scalar data is categorized
           in decades, changing the readout to 0-10 %, 11-20 %, 21-30 %,
           and so on. Reduces the dynamic range and as a leads to a loss
           of information but increase of comparability.

normalize: normalizes the data prior the representation. if 'total' all
           channels are normalized by the overal, maximum value in the
           matrix. If 'channels', each channel is normalized the maximum
           value in the respective channel.

Evaluates a csv file containing arbitary values measured for any given
combination of up to two subtypes. The csv file needs to have the following
layout:
    
    type;subtype;val0;val1
    A;x;v0_Ax;v1_Ax
    A;y;v0_Ay;v1_Ay
    B;x;v0_Bx;v1_Bx
    B;y;v0_By;v1_By

where A and B are the maintypes, x and y are the subtypes with the respective
measured valus v0 and v1 for each kombination of A, B and x, y. Of course
subtype and maintype are interchangeable, as long as the data is formated
as described.
    """

    def __init__(self, filename, reference=[0.0, 0.0], weights=[1.0, 1.0],
            annotate='none', stats=False, sort='none', legend='',
            ceil=False, normalize='total', debug=False, cmap='grey'):

        # check and parse parameter
        if not annotate in ('none', 'data', 'all'):
            raise ValueError("annotate must be 'none', 'data' or 'all'")

        if not sort in ('none', 'row', 'col', 'both'):
            raise ValueError("sort must be 'none', 'row', 'col' or 'both'")
        else:
            if not stats:
                raise ValueError("stats must be True for sorting. Please restart and use 'Matrix(..., stats=True, ...)' in the main script")
        
        if len(legend) != 2:
            raise ValueError("legend must be string of any combination of two out of 'r', 'g' and 'b'")
        elif len(legend) == 2:
            if sum(['r' in legend, 'g' in legend, 'b' in legend]) < 2:
                raise ValueError("legend must only have 'r', 'g' or 'b'")

        if not normalize in ('total', 'channels'):
                raise ValueError("normalize must be 'total' or 'channels'")

        if not ceil and legend != '':
            warnings.warn('with ceil=False the legend my not reflect all the colours shown. The mouseover will reference the data as if ceil was True!')

        # set flags
        self._normalizeflag = normalize
        self._annotateflag = annotate
        self._sortflag = sort
        self._statsflag = stats
        self._legendflag = legend.lower()
        self._ceilflag = ceil
        self._doneflag = False
        self._weights = np.array(weights, float)
        self._debugflag = debug

        # parse file remember the filename of the parsed file
        self.typnames, self.subnames, self.data = parse_csv(filename)
        self.filename = filename
        # set reference value
        self.ref = np.asarray(reference, float)
        # derive properties
        self.types = len(self.typnames)
        self.subtypes = len(self.subnames)
        # dummies for calculations
        self._dat_sc = None
        self._dat_eu = None
        self._dat_th = None
        # set datacontainer for matrix 
        if self._statsflag:
            self._matrix = np.zeros((self.types+1, self.subtypes+1, 3))
        else:
            self._matrix = np.zeros((self.types, self.subtypes, 3))
        # add label for the stats row/cols according to parameter 
        if annotate == 'all':
            self.typnames += ['stats']
            self.subnames += ['stats']
        # cmap
        self._heatmap_color = cmap

    def __str__(self):
        """
        Retrun some basic info
        """
        return '{} types with {} subtypes in {} datapoints'.format(self.types, self.subtypes, len(self.data))
    
    def _evaluate(self):
        a_ref, c_ref = self.ref
        self.antigen = self.data[:,:1].squeeze()
        self.carexpr = self.data[:,1:2].squeeze()
        w_a, w_c = self._weights
        ci0, ci1 = ['rgb'.index(lc) for lc in self._legendflag]
        for i, (a, c) in enumerate(zip(self.antigen, self.carexpr)):
            rgb = evaluate(a * w_a, c * w_c, a_ref * w_a, c_ref * w_c)
            if any(map(lambda x: x < 0, rgb)):
                raise ValueError('Evaluation function can not negative color value!')
            self._matrix[i/self.subtypes, i%self.subtypes][ci0] = rgb[ci0] 
            self._matrix[i/self.subtypes, i%self.subtypes][ci1] = rgb[ci1] 

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
            self._matrix = self._ceil_arr(self._matrix)

    def _ceil_arr(self, arr):
        arr *= 10
        arr = np.ceil(arr)
        arr /= 10
        return arr

    def _annotate(self):
        if self._debugflag:
            self.cont_dir.text(0, 0, 'DEBUG', c='r')
            for i, sn in enumerate(self.subnames):
                for j, tn in enumerate(self.typnames):
                    c = [(3 - np.sum(self._matrix[j, i])) / 4] * 3
                    self._matax.text(i-0.25, j+0.1, '{} {}'.format(tn, sn), color=c)

        if self._annotateflag != 'none':
            for i, sn in enumerate(self.subnames):
                for j, tn in enumerate(self.typnames):
                    c = [(3 - np.sum(self._matrix[j, i])) / 4] * 3
                    ci0, ci1 = ['rgb'.index(lc) for lc in self._legendflag]
                    self._matax.text(i-0.5, j+0.15, '{:.1f}'.format(self._matrix[j, i, ci1]), color=c)
                    self._matax.text(i-0.25, j-0.25, '{:.1f}'.format(self._matrix[j, i, ci0]), color=c)

        # for row in self._matrix[:-1,:-1,:]:
        #     for rgb in row:
        #         cidx = ['rgb'.index(lc) for lc in self._legendflag]
        #         i, j = [np.ceil((rgb[idx]-0.1)*10) for idx in cidx]
        #         if (i, j) != (-1, -1) and (i, j) != (0, 0):
        #             marked.setdefault((i, j), 0)
        #             marked[(i, j)] += 1
        # 

        marked = {}
        i0, i1 = ['rgb'.index(lc) for lc in self._legendflag]
        for row in self._matrix[:-1,:-1,:]:
            for rgb in row:
                i, j = map(lambda v: np.round(v*10.0-1), (rgb[i0], rgb[i1]))
                marked.setdefault((i, j), 0)
                marked[(i, j)] += 1

        for (i, j), val in marked.items():
            # self._legax.text(i+0.2, j+0.5, '*', color='w')
            self._legax.text(i+0.2, j+0.4, str(val), color='w', size='xx-small')
        

    def _plot_legend(self):
        axis = np.linspace(0.1, 1.0, 10)
        names = [str(v) for v in axis]
        leg_matrix = np.zeros((10, 10, 3))
        grid = np.meshgrid(axis, axis)
        for lc, g in zip(self._legendflag, grid):
            i = 'rgb'.index(lc)
            leg_matrix[:,:,i] = g
        self._legax.set_xlabel(self._legendflag[0].upper())
        self._legax.set_ylabel(self._legendflag[1].upper())
        self._legax.imshow(leg_matrix, interpolation='none')
        self._legax.set_xticks(range(10))
        self._legax.set_xticklabels(names)
        self._legax.xaxis.tick_top()
        self._legax.set_yticks(range(10))
        self._legax.set_yticklabels(names)
        self._legpatch, = self._legax.plot([],[], lw=2, c='r')   

    def _set_ticks(self, xticks, yticks):
        for ax in (self._matax, self._heatax):
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks)
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks)

    def _plotit(self):
        rows = 5
        cols = 6
        if not self._legendflag:
            self.fig, (self._heatax, self._matax) = plt.subplots(1, 2)
        else:
            gs = gridspec.GridSpec(rows, cols)

            # self._matax = plt.subplot2grid(   (rows, cols), ( 0, 0), rowspan=rows, colspan=2)
            # self._heatax = plt.subplot2grid(  (rows, cols), ( 0, 4), rowspan=rows, colspan=2)
            # self._legax = plt.subplot2grid(   (rows, cols), ( 0, 2), rowspan=3, colspan=2)
            # self.cont_dir = plt.subplot2grid( (rows, cols), ( 3, 2), colspan=2)

            self._matax   = plt.subplot(gs[ :,  :2])
            self._heatax  = plt.subplot(gs[ :, 4:])
            self._legax   = plt.subplot(gs[ :3, 2:4])
            self.cont_dir = plt.subplot(gs[ -1,  2])

            self.cont_dir.set_title('sort on click by')
            self._plot_legend()
            self._matpatches = []
            self.fig = self._matax.figure
            self.fig.tight_layout()
            self.fig.canvas.set_window_title('binderfinder ' + __version__)

        self._matimg = self._matax.imshow(self._matrix, interpolation='none')
        self._heatimg = self._heatax.imshow(np.zeros(self._matrix.shape[:2]),
                                            interpolation='none', vmin=0, vmax=1, cmap=self._heatmap_color)

        plt.colorbar(mappable=self._heatimg, ax=self._heatax)
        self._update_matrixdata()

        self._set_ticks(self.subnames, self.typnames)

        # self._check_color = RadioButtons(self.cont_rgb, ('R', 'G', 'B', 'mean'), (False, False, True))
        self._check_dir = RadioButtons(self.cont_dir, ('row', 'col', 'both'))
        self._event_handler = EventHandler(self.fig, self, debug=self._debugflag)

    def _get_heat(self):
        heat = np.asarray([[0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2] for rgb in row] for row in self._matrix])
        return heat / np.max(heat)

    def _update_matrixdata(self):
        self._matimg.set_data(self._matrix)
        self._heatimg.set_data(self._get_heat())
        self._set_ticks(self.subnames, self.typnames)
        self.fig.canvas.draw()

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

    def _sort_row(self, colidx):
        new_matrix = self._matrix.copy()
        new_subnames = []

        for n, i in enumerate(np.argsort(sort_reduction(self._matrix[colidx,:-1].squeeze()))):
            new_matrix[:,n] = self._matrix[:,i]
            new_subnames.append(self.subnames[i])

        self._matrix = new_matrix
        self.subnames = new_subnames
        if self._annotateflag == 'all':
            self.subnames += ['stats']

    def _sort_col(self, rowidx):
        new_matrix = self._matrix.copy()
        new_typnames = []

        for n, i in enumerate(np.argsort(sort_reduction(self._matrix[:-1,rowidx].squeeze()))):
            new_matrix[n,:] = self._matrix[i,:]
            new_typnames.append(self.typnames[i])

        self._matrix = new_matrix
        self.typnames = new_typnames
        if self._annotateflag == 'all':
            self.typnames += ['stats']

    def _sort_matrix(self):
        
        if self._sortflag in ('both', 'col'):
            for step in xrange(3):
                self._sort_col(-1)

            if self._sortflag in ('both', 'row'):
                self._sort_row(-1)

    def run(self, show=False):
        self._evaluate()
        self._normalize()

        if self._statsflag:
            self._calc_stats()

        if self._sortflag != 'none':
            self._sort_matrix()
        
        self._plotit()

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
