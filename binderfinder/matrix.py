import matplotlib
import sys
import os
from . import __version__

if os.name != 'nt' or sys.platform != 'win32':
    print 'falling back to TkAgg'
    matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
# from dataparser import parse_csv
from dataparser import Parser
from evaluate import evaluate, stats_calculation, sort_reduction, rgb_to_illumination
import warnings
from eventhandler import EventHandler
from matplotlib.widgets import RadioButtons
import matplotlib.gridspec as gridspec

#TODO: Data Pipeline


class Matrix(object):
    """Main class for matrix visualisation

    Evaluates a csv file containing arbitary values measured for any given
    combination of up to two subtypes. Evaluation is done by mapping the parameters
    to the RGB space.

    Parameters
    ----------
    filename : path
        path to csv file, containing parsable data
    reference : iterable
        reference values as iterable which are passed to the evaluate function.
    weight : iterable
        weights as iterable which are passed to the evaluate function.
    annotate : {'none', 'data', 'all'}
        'none':
          Tiles in Matrix are not labeled.
        'data':
          Only tiles referencing to datapoints from the inputfile are labeled.
        'all':
          Data and statistics tiles are labeled. Additionally the used RGB values
          are shown in the tiles.
    stats : {True, False}
        Show per row/col statistics. The calculations are defined in
        elvaluate.py/stats_calculation().
    sort : {'none', 'row', 'col', 'both'}
        'none':
          No sorting. For sorting `stats` needs to be True
        'row':
          Sort matrix according to row statistics.
        'col':
          Sort matrix according to column statistics.
        'both':
          First sort by row statistics and afterwards by column statistics
    legend : {'rb', 'br', 'rg', 'gr', 'gb', 'bg'}
        Defines the legend behaviour. Needs to be a string of two chars,
        the chars need to be 'r', 'g' or 'b'. The first char defines the
        color plotted along the x-axis, the second char defines the
        color along the y-axis (default='bg').
    ceil : {True, False}
        Round the values for the matrix. The scalar data is categorized
        in decades, changing the readout to 0-10 %, 11-20 %, 21-30 %,
        and so on. Reduces the dynamic range and as a leads to a loss
        of information but increase of comparability.
    normalize : {'total', 'channels'}
        'total':
          All channels are normalized by the overal, maximum value in the
          matrix.
        'channels':
          Each channel is normalized the maximum value in the respective
          channel.
    ch_lablels : list of strings
        Name displayed for the RGB channels

    Notes
    -----
    The csv file needs to have the following layout:

        +------------+-+---+-+-------+-+-------+ 
        | properties |;| n | |       | |       | 
        +------------+-+---+-+-------+-+-------+ 
        |  parameter |;| m | |       | |       | 
        +------------+-+---+-+-------+-+-------+ 
        |      A     |;| x |;| v0_Ax |;| v1_Ax |
        +------------+-+---+-+-------+-+-------+ 
        |      A     |;| y |;| v0_Ay |;| v1_Ay |
        +------------+-+---+-+-------+-+-------+ 
        |      A     |;| x |;| v0_Bx |;| v1_Bx |
        +------------+-+---+-+-------+-+-------+ 
        |      A     |;| y |;| v0_By |;| v1_By |
        +------------+-+---+-+-------+-+-------+ 

    where A and B are the maintypes, x and y are the subtypes with the respective
    measured valus v0 and v1 for each kombination of A, B and x, y. Of course
    subtype and maintype are interchangeable, as long as the data is formated
    as described.
    """

    def __init__(self, filename, reference=[0.0, 0.0], weights=[1.0, 1.0],
            annotate='none', stats=False, sort='none', legend='',
            ceil=False, normalize='total', debug=False, cmap='grey', figsize=[10, 5],
            ch_labels=['red', 'green', 'blue'], legend_font={'color': 'w', 'size': 'x-small'}):

        # first of set figure size by parameter
        plt.rcParams["figure.figsize"] = figsize

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

        if not normalize in ('total', 'channels', 'none'):
                raise ValueError("normalize must be 'none', 'total' or 'channels'")

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

        self._marked_samples = {}
        self._markedpatches = []

        # parse file remember the filename of the parsed file
        # self.typnames, self.subnames, self.data = parse_csv(filename)
        #TODO update to new style NxM parser
        parser = Parser(filename)
        self.typnames, self.subnames, self.data = parser.get_matrix_formatted()
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
        # legend font
        self._leg_count_font = legend_font
        # legend labels
        self._ch_labels = ch_labels
        
        # keep track of sorting for csv dump
        self._sorted_by = []        

        # keep maxvals at least
        self._maxvals = [None, None, None]


    def __str__(self):
        """
        Retrun some basic info
        """
        return '{} types with {} subtypes in {} datapoints'.format(self.types, self.subtypes, len(self.data))
    
    def _evaluate(self):
        # magic numbers, always the first two params are choosen. needs to be parameterized
        #TODO slice data accordingly to some slicing parameter
        self.param0 = self.data[:,:1].squeeze()
        self.param1 = self.data[:,1:2].squeeze()
        ci0, ci1 = ['rgb'.index(lc) for lc in self._legendflag]
        for i, params in enumerate(zip(self.param0, self.param1)):
            rgb = evaluate(params, self._weights, self.ref)
            if any(map(lambda x: x < 0, rgb)):
                raise ValueError('Evaluation function can not negative color value!')
            self._matrix[i/self.subtypes, i%self.subtypes][ci0] = rgb[ci0] 
            self._matrix[i/self.subtypes, i%self.subtypes][ci1] = rgb[ci1] 

    def _normalize(self):
        if self._normalizeflag == 'total':
            self._matrix /= np.max(self._matrix)
        elif self._normalizeflag == 'none': 
            pass
        else:
            for i in xrange(3):
                max_val = np.max(self._matrix[:,:,i])
                self._maxvals[i] = max_val
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
            self.cont_dir.text(0, 0, 'DEBUG', color='r')
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

        marked = {}
        i0, i1 = ['rgb'.index(lc) for lc in self._legendflag]
        for row in self._matrix[:-1,:-1,:]:
            for rgb in row:
                i, j = map(lambda v: np.round(v*10.0-1), (rgb[i0], rgb[i1]))
                marked.setdefault((i, j), 0)
                marked[(i, j)] += 1

        for (i, j), val in marked.items():
            # self._legax.text(i+0.2, j+0.5, '*', color='w')
            self._legax.text(i+0.2, j+0.4, str(val), **self._leg_count_font)

    def _plot_legend(self):
        axis = np.linspace(0.1, 1.0, 10)
        names = [str(v) for v in axis]
        leg_matrix = np.zeros((10, 10, 3))
        grid = np.meshgrid(axis, axis)
        for lc, g in zip(self._legendflag, grid):
            i = 'rgb'.index(lc)
            leg_matrix[:,:,i] = g

        #TODO very messy labelstuff
        cidx = ['rgb'.index(lc) for lc in self._legendflag]
        xlabel = self._ch_labels[cidx[0]]
        ylabel = self._ch_labels[cidx[1]]
        self._legax.set_xlabel(xlabel)
        self._legax.set_ylabel(ylabel)

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
            self.fig.canvas.set_window_title('binderfinder Matrix {} -- {}'.format(__version__, self.filename))
            self.fig.subplots_adjust(top=0.90)
            # self.fig.suptitle(self.filename)
            plt.figtext(0.07, 0.03, self.filename)

        self._matimg = self._matax.imshow(self._matrix, interpolation='none')
        self._heatimg = self._heatax.imshow(np.zeros(self._matrix.shape[:2]),
                                            interpolation='none', vmin=0, vmax=1, cmap=self._heatmap_color)

        plt.colorbar(mappable=self._heatimg, ax=self._heatax)
        self._update_matrixdata()

        self._set_ticks(self.subnames, self.typnames)

        # self._check_color = RadioButtons(self.cont_rgb, ('R', 'G', 'B', 'mean'), (False, False, True))
        self._check_dir = RadioButtons(self.cont_dir, ('row', 'col', 'both'))
        self._event_handler = EventHandler(self.fig, self, debug=self._debugflag)

        # spacer between stats and data
        self._plot_spacer()

        # remove spines
        for ax in (self._matax, self._heatax):
            # for line in ('top', 'bottom', 'left', 'right'):
            #     ax.spines[l].set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    def _plot_spacer(self):
        spacer_color = self.fig.get_facecolor()
        offset = 1.45
        hval, vval, _ = [val-offset for val in self._matrix.shape]

        for ax in (self._matax, self._heatax):
            ax.axvline(vval, lw=5, c=spacer_color)
            ax.axhline(hval, lw=5, c=spacer_color)

    def _get_heat(self):
        heat = np.asarray([[rgb_to_illumination(rgb) for rgb in row] for row in self._matrix])
        return heat / np.max(heat)

    def _update_matrixdata(self):
        self._matimg.set_data(self._matrix)
        self._heatimg.set_data(self._get_heat())
        self._set_ticks(self.subnames, self.typnames)
        self.fig.canvas.draw()

    def save_last_run(self):
        """Saves the calculated Matrix and transfomration

        Saves the matrix as png image and the transformed
        data at csv file. The files are written into the
        directory of the data csv file used for the Matrix
        calculation.
        """
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
        self._dumpsorting(os.path.join(path, fname) + '_sorting.txt')

    def _dumpsorting(self, fname):
        with open(fname, 'w') as df:
            df.write('FILE:{}\n'.format(self.filename))
            df.write('VERSION:{}\n\n'.format(__version__))
            for at, idx in self._sorted_by:
                df.write('{};{}\n'.format(at, idx))

    def _dump2csv(self, fname):

        param_names = []
        for lc in self._legendflag:
            i = 'rgb'.index(lc)
            param_names.append(self._ch_labels[i])

        with open(fname, 'w') as df:
            
            # generic info

            df.write('FILE:{}\n'.format(self.filename))
            df.write('VERSION:{}\n\n'.format(__version__))

            # make a header
            h0 = ''
            h1 = ''
            for i in xrange(2 * self.subtypes):
                h0 += ';{}'.format(self.subnames[i/2])
                h1 += ';{}'.format(param_names[i%2])
                if i%2:
                    h0 += ';' + h0.split(';')[-1]
                    h1 += ';marked(col,row)'


            df.write(h0 + '\n' + h1 + '\n')
            
            ci0, ci1 = ['rgb'.index(lc) for lc in self._legendflag]
            # cropt the stats
            if self._statsflag:
                lin_matrix = self._matrix[:-1,:-1].ravel().squeeze()
            else:
                lin_matrix = self._matrix.ravel().squeeze()

            lin_matrix = lin_matrix.reshape(lin_matrix.size/3, 3)

            for i, rgb in enumerate(lin_matrix):
                if i%self.subtypes == 0:
                    df.write('{}'.format(self.typnames[i/self.subtypes]))
                
                leg_key = (i/self.subtypes, i%self.subtypes)
                if self._marked_samples.get(leg_key, {}) != {}:
                    marked = '!' + str(self._marked_samples[leg_key]['col,row'])
                else:
                    marked = ''
                df.write(';{};{};{}'.format(rgb[ci0] * self._maxvals[ci0], rgb[ci1] * self._maxvals[ci1], marked))

                if i%self.subtypes == self.subtypes-1:
                    df.write('\n')

    def _calc_stats(self):
        for c in xrange(self.subtypes):
            col = self._matrix[:-1,c,:].squeeze()
            self._matrix[-1,c] = stats_calculation(col)

        for r in xrange(self.types):
            row = self._matrix[r,:-1,:].squeeze()
            self._matrix[r,-1] = stats_calculation(row)

    def _cleanup_sortby(self):
        if len(self._sorted_by) >= 6:
            both_c = reduce(lambda a, b: a if a == b else (), self._sorted_by[-5::2])
            both_r = reduce(lambda a, b: a if a == b else (), self._sorted_by[-6::2])
            if both_r != () and both_c != ():
                self._sorted_by = self._sorted_by[:-6]
                title = '{},{}'.format(both_r[0], both_c[0])
                idx = (both_r[1], both_c[1])
                self._sorted_by.append((title, idx))

    def _sort_row(self, colidx):
        new_matrix = self._matrix.copy()
        new_subnames = []

        self._sorted_by.append(('row', colidx))

        for n, i in enumerate(np.argsort(sort_reduction(self._matrix[colidx,:-1].squeeze()))):
            new_matrix[:,n] = self._matrix[:,i]
            new_subnames.append(self.subnames[i])

        self._matrix = new_matrix
        self.subnames = new_subnames
        if self._annotateflag == 'all':
            self.subnames += ['stats']
        self._cleanup_sortby()

    def _sort_col(self, rowidx):
        new_matrix = self._matrix.copy()
        new_typnames = []

        self._sorted_by.append(('col', rowidx))

        for n, i in enumerate(np.argsort(sort_reduction(self._matrix[:-1,rowidx].squeeze()))):
            new_matrix[n,:] = self._matrix[i,:]
            new_typnames.append(self.typnames[i])

        self._matrix = new_matrix
        self.typnames = new_typnames
        if self._annotateflag == 'all':
            self.typnames += ['stats']
        self._cleanup_sortby()

    def _sort_matrix(self):
        if self._sortflag in ('both', 'col'):
            for step in xrange(3):
                self._sort_col(-1)

            if self._sortflag in ('both', 'row'):
                self._sort_row(-1)

    def run(self, show=False):
        """Runs the transformation and the Matrix creation.

        Parameters
        ----------
            show : {True, False}
              True:
                Plot and show the matrix directly after computation.
              False:
                Only calculate, but do not plot the matrix
        """
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
