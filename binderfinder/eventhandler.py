import numpy as np
import os
import matplotlib.pyplot as plt

class EventHandler(object):

    def __init__(self, fig, other, debug=False):
        self.motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.press = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.other = other
        self.other._check_dir.on_clicked(self.update_dir)
        self.other._sortflag = 'none'
        self._debugflag = debug
        self._over_legend = {}
        if debug == True:
            self._logfile = open('logfile.txt', 'w')
            self._logfile.write(str(os.name) + '\n\n')
            self._logfile.flush()

    def catch(f, *args, **kwargs):
        def _logged(self, event):
            if self._debugflag:
                self._logfile.write(str(event) + '\n')
                self._logfile.flush()
            return f(self, event)

        return _logged

    def _mark_legend(self, event):
        col, row = map(lambda x: int(np.round(x)), (event.xdata, event.ydata))
        cidx = ['rgb'.index(lc) for lc in self.other._legendflag]
        vals = [self.other._matrix[row, col, rgb]*10 for rgb in cidx]
        x, y = map(np.ceil, vals)
        
        self.other._legpatch.set_xdata([x-1.5, x-0.5, x-0.5, x-1.5, x-1.5])
        self.other._legpatch.set_ydata([y-1.5, y-1.5, y-0.5, y-0.5, y-1.5])

    def _mark_matrix(self, event):
        col, row = map(lambda x: int(np.round(x)), (event.xdata, event.ydata))
        cval, rval = map(lambda v: (v + 1)/10.0, (col, row))
        
        to_draw = []
        i0, i1 = ['rgb'.index(lc) for lc in self.other._legendflag]
        self._over_legend = {}
        for j, row in enumerate(self.other._matrix):
            for i, rgb in enumerate(row):
                if rgb[i0] == cval and rgb[i1] == rval:
                    self._over_legend[(j, i)] ={'colname': self.other.subnames[i],
                                                'rowname': self.other.typnames[j],
                                                'col,row': (cval, rval),
                                                'xylegend': map(lambda x: int(np.round(x)), (event.xdata, event.ydata)),
                                               }
                    to_draw.append([(i+1, j+1), rgb])
        
        while len(to_draw) > len(self.other._matpatches):
            self.other._matpatches.append(self.other._matax.plot([], [], c='r', lw=2.2) +
                                          self.other._heatax.plot([], [], c='r', lw=2.2)
                                         )
        
        # set new patches
        for ((x, y), rgb), lines in zip(to_draw, self.other._matpatches):
            for l in lines:
                l.set_xdata([x-1.5, x-0.5, x-0.5, x-1.5, x-1.5])
                l.set_ydata([y-1.5, y-1.5, y-0.5, y-0.5, y-1.5])
                # l.set_color(1 - rgb)

        # set the rest to empty
        if len(to_draw) < len(self.other._matpatches):
            for lines in self.other._matpatches[len(to_draw):]:
                for l in lines:
                    l.set_data([], [])

    def _set_marked(self):
        to_draw = [v['xylegend'] for v in self.other._marked_samples.values()]
        while len(to_draw) > len(self.other._markedpatches):
            self.other._markedpatches.append(self.other._legax.plot([], [], c='m', lw=1.2, ls=':')[0])

        # set new patches
        for (x, y), l in zip(to_draw, self.other._markedpatches):
            l.set_xdata([x-0.25, x+0.25])
            l.set_ydata([y-0.25, y+0.25])

        # set the rest to empty
        if len(to_draw) < len(self.other._markedpatches):
            for l in self.other._markedpatches[len(to_draw):]:
                l.set_data([], [])

    @catch
    def on_motion(self, event):

        if event.inaxes == self.other._matax or event.inaxes == self.other._heatax:
            self._mark_legend(event)
        else:
            try:
                self.other._legpatch.set_data([], [])
            except:
                pass

        if event.inaxes == self.other._legax:
            self._mark_matrix(event)
        else:
            try:
                for lines in self.other._matpatches:
                    for l in lines:
                        l.set_data([], [])
            except:
                pass

        self.other.fig.canvas.draw()

    @catch
    def on_click(self, event):
        if event.inaxes == self.other._matax:
            self._clicked_matrix(event)
        elif event.inaxes == self.other._legax:
            self._clicked_legend(event)
            self.other.fig.canvas.draw()
    
    def _clicked_legend(self, event):
        for k, v in self._over_legend.items():
            marked = self.other._marked_samples.get(k, ())
            if marked == ():
                self.other._marked_samples[k] = v 
                print 'adding', v
            else:
                print 'removing', v
                _ = self.other._marked_samples.pop(k)
        self._set_marked()

    def _clicked_matrix(self, event):
        col, row = map(lambda x: int(np.round(x)), (event.xdata, event.ydata))

        if self.other._sortflag == 'row':
            self.other._sort_row(row)
        elif self.other._sortflag == 'col':
            self.other._sort_col(col)
        elif self.other._sortflag == 'both':
            for i in xrange(3):
                self.other._sort_row(row)
                self.other._sort_col(col)
        elif self.other._sortflag == 'none':
            # if non, set to row automatically, where row is the default option
            self.other._sortflag = 'row'
            return self.on_click(event)
        else:
            raise ValueError('Its a bug, not a feature...')

        self.other._update_matrixdata()

    def update_dir(self, label):
        self.other._sortflag = label
