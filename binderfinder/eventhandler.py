import numpy as np
import matplotlib.pyplot as plt

class EventHandler(object):

    def __init__(self, fig, other):
        self.motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.press = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.other = other
        self.other._check_dir.on_clicked(self.update_dir)
        self.other._sortflag = 'none'

    def on_motion(self, event):
        if event.inaxes != self.other.ax:
            try:
                self.other._legpatch.set_xdata([])
                self.other._legpatch.set_ydata([])
                self.other._legpatch.figure.canvas.draw()
                return
            except:
                return

        col, row = map(lambda x: int(np.round(x)), (event.xdata, event.ydata))
        cidx = ['rgb'.index(lc) for lc in self.other._legendflag]
        vals = [self.other._matrix[row, col, rgb]*10 for rgb in cidx]
        x, y = map(np.ceil, vals)
        
        self.other._legpatch.set_xdata([x-1.5, x-0.5, x-0.5, x-1.5, x-1.5])
        self.other._legpatch.set_ydata([y-1.5, y-1.5, y-0.5, y-0.5, y-1.5])
        self.other._legpatch.figure.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.other.ax:
            return
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
            self.other._sortflag = 'row'
            return self.on_click(event)
        else:
            raise ValueError('Its a bug, not a feature...')
        self.other._img.set_data(self.other._matrix)
        self.other.ax.set_xticklabels(self.other.subnames)
        self.other.ax.set_yticklabels(self.other.typnames)
        self.other._img.figure.canvas.draw()

    def update_dir(self, label):
        self.other._sortflag = label
