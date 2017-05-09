import binderfinder as bf
from binderfinder.matrix import Matrix
import sys
import os

def start_binderfinder(defaults):
    m = Matrix(**defaults)
    m.show_me_where_the_white_rabbit_goes()
    m.save_last_run()

defaults = { 'filename': './data/mock_data_rnd.csv',
            'reference': [100, 100],
              'weights': [1.0, 1.0],
             'annotate': 'none',
                'stats': True,
                 'sort': 'none',
               'legend': 'bg',
                 'ceil': True,
            'normalize': 'channels',
                'debug': False,
                 'cmap': 'jet',
              'figsize': [13, 6],
            'ch_labels':  ['red', 'Rnd', 'Rnd'],
          'legend_font': {'color': 'r',
                          'size': 'x-small',
                         },
           }

if not '-noconsole' in sys.argv:
    print "starting binderfinder " + bf.__version__  + '\n'
    print 'started with pid', os.getpid()
    try:
        os.system('hg branch')
    except:
        pass

start_binderfinder(defaults)


