from binderfinder import start_binderfinder


# defaults = { 'filename': './data/mock_data_rnd.csv',
defaults = { 'filename': './data/iris_dataset/iris.data',
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


start_binderfinder(defaults)


