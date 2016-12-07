from binderfinder import Matrix
import sys
import os

if not '-noconsole' in sys.argv:
    print "starting binderfinder v0.91 rev 13+\n" # revision mark
    print 'started with pid', os.getpid()


# help(Matrix)
m = Matrix('./data/mock_data_rnd.csv', reference=[100, 100], weights=[0.05, 1],
        annotate='none', stats=True, sort='none', legend='bg', ceil=True,
        normalize='channels', debug=False)

m.show_me_where_the_white_rabbit_goes()
m.save_last_run()
