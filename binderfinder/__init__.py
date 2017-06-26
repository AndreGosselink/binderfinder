"""
Copyright (c) Andre Gosselink, Niels Werchau

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    All advertising materials mentioning features or use of this software must display the following acknowledgement: "This product includes software developed by Andre Gosselink and Niels Werchau"
    Neither the name of the Authors nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__hgrev__ = 131
__version__ = "1.34 rev {}".format(__hgrev__+1)

from matrix import Matrix
from pca import PCA

import os
import sys
import subprocess as sub

import matplotlib as mpl


def start_binderfinder(defaults):
    m = Matrix(**defaults)
    m.show_me_where_the_white_rabbit_goes()
    m.save_last_run()

if not '-noconsole' in sys.argv:
    
    try:
        hg = sub.Popen('hg branch', stdout=sub.PIPE)
        branch = hg.stdout.read()
        if not ('default' in branch or 'release' in branch):
            branch = ''
        hg.kill()
    except:
        branch = ''

    print 'starting binderfinder ' + __version__  + ' ' + branch
    print 'using matplotlib ' + mpl.__version__
    print 'started with pid', os.getpid()
