import subprocess as sub
import sys

print "starting binderfinder v0.91 rev 8+\n" # revision mark

p = sub.Popen(['pythonw', 'main.py'])

print 'started with pid', p.pid

sys.exit(p.wait())
