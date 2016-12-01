REVISION = $(shell hg identify --num)

all:
	sed -i -- 's/rev.*\\n\" # revision mark/rev $(REVISION)\\n\" # revision mark/g' ./binderfinder.py
	ipython binderfinder.py
