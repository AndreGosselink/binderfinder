REVISION = $(shell hg identify --num)

all:
	sed -i -- 's/rev.*\" # revision mark/rev $(REVISION)\" # revision mark/g' ./binderfinder/__init__.py
	ipython binderfinder.py

commit:
	hg addremove
	hg commit
	hg push

update_rev:
	sed -i -- 's/rev.*\" # revision mark/rev $(REVISION)\" # revision mark/g' ./binderfinder/__init__.py


.PHONY: deploy
deploy: commit update_rev
