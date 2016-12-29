.PHONY:
deploy: merge update_rev commit

update_rev:
	sed -i -- 's/^__hgrev__.*/__hgrev__ = $(REVISION)/g' ./binderfinder/__init__.py

REVISION = $(patsubst %+,%,$(shell hg id -r release -n))
commit:
	hg addremove
	hg commit -m "merged from default"

merge:
	hg up release
	hg merge default
