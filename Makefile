.PHONY:
deploy: merge update_rev commit

update_rev:
	REVISION = $(patsubst %+,%,$(shell hg id -r release -n))
	sed -i -- 's/^__hgrev__.*/__hgrev__ = $(REVISION)/g' ./binderfinder/__init__.py

commit:
	hg addremove
	hg commit -m "merged from default"

merge:
	hg up release
	hg merge default
