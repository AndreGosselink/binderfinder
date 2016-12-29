REVISION=
.PHONY:
deploy: merge update_rev commit

update_rev: get_rev
	sed -i -- 's/^__hgrev__.*/__hgrev__ = $(REVISION)/g' ./binderfinder/__init__.py

commit:
	hg addremove
	hg commit -m "merged from default"

merge:
	hg up release
	hg merge default

.PHONY:
get_rev:
	REVISION+=$(patsubst %+,%,$(shell hg id -r release -n))
