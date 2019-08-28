.PHONY: all clean site-clean site test

MODS := Main.x	\
	Setup.x	\
	LinkMetadata.x	\
	Inflation.x

all: $(MODS:.x=.hs)
	@cabal v2-build

clean:
	@cabal v2-clean

site-clean: all
	@cabal v2-run wiki -- clean

site: all
	@cabal v2-run wiki -- build
