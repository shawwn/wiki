.PHONY: all clean site-clean site test

MODS := Main.x	\
	Setup.x	\
	LinkMetadata.x	\
	Inflation.x

all: $(MODS:.x=.hs)
	@cabal v2-build

clean:
	@cabal v2-clean
	@rm -f Main wiki *.hi *.o

site-clean: all
	@cabal v2-run wiki -- clean

dist-clean: clean
	@rm -rf _site _cache

site: all
	@./build.sh
