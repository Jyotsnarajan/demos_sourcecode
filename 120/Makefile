# Copyright 2010-2011 Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
# Copyright 2014 Jacques Froment <Jacques.Froment@univ-ubs.fr>
#
# Copying and distribution of this file, with or without
# modification, are permitted in any medium without royalty provided
# the copyright notice and this notice are preserved.  This file is
# offered as-is, without any warranty.

# source code
NLMPSRC		= src/NLMeansP.c src/nlmp.c src/io_png.c src/mt19937ar.c src/libauxiliary.c
MSESRC	= src/io_png.c src/img_mse_ipol.c 
NOISRC	= src/mt19937ar.c src/libauxiliary.c src/addgaussnoise.c
DIFFSRC = src/img_diff_ipol.cpp

SRC	= $(MSESRC) $(NOISRC) $(DIFFSRC)

# object files (partial compilation)
NLMPOBJ	= $(NLMPSRC:.c=.o)
MSEOBJ	= src/io_png.o src/img_mse_ipol.o
NOIOBJ	= src/io_png.o src/mt19937ar.o src/libauxiliary.o src/addgaussnoise.o
DIFFOBJ = src/io_png.o src/img_diff_ipol.o

# binary executable programs
BIN	= NLMeansP img_mse_ipol addgaussnoise img_diff_ipol

# C compiler options
CFLAGS	= -std=c89 -pedantic -Wall -Wextra
# C++ compiler options
CXXFLAGS= -ansi -Wall -Wextra -Wno-write-strings -Wno-deprecated
# linker options
LDFLAGS	= -lpng -lm

#  OpenMP (Open Multi-Processing) with `make OMP=1`
ifdef OMP
CFLAGS	+= -fopenmp
LDFLAGS += -lgomp
else
CFLAGS	+= -Wno-unknown-pragmas
endif

# default target: the programs
default: $(BIN)

# partial C compilation xxx.c -> xxx.o
%.o	: %.c %.h
	$(CC) -c $< $(CFLAGS) -o $@

# partial compilation of C++ source code
%.o: %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) -o $@

# final link
NLMeansP: 	$(NLMPOBJ)
		$(CC) $^ $(LDFLAGS) -o $@

img_mse_ipol	: $(MSEOBJ)
		$(CC) $^ $(LDFLAGS) -o $@

addgaussnoise	: $(NOIOBJ)
		$(CC) $^ $(LDFLAGS) -o $@

img_diff_ipol   : $(DIFFOBJ)
		$(CXX) $^ $(LDFLAGS) -o $@ 

# cleanup
clean	:
	$(RM) $(MSEOBJ) $(NOIOBJ) $(NLMPOBJ) $(DIFFOBJ)
distclean	: clean
	$(RM) $(BIN) *~ src/*~
	$(RM) -r doc/html

.PHONY	: clean distclean

################################################
# extra non-essential tasks

# strict compilation options for gcc
CSTRICT	= -std=c89 -pedantic -Wall -Wextra
# optimized compilation options for gcc
COPT	= -O3 -ffast-math -march=native -funroll-loops -fomit-frame-pointer

# optimized build
opti	: $(SRC)
	$(MAKE) -B CFLAGS="$(CFLAGS) $(COPT)"

# debugging build
debug	: $(SRC)
#	$(MAKE) -B CFLAGS="$(CFLAGS) -g" LDFLAGS="$(LDFLAGS) -lefence"
	$(MAKE) -B CFLAGS="$(CFLAGS) -g" 

# source documentation
doc	: $(SRC)
	doxygen doc/doxygen.conf

# static code analysis and strict build
lint	: $(SRC)
	clang --analyze -ansi $^
	splint -ansi-lib -weak $^
	$(RM) *.plist
	$(MAKE) -B CFLAGS="$(CFLAGS) $(CSTRICT)"
	@echo OK

# code cleanup
tidy	: $(SRC)
	for FILE in $^; do \
		expand $$FILE | sed 's/[ \t]*$$//' > $$FILE.$$$$ \
		&& indent -kr -i4 -l78 -nut -nce -sob -sc \
			$$FILE.$$$$ -o $$FILE \
		&& rm $$FILE.$$$$; \
	done

.PHONY	: opti debug doc lint tidy
