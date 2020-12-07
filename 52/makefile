#Sources
SHR_CSOURCES=io_png.c randmt.c
SPLE_CXXSOURCES=util.cpp SPLE_lib.cpp SPLE_denoise.cpp DataProvider.cpp
PLE_CXXSOURCES=util.cpp PLE_lib.cpp PLE_denoise.cpp DataProvider.cpp
SAMPLE_CXXSOURCES=util.cpp imsample.cpp DataProvider.cpp
IMDIFF_SOURCES=imdiff.c conv.c imageio.c basic.c
IMNOISE_SOURCES=imnoise.c randmt.c imageio.c basic.c
#Objects
SHR_COBJ=$(SHR_CSOURCES:.c=.o)
PLE_CXXOBJ=$(PLE_CXXSOURCES:.cpp=.o)
SPLE_CXXOBJ=$(SPLE_CXXSOURCES:.cpp=.o)
SAMPLE_CXXOBJ=$(SAMPLE_CXXSOURCES:.cpp=.o)
PLE_OBJ=$(SHR_COBJ) $(PLE_CXXOBJ)
SPLE_OBJ=$(SHR_COBJ) $(SPLE_CXXOBJ)
SAMPLE_OBJ=$(SHR_COBJ) $(SAMPLE_CXXOBJ)
IMNOISE_OBJECTS=$(IMNOISE_SOURCES:.c=.o)
IMDIFF_OBJECTS=$(IMDIFF_SOURCES:.c=.o)
OBJ=$(PLE_OBJ) $(SPLE_OBJ) $(SAMPLE_OBJ) $(IMNOISE_OBJECTS) $(IMDIFF_OBJECTS)
#Executables
EXEC=denoiseSPLE denoisePLE imdiff imnoise imsample 

# C optimization flags DNDEBUG accelerates Eigen
COPT=-O2 -ftree-vectorize -funroll-loops -DNDEBUG 
# C compilation flags
CFLAGS=$(COPT) -Wall -Wextra -Wno-write-strings -ansi
# C++ compilation flags
CXXFLAGS=$(COPT) 
# link flags (-L tells where the library lnewmat is)
LDFLAGS=-lpng -L./newmat10 -lnewmat -lm
LDLIB=-lm -lpng -lz

# use openMP with `make OMP=1`
ifdef OMP
CFLAGS+=-fopenmp
CXXFLAGS+=-fopenmp
LDFLAGS+=-lgomp
else
CFLAGS+=-Wno-unknown-pragmas
CXXFLAGS+=-Wno-unknown-pragmas
endif

#included files search directories added
CXXFLAGS+= -I./eigen -I./newmat10

all: $(EXEC)

# link all the opject code
libnewmat.a:
	$(MAKE) -C newmat10 -f nm_gnu.mak

denoiseSPLE: $(SPLE_OBJ) libnewmat.a 
	$(CXX) $(SPLE_OBJ) $(LDFLAGS) -o $@ 

denoisePLE: $(PLE_OBJ) libnewmat.a 
	$(CXX) $(PLE_OBJ) $(LDFLAGS) -o $@ 

imsample: $(SAMPLE_OBJ) libnewmat.a 
	$(CXX) $(SAMPLE_OBJ) $(LDFLAGS) -o $@ 

imnoise: $(IMNOISE_OBJECTS)
	$(CC) $(IMNOISE_OBJECTS) $(LDLIB) -o $@

imdiff: $(IMDIFF_OBJECTS)
	$(CC) $(IMDIFF_OBJECTS) $(LDLIB) -o $@


#build objects
%.o: %.c %.h
	$(CC) -o $@ -c $< $(CFLAGS)
%.o: %.cpp %.h
	$(CXX) -o $@ -c $< $(CXXFLAGS)

# housekeeping
.PHONY	: clean
clean	:
	$(RM) $(OBJ)
srcdoc:
	doxygen doxygen.conf
