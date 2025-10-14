CXX=g++

ifeq ($(OS),Windows_NT)
OCV_I1PATH=/opencv/build/install/include
OCV_I2PATH=$(OCV_I1PATH)/opencv2
OCV_lPATH=/opencv/build/install/x64/mingw/bin
CXXFLAGS=-I$(OCV_I1PATH) -I$(OCV_I2PATH) -lm -g
CXXLINKS=$(wildcard $(OCV_lPATH)/*.dll)
else
CXXFLAGS = `pkg-config --cflags --libs opencv4` -lm -g
endif
MY_HEADER=$(wildcard ./code/src/*.hxx)

PROJECTROOT=code
SRCDIR=$(PROJECTROOT)/src
BUILDDIR=$(PROJECTROOT)/build
TAREADIR=$(PROJECTROOT)/tareas
STESTDIR=$(SRCDIR)/test
BTESTDIR=$(BUILDDIR)/test
BINDIR=$(BUILDDIR)/bin
PDICXX=$(SRCDIR)/ProcesamientoDigitalImagenes.cxx

all:
	$(CXX) $(SRCDIR)/ProcesamientoDigitalImagenes.cxx $(SRCDIR)/ProcesamientoDigitalImagenes.hxx  $(CXXLINKS) $(CXXFLAGS) -o $(BINDIR)/exe

holaMundo:
	$(CXX) $(STESTDIR)/HolaMundo.cxx $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/holaMundo
	cd $(BTESTDIR) && ./holaMundo

negativo:
	$(CXX) $(STESTDIR)/Negativo.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/Negativo
	cd $(BTESTDIR) && ./Negativo

access:
	$(CXX) $(STESTDIR)/AccesoPixel.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/accesoPixel
	cd $(BTESTDIR) && ./accesoPixel

minmax:
	$(CXX) $(STESTDIR)/MaxMin.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/minmax
	cd $(BTESTDIR) && ./minmax

dynrngnrm:
	$(CXX) $(STESTDIR)/DynRngNrm.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/DynRngNrm
	cd $(BTESTDIR) && ./DynRngNrm

tarea01:
	$(CXX) $(TAREADIR)/tarea01/main.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(TAREADIR)/tarea01/exe
	cd $(TAREADIR)/tarea01 && ./exe

help:
	@echo "make holaMundo"
	@echo "make negativo"
	@echo "make access"
	@echo "make minmax"
	@echo "make dynrngnrm"
