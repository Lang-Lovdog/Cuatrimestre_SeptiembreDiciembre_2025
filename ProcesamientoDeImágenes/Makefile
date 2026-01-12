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

defa: help

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

thresholding:
	$(CXX) $(STESTDIR)/Thresholding.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/thres
	cd $(BTESTDIR) && ./thres

mthresholding:
	$(CXX) $(STESTDIR)/mThresholding.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/mthres
	cd $(BTESTDIR) && ./mthres

dynrngnrm:
	$(CXX) $(STESTDIR)/DynRngNrm.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/DynRngNrm
	cd $(BTESTDIR) && ./DynRngNrm

tarea01:
	$(CXX) $(TAREADIR)/tarea01/main.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(TAREADIR)/tarea01/exe
	cd $(TAREADIR)/tarea01 && ./exe

tarea02:
	$(CXX) $(TAREADIR)/tarea02/main.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(TAREADIR)/tarea02/exe
	cd $(TAREADIR)/tarea02 && ./exe

tarea03:
	$(CXX) $(TAREADIR)/tarea03/main.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(TAREADIR)/tarea03/exe
	cd $(TAREADIR)/tarea03 && valgrind ./exe

help:
	@echo "make holaMundo"
	@echo "make negativo"
	@echo "make access"
	@echo "make minmax"
	@echo "make dynrngnrm"
	@echo "make tarea01"
	@echo "make tarea02"
	@echo "make tarea03"
