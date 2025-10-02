CXX=g++

ifeq ($(OS),Windows_NT)
OCV_I1PATH=/opencv/build/install/include
OCV_I2PATH=$(OCV_I1PATH)/opencv2
OCV_lPATH=/opencv/build/install/x64/mingw/bin
CXXFLAGS=-I$(OCV_I1PATH) -I$(OCV_I2PATH)
CXXLINKS=$(wildcard $(OCV_lPATH)/*.dll)
else
CXXFLAGS = `pkg-config --cflags --libs opencv4`
endif

PROJECTROOT=code
SRCDIR=$(PROJECTROOT)/src
BUILDDIR=$(PROJECTROOT)/build
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

access:
	$(CXX) $(STESTDIR)/AccesoPixel.cxx  $(PDICXX)  $(CXXLINKS) $(CXXFLAGS) -o $(BTESTDIR)/accesoPixel
	cd $(BTESTDIR) && ./accesoPixel
