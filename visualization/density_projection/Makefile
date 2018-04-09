CXX=g++
CXXFLAGS=-g -O2 -fopenmp
LDLIBS=
LDFLAGS=
TRUNK_LOC=/homes/jphollowed/code/lc_interpolation/jupiter
INCLUDES := -I $(SRCDIR) -I /homes/jphollowed/code/lc_interpolation/genericio
LIBS := $(TRUNK_LOC)/mpi/lib/libGenericIOMPI.a

SRCDIR := ./src
OBJDIR := ./obj
OBJECTS := $(OBJDIR)/ProjectDensity.o $(OBJDIR)/Particles.o $(OBJDIR)/ChainingMesh.o $(OBJDIR)/CloudsInCells.o $(OBJDIR)/Skewer.o

all: densityProjection

densityProjection: $(OBJECTS) $(LIBS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -L$(TRUNK_LOC)/mpi/lib -lGenericIOMPI -L$(TRUNK_LOC)/frontend/lib -lGenericIO -o $@ $^ $(LDLIBS)

obj/ProjectDensity.o: $(SRCDIR)/ProjectDensity.cxx $(SRCDIR)/Particles.h $(SRCDIR)/ChainingMesh.h $(SRCDIR)/CloudsInCells.h $(SRCDIR)/Skewer.h $(LIBS)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/Particles.o: $(SRCDIR)/Particles.cxx $(LIBS)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/ChainingMesh.o: $(SRCDIR)/ChainingMesh.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/CloudsInCells.o: $(SRCDIR)/CloudsInCells.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/Skewer.o: $(SRCDIR)/Skewer.cxx 
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	rm -f densityProjection obj/*.o 
