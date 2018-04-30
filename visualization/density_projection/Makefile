# On jupiter, first run module load mvapich2-2.2b-gcc-5.3.0-o4of6w7
CXX := mpicxx
CXXFLAGS := -O3 -fopenmp

SRCDIR := ./src
OBJDIR := ./obj
OBJECTS := $(OBJDIR)/ProjectDensity.o $(OBJDIR)/Particles.o $(OBJDIR)/ChainingMesh.o $(OBJDIR)/CloudsInCells.o $(OBJDIR)/Skewer.o

INCLUDES := -I $(SRCDIR)

all: densityProjection

# linking
#
densityProjection: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# compilation
#
obj/Particles.o: $(SRCDIR)/Particles.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/Particles.cxx  -o $(OBJDIR)/Particles.o

obj/ChainingMesh.o: $(SRCDIR)/ChainingMesh.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/ChainingMesh.cxx  -o $(OBJDIR)/ChainingMesh.o

obj/CloudsInCells.o: $(SRCDIR)/CloudsInCells.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/CloudsInCells.cxx  -o $(OBJDIR)/CloudsInCells.o

obj/Skewer.o: $(SRCDIR)/Skewer.cxx 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/Skewer.cxx  -o $(OBJDIR)/Skewer.o

obj/ProjectDensity.o: $(SRCDIR)/ProjectDensity.cxx $(SRCDIR)/Particles.h $(SRCDIR)/ChainingMesh.h $(SRCDIR)/CloudsInCells.h $(SRCDIR)/Skewer.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SRCDIR)/ProjectDensity.cxx  -o $(OBJDIR)/ProjectDensity.o

clean:
	rm -f densityProjection $(OBJDIR)/*.o 

