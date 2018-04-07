CXX=g++
CXXFLAGS=-g -O2 -fopenmp
LDLIBS=
LDFLAGS=

all: runDensity

runDensity: ProjectDensity.o Particles.o ChainingMesh.o CloudsInCells.o Skewer.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

ProjectDensity.o: ProjectDensity.cxx Particles.h ChainingMesh.h CloudsInCells.h Skewer.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

Particles.o: Particles.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $<

ChainingMesh.o: ChainingMesh.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $<

CloudsInCells.o: CloudsInCells.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $<

Skewer.o: Skewer.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	rm -f runDensity *.o 

