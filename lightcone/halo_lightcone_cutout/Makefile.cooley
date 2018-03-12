CPP := mpicxx
CPPFLAGS := -O3
SRCDIR := ./src
OBJDIR := ./obj
OBJECTS := $(OBJDIR)/main.o
TRUNK_LOC := /home/hollowed/repos/lc_interpolation/cooley
INCLUDES := -I $(SRCDIR) -I /home/hollowed/repos/lc_interpolation/genericio
LIBS := $(TRUNK_LOC)/mpi/lib/libGenericIOMPI.a

#linking
halo_lightcone_cutout : $(OBJECTS) $(LIBS)
	$(CPP) $(CPPFLAGS) $(OBJECTS) -L$(TRUNK_LOC)/mpi/lib -lGenericIOMPI -L$(TRUNK_LOC)/frontend/lib -lGenericIO -o halo_lc_cutout -fopenmp

#compilation
$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(LIBS)
	$(CPP) $(CPPFLAGS) $(INCLUDES) -c $(SRCDIR)/main.cpp -o $(OBJDIR)/main.o -fopenmp

clean:
	rm $(OBJDIR)/*.o halo_lc_cutout
