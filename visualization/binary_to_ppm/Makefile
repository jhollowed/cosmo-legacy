C 		= c++
C_COMPILE	= -c
C_OPT 		= -O

.C.o: 
	$(C) $(C_COMPILE) $*.C

colormap: colormap.o
	$(C) $(C_OPT) -L$(LIBDIR) colormap.o -o colormap colormap.C	

clean:
	rm *.o 
