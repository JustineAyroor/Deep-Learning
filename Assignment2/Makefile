CXX = icc
CXXFLAGS = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dot_prod

all : $(TARGET1)
    
$(TARGET1) : dot_prod.c
	$(CXX) $(CXXFLAGS) -o $(TARGET1) dot_prod.c -lm -fopenmp
clean : 
	rm -f $(TARGET1)
