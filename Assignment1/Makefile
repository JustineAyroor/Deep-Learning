CXX = nvcc
CXXFLAGS2 = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dotPro

all : $(TARGET1)
    
$(TARGET1) : dot_prod.cu
	$(CXX) -lm -o $(TARGET1) dot_prod.cu
clean : 
	rm -f $(TARGET1)
