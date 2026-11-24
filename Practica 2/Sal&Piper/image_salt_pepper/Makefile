
CXX = icpx

CXXFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda 

LIBS=


main: main.cpp kernels.cpp 
	$(CXX) $^ $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f main
