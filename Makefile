CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic
CXXFLAGS_DEBUG=$(CXXFLAGS) -g3 -O0
CXXFLAGS_SANITIZE=$(CXXFLAGS) -O0 -g3 \
				  -fsanitize=address,undefined -fno-omit-frame-pointer \
				  -fno-optimize-sibling-calls
CXXFLAGS_RELEASE=$(CXXFLAGS) -Ofast -march=native -DNDEBUG -g3
INC=-I spqlios/
LIB=-L spqlios/build -lspqlios

main: main.cpp aqtfhe2.hpp spqlios
	#clang++ $(CXXFLAGS_SANITIZE) -o $@ $< $(INC) $(LIB)
	#clang++ $(CXXFLAGS_DEBUG) -o $@ $< $(INC) $(LIB)
	clang++ $(CXXFLAGS_RELEASE) -o $@ $< $(INC) $(LIB) #-lprofiler

spqlios: FORCE
	mkdir -p spqlios/build
	cd spqlios/build && \
		cmake \
			-DCMAKE_C_COMPILER="gcc" \
			-DCMAKE_C_FLAGS="-march=native -Ofast -DNDEBUG" \
			-DCMAKE_CXX_COMPILER="g++" \
			-DCMAKE_CXX_FLAGS="-march=native -Ofast -DNDEBUG" \
			-DCMAKE_BUILD_TYPE=Release \
			.. && \
		make

FORCE:
