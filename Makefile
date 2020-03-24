CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic
CXXFLAGS_DEBUG=$(CXXFLAGS) -g3 -O0
CXXFLAGS_SANITIZE=$(CXXFLAGS) -O0 -g3 \
				  -fsanitize=address,undefined -fno-omit-frame-pointer \
				  -fno-optimize-sibling-calls
CXXFLAGS_RELEASE=$(CXXFLAGS) -Ofast -march=native -DNDEBUG #-g3
INC=
LIB=

main: main.cpp aqtfhe2.hpp
	#clang++ $(CXXFLAGS_SANITIZE) -o $@ $< $(INC) $(LIB)
	#g++ $(CXXFLAGS_DEBUG) -o $@ $< $(INC) $(LIB)
	g++ $(CXXFLAGS_RELEASE) -o $@ $< $(INC) $(LIB) #-lprofiler
