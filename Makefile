CXX = g++
CXXFLAGS = -std=c++0x -O2
DEBUG = -g
INCLUDES = -I ./include/

SRC = $(wildcard src/*.cpp)
ALL_OBJS = $(patsubst src/%.cpp, build/%.o, $(SRC))

TEST = $(wildcard test/*.cpp)
TEST_OBJS = $(patsubst test/%.cpp, test/%.o, $(TEST))
TEST_BINS = $(patsubst test/%.cpp, bin/%.out, $(TEST))

# src build
build/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

test/%.o: test/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

bin/%.out: test/%.o $(ALL_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

.PHONY: all

all: $(TEST_BINS)
	rm -rf $(ALL_OBJS) $(TEST_OBJS)
clean: 
	rm -rf $(ALL_OBJS) $(TEST_OBJS) $(TEST_BINS)