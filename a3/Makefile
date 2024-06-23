.PHONY: all clean

all: 
	mpic++ -o maze.out ./src/maze.cpp
	g++ -o mazegen.out ./src/generator/mazegenerator.cpp
	mpic++ -o bfs.out ./src/generator/bfs.cpp
	mpic++ -o kruskal.out ./src/generator/kruskal.cpp
	g++ -o mazesolve.out ./src/solver/mazesolver.cpp
	mpic++ -o dfs.out ./src/solver/dfs.cpp
	mpic++ -o dijkstra.out ./src/solver/dijkstra.cpp\
	
clean:
	rm -f maze.out mazegen.out mazesolve.out bfs.out kruskal.out dfs.out dijkstra.out maze.txt output.txt

# Default target


