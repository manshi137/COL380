make -> maze.cpp ->maze.out


mpic++ -o bfs bfs.cpp
mpirun -np 4 ./bfs

mpic++ -o kruskal kruskal.cpp
mpirun -np 4 ./kruskal

mpic++ -o dfs dfs.cpp
mpirun -np 4 ./dfs

mpic++ -o dijkstra dijkstra.cpp
mpirun -np 4 ./dijkstra


mpic++ -o maze.out maze.cpp
mpirun -np 4 ./maze.out -g kruskal -s dfs



