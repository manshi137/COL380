#ifndef BFS_HPP
#define BFS_HPP

#include <vector>

int randInt(int min, int max);
void bfs(std::vector<std::vector<int>>& maze, int entryRow, int entryCol, int exitRow, int exitCol);
#endif // BFS_HPP
