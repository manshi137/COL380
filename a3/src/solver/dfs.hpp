#ifndef DFS_HPP
#define DFS_HPP

#include <vector>
using namespace std;
#include <utility>

class MazeSolver  {
public:
    vector<pair<int, int>> solveMaze(vector<vector<char>>& maze);
private:
    bool dfs(vector<vector<char>>& maze, int x, int y, vector<pair<int, int>>& path);
    bool isValid(vector<vector<char>>& maze, int x, int y);
};


#endif // DFS_HPP