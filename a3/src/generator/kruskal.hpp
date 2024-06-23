#ifndef KRUSKAL_HPP
#define KRUSKAL_HPP

#include <vector>
using namespace std;
#include <utility>

class DisjointSet {
public:
    std::vector<std::vector<std::pair<int, int>>> parent;
    std::vector<std::vector<int>> rank;

    DisjointSet(int rows, int cols);

    std::pair<int, int> find(const std::pair<int, int>& cell);

    void merge(const std::pair<int, int>& cell1, const std::pair<int, int>& cell2);
};

int randInt(int min, int max);

void kruskal(vector<vector<int>>& maze, int entryRow, int entryCol, int exitRow, int exitCol) ;

#endif // KRUSKAL_HPP
