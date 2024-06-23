
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <mpi.h>
#include <numeric>
#include "dfs.hpp"

using namespace std;
 int rows = 64;
 int cols = 64;

// class MazeSolver {
// public:
    vector<pair<int, int>> MazeSolver:: solveMaze(vector<vector<char>>& maze) {
        int startX = -1, startY = -1;
        int rows = maze.size();
        int cols = maze[0].size();

        // Find the star`t point
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (maze[i][j] == 'S') {
                    startX = i;
                    startY = j;
                    break;
                }
            }
        }

        if (startX == -1 || startY == -1) {
            // Start point not found
            return {};
        }

        vector<pair<int, int>> path;
        if (dfs(maze, startX, startY, path)) {
            return path;
        } else {
            // No solution
            return {};
        }
    }

// private:
    bool MazeSolver:: dfs(vector<vector<char>>& maze, int x, int y, vector<pair<int, int>>& path) {
        int rows = maze.size();
        int cols = maze[0].size();

        // Check if we reached the end point
        if (maze[x][y] == 'E') {
            // cout<<x<<y<<endl;
            path.push_back({x, y});
            return true;
        }

        // Mark the current cell as visited
        maze[x][y] = 'X';

        int dx[] = {0, 0, 1, -1};
        int dy[] = {1, -1, 0, 0};

        // Try all four possible directions
        for (int i = 0; i < 4; i++) {
            int newX = x + dx[i];
            int newY = y + dy[i];

            if (isValid(maze, newX, newY)) {
                path.push_back({x, y});
                if (dfs(maze, newX, newY, path)) {
                    return true;
                }
                path.pop_back();
            }
        }

        return false;
    }

    bool MazeSolver :: isValid(vector<vector<char>>& maze, int x, int y) {
        int rows = maze.size();
        int cols = maze[0].size();
        return x >= 0 && x < rows && y >= 0 && y < cols && maze[x][y] != '*' && maze[x][y] != 'X';
    }
// };


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

        MazeSolver solver;

        vector<vector<char>> maze;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    int localRows = rows / size;
    int startRow = rank * localRows;
    int endRow = startRow + localRows - 1;
int startCol;
int endCol;

    string mazefile="a";
    string outputfile= "b";
    ifstream infile;
    // if(rank==0){
        // cout<<"argc = "<<argc<<endl;


        if (argc != 3) {
            return 1;
        }
        mazefile= argv[1];
        outputfile= argv[2];
        // cout << "Maze file: " << mazefile << endl;
        // cout << "Output file: " << outputfile << endl;
        // ifstream infile(mazefile);
        infile.open(mazefile);
        if (!infile.is_open()) {
            cerr << "Error: Unable to open input file." << endl;
            return 1;
        }

        string line;

        // Read lines from the file
        while (getline(infile, line)) {
            vector<char> row; // Vector to store each row of the matrix

            // Iterate over the characters in the line and add them to the row vector
            for (char c : line) {
                row.push_back(c);
            }

            // Add the row vector to the matrix
            maze.push_back(row);
        }

        // Close the input file
        infile.close();
        
    if (rank == 0) {
        vector<int> points;
        for (int i = 0; i < 3; ++i) {
            int rowunderconsideration=(i+1)*localRows-1;
            // cout<<rowunderconsideration<<endl;
            for(int j=0;j<cols;j++){
                // cout<<j<<endl;
                if (maze[rowunderconsideration][j]==' ' && maze[rowunderconsideration+1][j]==' '){
                    // cout<<"j"<<j<<"row"<<rowunderconsideration<<endl;
                    // pair<int,int> point=make_pair(rowunderconsideration,j);
                    points.push_back(j);
                    break;
                }
            }
        }
        if(points.size() < 3) {
            cout << "Path Not Possible!!" << endl;
            return 0; // or exit(0);
        }
            startCol=cols-1;
            endCol=points[0];

            // Send all three random integers to other processes
            for (int i = 1; i < size; ++i) {
                MPI_Send(points.data(), 3, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
    } else {
        std::vector<int> points(3);
        MPI_Recv(points.data(), 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(rank!=3){
            startCol=points[rank-1];
            endCol=points[rank];
        }
        else{
            startCol=points[rank-1];
            endCol=0;
        }
    }

    vector<vector<char>> localMaze(localRows, vector<char>(cols, 0));
    for (int i = startRow; i <= endRow; ++i) {
        for (int j = 0; j < cols; ++j) {
            localMaze[i - startRow][j] = maze[i][j];
        }
    }
    localMaze[0][startCol]='S';
    localMaze[localRows-1][endCol]='E';

    vector<pair<int, int>> solution = solver.solveMaze(localMaze);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (!solution.empty()) {
        // cout << "Path exists!" << endl;
        // cout << "Path coordinates: ";
        for (auto point : solution) {
            localMaze[point.first][point.second]='P';
            // cout << "(" << point.first << "," << point.second << ") ";
        }
        // cout << endl;
    } else {
        cout << "No path found!" << endl;
    }

    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if(localMaze[i][j]=='X'){
                localMaze[i][j]=' ';
            }
        }
        // cout << endl;
    }


vector<char> localMazeData(localRows * cols);
vector<char> globalMazeData(rows * cols);

for (int i = 0; i < localRows; ++i) {
    for (int j = 0; j < cols; ++j) {
        localMazeData[i * cols + j] = localMaze[i][j];
    }
}

MPI_Gather(localMazeData.data(), localRows * cols, MPI_CHAR, 
           globalMazeData.data(), localRows * cols, MPI_CHAR, 0, MPI_COMM_WORLD);



if (rank == 0) {
// cout<<"printing the final global one"<<endl;
        vector<vector<char>> globalMaze(rows, vector<char>(cols, 0));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                globalMaze[i][j] = globalMazeData[i * cols + j];
            }
        }
globalMaze[0][63]='S';
globalMaze[63][0]='E';


    ofstream outfile(outputfile);
    if (!outfile.is_open()) {
        cerr << "Error: Unable to open output file." << endl;
        return 1;
    }

    // Output the maze contents to the file
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            outfile << globalMaze[r][c];
        }
        outfile << endl;
    }

    // Close the output file
    outfile.close();
    // cout<<"printing final result"<<endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            cout << globalMaze[r][c];
        }
        cout << endl;
    }
    cout << endl;


}
    MPI_Finalize();
    return 0;

}























