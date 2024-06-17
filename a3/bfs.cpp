#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <mpi.h>

using namespace std;

// Define dimensions of the maze
 int rows = 64;
 int cols = 64;
int wanttoincludeprob=1;
int threshtime=15;
// Function to generate random integer between min and max
int randInt(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Function to generate the perfect maze using BFS algorithm
// Function to generate the perfect maze using BFS algorithm
void bfs(vector<vector<int>>& maze, int entryRow, int entryCol, int exitRow, int exitCol) {
    int localrows=rows/4;
    int timing=0;
    int ranklocal;
    MPI_Comm_rank(MPI_COMM_WORLD, &ranklocal);
    // cout << "Debugging on process " << ranklocal << endl;
    queue<pair<int, int>> q;
    q.push({entryRow, entryCol});

    while (!q.empty()) {
        timing=timing+1;
        // cout << "Contents of the queue at process " << ranklocal << ":" << endl;
        int r = q.front().first;
        int c = q.front().second;
        q.pop();

        maze[r][c] = 1; // Mark the cell as visited
        // Directions: Up, Down, Left, Right
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        vector<int> dir = {0, 1, 2, 3};
        random_shuffle(dir.begin(), dir.end() );

        for (int d : dir) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            int wr = r + 2*dr[d];
            int wc = c + 2*dc[d];

            if (nr >= 0 && nr < maze.size()-1 && nc >= 0 && nc < maze[0].size() && maze[nr][nc] == 0) {
                // Check if all adjacent cells are 0
                bool allZero = true;
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (dr == 0 && dc == 0) continue; // Skip the current cell
                        if (dr != 0 && dc != 0) continue; // Skip diagonal cells
                        int newr = nr + dr;
                        int newc = nc + dc;
                        // cout << "formulating New point (" << newr << ", " << newc << ")" << endl;

                        if (newr == r && newc == c ) {
                            // Out of bounds, ignore this cell
                            continue;
                        }

                        if (newr < 0 || newr > localrows-1 || newc < 0 || newc > cols ) {
                            // cout<<"ignored";
                            // Out of bounds, ignore this cell
                            continue;
                        }
                        // cout << "****New point (" << newr << ", " << newc << ")" << endl;
                        if (maze[newr][newc] != 0) {
                            allZero = false;
                            break; // Exit inner loop early if a non-zero cell is found
                        }
                    }
                    if (!allZero) {
                        break; // Exit outer loop early if a non-zero cell is found
                    }
                }
                    // cout<<endl<<bool(allZero)<<endl;

                if(wanttoincludeprob==1 && timing>threshtime){
                    timing=0;
                    double prob = (double)rand() / RAND_MAX;
                    double probabilityThreshold = 0.8; 
                    if (prob < probabilityThreshold) {
                        // std::cout << "Continuing loop" << std::endl;
                    } else {
                        // std::cout << "Exiting loop" << std::endl;
                        continue;
                    }
                }


                if (!allZero) {
                    continue; // Exit outer loop early if a non-zero cell is found
                }


                maze[nr][nc] = 1;
                // maze[wr][wc] = 1;
                q.push({nr, nc});
                // cout << "New point (" << nr << ", " << nc << ")" << endl;
                // cout << "Wall point (" << wr << ", " << wc << ")" << endl;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);
    int entryCol;
    int exitCol;
    if (rank == 0) {
        // Generate 3 different random integers between 1 and 63
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, cols-1);

        std::vector<int> random_integers(3);
        for (int i = 0; i < 3; ++i) {
            int random_int = dis(gen);
            random_integers[i] = random_int;
        }
        entryCol=cols-1;
        exitCol=random_integers[0];
        // Send all three random integers to other processes
        for (int i = 1; i < size; ++i) {
            MPI_Send(random_integers.data(), 3, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        // std::cout << "Process " << rank << ": EntryCol = " << entryCol << ", ExitCol = " << exitCol << std::endl;
    } else {
        // Receive the random integers from rank 0
        std::vector<int> received_integers(3);
        MPI_Recv(received_integers.data(), 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(rank!=3){
            entryCol=received_integers[rank-1];
            exitCol=received_integers[rank];
        }
        else{
            entryCol=0;
            exitCol=0;
        }
    }
    cout.flush();


    int localRows = rows / size;
    int startRow = rank * localRows;
    int endRow = startRow + localRows - 1;
    int entryRow = 0;

    if(rank==3){
        entryRow = localRows-1;
    }
    int exitRow = localRows-1;
    
    vector<vector<int>> localMaze(localRows, vector<int>(cols, 0));
    cout.flush();
    // std::cout << "Process " << rank << ": Entrypoint = " << entryRow<<","<<entryCol << ", Exitpoint = " << exitRow<<","<<exitCol << std::endl;

    bfs(localMaze, entryRow, entryCol,exitRow,exitCol);
    for (int c = 0; c < cols; ++c) {
        if (localMaze[exitRow-1][c]==1) {localMaze[exitRow][c]=1; break;}
    }

    vector<int> localMazeData(localRows * cols);
    vector<int> globalMazeData(rows * cols);

// Flatten the localMaze and globalMaze for MPI_Gather
for (int i = 0; i < localRows; ++i) {
    for (int j = 0; j < cols; ++j) {
        localMazeData[i * cols + j] = localMaze[i][j];
    }
}

MPI_Gather(localMazeData.data(), localRows * cols, MPI_INT, 
           globalMazeData.data(), localRows * cols, MPI_INT, 0, MPI_COMM_WORLD);

if (rank == 0) {
    vector<vector<int>> globalMaze(rows, vector<int>(cols, 0));
    
    // Unflatten the globalMazeData into globalMaze
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            globalMaze[i][j] = globalMazeData[i * cols + j];
        }
    }

    for (int r = 0; r < rows; ++r) {
        cout<<"{";
        for (int c = 0; c < cols; ++c) {
        // cout<<"{";

            if (r == 0 && c == 63) cout << "'S'"; // Entry
            else if (r == 63 && c == 0) cout << "'E',"; // Exit
            else cout << (globalMaze[r][c] == 1 ? "' '," : "'#',");
        }
        cout << "}"<<endl;
    }
    cout << "}"<<endl;

}


    // // Gather local mazes into global maze
//     vector<vector<int>> globalMaze(rows, vector<int>(cols, 0));
//     MPI_Gather(&localMaze[0][0], localRows * cols, MPI_INT, &globalMaze[0][0], localRows * cols, MPI_INT, 0, MPI_COMM_WORLD);
// cout<<"gathered";
//     if (rank == 0) {
//         for (int r = 0; r < rows; ++r) {
//             for (int c = 0; c < cols; ++c) {
//                 if (r == entryRow && c == entryCol) cout << "E"; // Entry
//                 // else if (r == exitRow && c == exitCol) cout << "X"; // Exit
//                 else cout << (globalMaze[r][c] == 1 ? "  " : "*");
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }

    MPI_Finalize();
    return 0;
}








// #include <iostream>
// #include <vector>
// #include <queue>
// #include <cstdlib>
// #include <ctime>
// #include <numeric>
// #include <algorithm>
// #include <mpi.h>

// using namespace std;

// // Define dimensions of the maze
// const int rows = 64;
// const int cols = 64;

// // Define MPI tags for communication
// const int TAG_SEND = 0;
// const int TAG_RECEIVE = 1;

// // Function to generate random integer between min and max
// int randInt(int min, int max) {
//     return min + rand() % (max - min + 1);
// }

// // Function to generate the perfect maze using BFS algorithm
// void generateMazeBFS(vector<vector<int>>& maze, int startRow, int startCol) {
//     queue<pair<int, int>> q;
//     q.push({startRow, startCol});

//     while (!q.empty()) {
//         int r = q.front().first;
//         int c = q.front().second;
//         q.pop();

//         maze[r][c] = 1; // Mark the cell as visited

//         // Directions: Up, Down, Left, Right
//         int dr[] = {-1, 1, 0, 0};
//         int dc[] = {0, 0, -1, 1};
//         vector<int> dir = {0, 1, 2, 3};
//         random_shuffle(dir.begin(), dir.end());

//         for (int d : dir) {
//             int nr = r + dr[d];
//             int nc = c + dc[d];
//             int wr = r + 2 * dr[d];
//             int wc = c + 2 * dc[d];

//             if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && maze[nr][nc] == 0) {
//                 maze[nr][nc] = 1;
//                 maze[wr][wc] = 1;
//                 q.push({wr, wc});
//             }
//         }
//     }
// }

// // Function to find the parent of a cell
// int find(vector<int>& parent, int x) {
//     if (x != parent[x]) {
//         parent[x] = find(parent, parent[x]);
//     }
//     return parent[x];
// }

// // Function to generate the perfect maze using Kruskal's algorithm
// void generateMazeKruskal(vector<vector<int>>& maze) {
//     // Initialize walls
//     vector<pair<int, int>> walls;

//     // Add all the vertical walls
//     for (int r = 1; r < rows; r += 2) {
//         for (int c = 0; c < cols; c += 2) {
//             walls.push_back({r, c});
//         }
//     }

//     // Add all the horizontal walls
//     for (int r = 0; r < rows; r += 2) {
//         for (int c = 1; c < cols; c += 2) {
//             walls.push_back({r, c});
//         }
//     }

//     // Randomly shuffle the walls
//     random_shuffle(walls.begin(), walls.end());

//     // Initialize disjoint set data structure
//     vector<int> parent(rows * cols);
//     iota(parent.begin(), parent.end(), 0); // Use 'iota' from the standard library

//     // Remove walls to create the maze
//     for (auto& wall : walls) {
//         int r = wall.first;
//         int c = wall.second;

//         int cell1 = r * cols + c;
//         int cell2 = -1;

//         if (r % 2 == 0) {
//             cell2 = (r - 1) * cols + c;
//         } else {
//             cell2 = r * cols + c - 1;
//         }

//         if (find(parent, cell1) != find(parent, cell2)) {
//             maze[r][c] = 1;
//             parent[find(parent, cell2)] = find(parent, cell1);
//         }
//     }
// }


// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     srand(time(NULL) + rank);

//     // Divide rows among processes
//     int localRows = rows / size;
//     int startRow = rank * localRows;
//     int endRow = startRow + localRows - 1;

//     // Allocate memory for local portion of maze
//     vector<vector<int>> localMaze(localRows, vector<int>(cols, 0));

//     // Generate maze using BFS
//     cout<<"jiji"<<endl;
//     generateMazeBFS(localMaze, startRow, 0);

//     // Gather local mazes into global maze
//     vector<vector<int>> globalMaze(rows, vector<int>(cols, 0));
//     MPI_Gather(&localMaze[0][0], localRows * cols, MPI_INT, &globalMaze[0][0], localRows * cols, MPI_INT, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         // Display the global maze
//         cout << "BFS Maze:" << endl;
//         for (int r = 0; r < rows; ++r) {
//             for (int c = 0; c < cols; ++c) {
//                 cout << (globalMaze[r][c] == 1 ? "  " : "\u2588\u2588");
//             }
//             cout << endl;
//         }
//         cout << endl;

//         // Generate maze using Kruskal's algorithm
//         generateMazeKruskal(globalMaze);

//         // Display the global maze
//         cout << "Kruskal Maze:" << endl;
//         for (int r = 0; r < rows; ++r) {
//             for (int c = 0; c < cols; ++c) {
//                 cout << (globalMaze[r][c] == 1 ? "  " : "\u2588\u2588");
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }

//     MPI_Finalize();
//     return 0;
// }
