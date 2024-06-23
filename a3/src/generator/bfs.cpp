#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <mpi.h>

#include "bfs.hpp"
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



        // cout << "Current point (" << r << ", " << c << ")" << endl;
        // cout<<"\n";
        // break;
        maze[r][c] = 1; // Mark the cell as visited

        // Directions: Up, Down, Left, Right
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        vector<int> dir = {0, 1, 2, 3};
        random_shuffle(dir.begin(), dir.end());

        for (int d : dir) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            int wr = r + 2*dr[d];
            int wc = c + 2*dc[d];

            if (nr >= 0 && nr < maze.size()-1 && nc >= 0 && nc < maze[0].size() && maze[nr][nc] == 0) {

// cout << "**New point (" << nr << ", " << nc << ")" << endl;
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

            if (argc != 2) {
                return 1;
            }

            string outputfile= argv[1];
            ofstream outfile(outputfile);
            if (!outfile.is_open()) {
                cerr << "Error: Unable to open output file." << endl;
                return 1;
            }

            // check if there is path from 0, 63 to 63, 0
            // if not - 
            // Output the maze contents to the file
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    if(globalMaze[r][c]==1){
                        if(r==0 && c==63){
                            outfile<<'S';
                        }
                        else if(r==63 && c==0){
                            outfile<<'E';
                        }
                        else{
                            outfile<<' ';
                        }
                    }
                    else{
                        outfile<<'*';
                    }
                }
                outfile << endl;
            }

            // Close the output file
            outfile.close();
        //     for (int r = 0; r < rows; ++r) {
        //     // cout<<"{";
        //     for (int c = 0; c < cols; ++c) {
        //     // cout<<"{";

        //         // if (r == 0 && c == 63) cout << "'S'"; // Entry
        //         // else if (r == 63 && c == 0) cout << "'E',"; // Exit
        //         // else cout << (globalMaze[r][c] == 1 ? "' '," : "'*',");
        //     }
        //     // cout << "}"<<endl;
        // }
        // cout << "}"<<endl;

    }
    MPI_Finalize();
    return 0;
}


