#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <mpi.h>
#include <numeric>

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
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

class DisjointSet {
public:
    vector<vector<pair<int, int>>> parent;
    vector<vector<int>> rank;

    DisjointSet(int rows, int cols) {
        parent.resize(rows, std::vector<std::pair<int, int>>(cols));
        rank.resize(rows, std::vector<int>(cols, 0));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                parent[i][j] = {i, j};
            }
        }
    }

    // Find operation with path compression
    pair<int, int> find(const pair<int, int>& cell) {
        int x = cell.first;
        int y = cell.second;
        if (parent[x][y] != make_pair(x, y)) {
            parent[x][y] = find(parent[x][y]);
        }
        return parent[x][y];
    }

    // Union operation with union by rank
    void merge(const pair<int, int>& cell1, const pair<int, int>& cell2) {
        auto root1 = find(cell1);
        auto root2 = find(cell2);
        if (root1 != root2) {
            if (rank[root1.first][root1.second] < rank[root2.first][root2.second]) {
                parent[root1.first][root1.second] = root2;
            } else if (rank[root1.first][root1.second] > rank[root2.first][root2.second]) {
                parent[root2.first][root2.second] = root1;
            } else {
                parent[root2.first][root2.second] = root1;
                rank[root1.first][root1.second]++;
            }
        }
    }
};



void kruskal(vector<vector<int>>& maze, int entryRow, int entryCol, int exitRow, int exitCol) {
    int localrows = maze.size();
    int cols = maze[0].size();

    vector<pair<pair<int, int>,pair<int, int>>> walls;
    
    for (int r = 0; r < localrows; r += 2) {
        for (int c = 0; c < cols; c += 2) {
            if((r==entryRow && c==entryCol)||(r+2==entryRow && c==entryCol) || (r+2>=localrows)) continue;
            walls.push_back({{r, c},{r+2,c}});
        }
    }

    for (int r = 0; r < localrows; r += 2) {
        for (int c = 0; c < cols; c += 2) {
            if((r==entryRow && c==entryCol)||(r==entryRow && c+2==entryCol)|| (c+2>=cols)) continue;
             walls.push_back({{r, c},{r,c+2}});
        }
    }
        random_shuffle(walls.begin(), walls.end());

    DisjointSet ds(localrows , cols);

    for (auto& wall : walls) {
        
        pair<int,int> cell1=wall.first;
        pair<int,int> cell2=wall.second;

        int r1 = wall.first.first;
        int c1 = wall.first.second;
        int r2 = wall.second.first;
        int c2 = wall.second.second;
        // cout << "cell1: (" << r1 << "," << c1 << ") cell2: (" << r2 << "," << c2<<")" << endl;

if(r1==r2){
// cout<<"horizontal"<<endl;

int br=int((r1+r2)/2);
int bc=int((c1+c2)/2);
int ar=br-1;
int ac=bc;
int cr=br+1;
int cc=bc;
pair<int,int> top={ar,ac};
pair<int,int> middle={br,bc};
pair<int,int> bottom={cr,cc};
if(ar<0){
    continue;
}
if(cr>=localrows){
    continue;
}
// cout<<"horizontal pass"<<endl;

if(ds.find(top)!=ds.find(bottom)){
// cout<<"horizontal not match"<<endl;

    maze[ar][ac]=1;
    maze[br][bc]=1;
    maze[cr][cc]=1;
    ds.merge(top, middle);
    ds.merge(middle, bottom);
}

}
else{
// cout<<"vertical"<<endl;
//vertival wall
int br=int((r1+r2)/2);
int bc=int((c1+c2)/2);
int ar=br;
int ac=bc-1;
int cr=br;
int cc=bc+1;
pair<int,int> left={ar,ac};
pair<int,int> middle={br,bc};
pair<int,int> right={cr,cc};
if(ac<0){
    continue;
}
if(cc>=cols){
    continue;
}
// cout<<"vertical pass"<<endl;

if(ds.find(left)!=ds.find(right)){
// cout<<"vertical not match"<<endl;

    maze[ar][ac]=1;
    maze[br][bc]=1;
    maze[cr][cc]=1;
    ds.merge(left, middle);
    ds.merge(middle, right);
}



}

        //                             cout << "Maze generated using BFS:" << endl;
        // for (int r = 0; r < maze.size(); ++r) {
        //     for (int c = 0; c < maze[0].size(); ++c) {
        //         if (r == entryRow && c == entryCol) cout << "E"; // Entry
        //         // else if (r == exitRow && c == exitCol) cout << "X"; // Exit
        //         else cout << (maze[r][c] == 1 ? "-" : "*");
        //     }
        //     cout << endl;
        // }
        // cout << endl;


        }

// cout<<endl<<"returning"<<endl;
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
    } else {
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
// if (rank==0)
    kruskal(localMaze, entryRow, entryCol,exitRow,exitCol);
        // for (int c = 0; c < cols; ++c) {
        //     if (localMaze[exitRow-1][c]==1) {localMaze[exitRow][c]=1; break;}
        //     }

//printing 
MPI_Barrier(MPI_COMM_WORLD);
// cout<<"////////////////////////////////////////////////////donee!!";
vector<int> localMazeData(localRows * cols);
vector<int> globalMazeData(rows * cols);

for (int i = 0; i < localRows; ++i) {
    for (int j = 0; j < cols; ++j) {
        localMazeData[i * cols + j] = localMaze[i][j];
    }
}

MPI_Gather(localMazeData.data(), localRows * cols, MPI_INT, 
           globalMazeData.data(), localRows * cols, MPI_INT, 0, MPI_COMM_WORLD);



if (rank == 0) {

        vector<vector<int>> globalMaze(rows, vector<int>(cols, 0));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                globalMaze[i][j] = globalMazeData[i * cols + j];
            }
        }
for(int d=0;d<size;d++){
            for (int c = 0; c < cols; ++c) {
                if(d*localRows-1<0) continue;
            if (globalMaze[d*localRows-1][c]==1) {globalMaze[d*localRows][c]=1; break;}
            }}

            for (int r = 2; r < cols; ++r) {
globalMaze[rows-1][r]=0;
}
for (int r = 2; r < rows; ++r) {
globalMaze[r][cols-1]=0;
}
if (globalMaze[0][62]==0 && globalMaze[1][63]==0) globalMaze[0][62]=1;
        cout<<"{";

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


    // for (int r = 0; r < rows; ++r) {
    //     cout<<"[";
    //     for (int c = 0; c < cols; ++c) {
    //     // cout<<"{";

    //         if (r == 0 && c == 63) cout << "1,"; // Entry
    //         else if (r == 63 && c == 0) cout << "1,"; // Exit
    //         else cout << (globalMaze[r][c] == 1 ? "1," : "0,");
    //     }
    //     cout << "]"<<endl;
    // }
    // cout << "]"<<endl;

}
    MPI_Finalize();
    return 0;
}