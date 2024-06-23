/*
reference - https://github.com/Lehmannhen/MPI-Dijkstra
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <mpi.h>
#include <map>
using namespace std;
#define INFINITY 1000000

map<int, pair<int,int>> mp;
int Read_n(int n,int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
                   int my_rank, int loc_n);
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
              MPI_Comm comm);
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n);
vector<int> Print_paths(int global_pred[], int n);


void solve(vector<vector<int>>& maze){
    int n= maze.size();
    // cout<<n<<endl;
    int c=0;

    for(int i=0;i<n;i++){
        for(int j=n-1; j>=0;j--){
            // if(maze[i][j]==0){
            //     maze[i][j]= -1;
            // }

            if(maze[i][j]==1){

                maze[i][j]= c;
                mp[c] = {i, j};
                // cout<<c<<" - "<<i<<" "<<j<<"   |    ";
                c++;
            }
        }

    }
    // 0=wall

    // c=100;
    // cout<<"c= "<<c<<endl;
    vector<vector<int>> adj(c, vector<int> (c, INFINITY));
    // cout<<"-----------------"<<endl;
    for(int i=0;i<n;i++){
        // adj[i][i]=0;
        for(int j=0; j<n;j++){
            if(maze[i][j]!=-1){ //movable cell =
                if(i<2){

                    // cout<<"i="<<i<<"j="<<j<<"c="<<maze[i][j]<<endl;
                }
                adj[maze[i][j]][maze[i][j]] = 0;
                if(i+1<n && maze[i+1][j]!=-1){
                    // cout<<maze[i][j]<<" , "<<maze[i+1][j]<<endl;
                    // cout<<"i= "<<maze[i][j]<<" j= "<<maze[i+1][j]<<endl;
                    adj[maze[i][j]][maze[i+1][j]]=1;
                }
                if(j+1<n && maze[i][j+1]!=-1){
                    // cout<<maze[i][j]<<" ,, "<<maze[i][j+1]<<endl;
                    // cout<<"i= "<<maze[i][j]<<" j= "<<maze[i][j+1]<<endl;
                    adj[maze[i][j]][maze[i][j+1]]=1;
                }
                if(i-1>=0 && maze[i-1][j]!=-1 ){
                    // cout<<maze[i][j]<<" ,,, "<<maze[i-1][j]<<endl;
                    // cout<<"i= "<<maze[i][j]<<" j= "<<maze[i-1][j]<<endl;
                    adj[maze[i][j]][maze[i-1][j]]=1;
                }
                if(j-1>=0 && maze[i][j-1]!=-1){
                    // cout<<maze[i][j]<<" ,,,, "<<maze[i][j-1]<<endl;
                    // cout<<"i= "<<maze[i][j]<<" j= "<<maze[i][j-1]<<endl;
                    adj[maze[i][j]][maze[i][j-1]]=1;
                }
                // c--;
                // cout<<c<<endl;
            }
        }
        // adj[i][i]=0;
    }
    // cout<<"-----------------"<<endl;    
    // for(int i=0;i<c;i++){
    //     for(int j=0;j<c;j++){
    //         cout<<adj[i][j]<<"x";
    //     }
    //     cout<<endl;
    // }
    maze =adj;
}


int main(int argc, char **argv) {
    int *loc_mat, *loc_dist, *loc_pred, *global_dist = NULL, *global_pred = NULL;
    int my_rank, p, loc_n, n;
    
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &p);
    string mazefile="a";
    string outputfile= "b";
    ifstream infile;
    if(my_rank==0){
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
        // infile >> n;
    }
    n=64;
    
    // char *mat2 = NULL;
    int *mat4 = NULL;
    vector<vector<char>> mat2(64,vector<char>(64));
    if(my_rank==0){
        
        // mat2 = (char *)malloc((n) * (n) * sizeof(char));
        
        string line;
        // Read lines from the file
        // cout<<"--------------------mat2-----------------------"<<endl;
        
        // printf("(mat2):\n");
        vector<vector<int>> mat3(64, vector<int>(64, 5));

        int x=0, y=0;
        while (getline(infile, line)) {

            for (char c : line) {
                // row.push_back(c);
                if(c=='S' || c=='E' || c==' '){
                    mat3[x][y] = 1;
                }
                else{
                    mat3[x][y]= -1;
                }
                // cout<<c;
                // cout<<mat3[x][y];
                y++;
            }
            
            // cout<<endl;
            x++;
            y=0;
        }

        // for(int i=0;i<mat3.size();i++){
        //     for(int j=0;j<mat3.size();j++){
        //         cout<<mat3[i][j]<<",";
        //     }
        //     cout<<endl;
        // }
        
        infile.close();
        
        solve(mat3);
        n = mat3.size();
        // for(int i=0;i<2;i++){
        //     for(int j=0;j<n;j++){
        //         cout<<mat3[i][j]<<"'";
        //     }
        //     cout<<endl;
        //     cout<<endl;
        // }
        int pad=0;
        if(n%p!=0){
            pad = p - n%p;
        }
        n=n+pad;
        // int *mat4;
        mat4= (int *)malloc((n) * (n) * sizeof(int));
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(i<n-pad && j<n-pad){
                    mat4[i*n + j ] = mat3[i][j];
                }
                else{
                    mat4[i*n +j] = INFINITY;
                }
            }
        }
        // printf("here");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    n = Read_n(n,my_rank, comm);

    loc_n = n / p;
    loc_mat = (int *)malloc(n * loc_n * sizeof(int));
    loc_dist =(int *) malloc(loc_n * sizeof(int));
    loc_pred = (int *)malloc(loc_n * sizeof(int));
    blk_col_mpi_t = Build_blk_col_type(n, loc_n);

    if (my_rank == 0) {
        global_dist = (int *)malloc(n * sizeof(int));
        global_pred = (int *)malloc(n * sizeof(int));
    }

    // printf("-------scatter-----------\n");
    MPI_Scatter(mat4, 1, blk_col_mpi_t, loc_mat, (n) * loc_n, MPI_INT, 0, comm);
    // printf("--------------scatter done---------------\n");

    Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n, comm);

    MPI_Gather(loc_dist, loc_n, MPI_INT, global_dist, loc_n, MPI_INT, 0, comm);
    MPI_Gather(loc_pred, loc_n, MPI_INT, global_pred, loc_n, MPI_INT, 0, comm);


    vector<int> path;
    if (my_rank == 0) {
        // Print_dists(global_dist, n);
        path = Print_paths(global_pred, n);
        // for(int x:path)cout<<x<<" ->";


        int exit = mp.size()-1;
        // cout<<"exit-----------"<<exit<<endl;

        vector<vector<char>> final(64, vector<char>(64, '*'));

        for(int i=0; i<mp.size(); i++){
            int x_index= mp[i].first;
            int y_index= mp[i].second;
            final[x_index][y_index]= ' ';
        }
        for(int i=0; i<path.size(); i++){
            final[mp[path[i]].first][mp[path[i]].second]='P';
        }
        final[0][63]='S';
        final[63][0]='E';
        // cout<<"-----------------------------------------final ans--------------------"<<endl;
        for(int i=0; i<final.size(); i++){
            for(int j=0; j<final.size(); j++){
                cout<<final[i][j];
            }
            cout<<endl;
        }
        
        ofstream outfile(outputfile);
        if (!outfile.is_open()) {
            cerr << "Error: Unable to open output file." << endl;
            return 1;
        }

        // Output the maze contents to the file
        for (int r = 0; r < 64; ++r) {
            for (int c = 0; c < 64; ++c) {
                outfile << final[r][c];
            }
            outfile << endl;
        }

        // Close the output file
        outfile.close();

        free(global_dist);
        free(global_pred);
    }
    free(loc_mat);
    free(loc_pred);
    free(loc_dist);
    MPI_Type_free(&blk_col_mpi_t);
    MPI_Finalize();
    return 0;
}


int Read_n(int n, int my_rank, MPI_Comm comm) {
    // int n;

    // if (my_rank == 0)
    //     scanf("%d", &n);

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    return n;
}


MPI_Datatype Build_blk_col_type(int n, int loc_n) {
    MPI_Aint lb, extent;
    MPI_Datatype block_mpi_t;
    MPI_Datatype first_bc_mpi_t;
    MPI_Datatype blk_col_mpi_t;

    MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
    MPI_Type_get_extent(block_mpi_t, &lb, &extent);
    MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);

    MPI_Type_create_resized(first_bc_mpi_t, lb, extent, &blk_col_mpi_t);

    MPI_Type_commit(&blk_col_mpi_t);

    MPI_Type_free(&block_mpi_t);
    MPI_Type_free(&first_bc_mpi_t);

    return blk_col_mpi_t;
}


/*

6
0 4 2 1000000 1000000 1000000
1000000 0 5 10 1000000 1000000
1000000 1000000 0 1000000 3 1000000
1000000 1000000 1000000 0 1000000 11
1000000 1000000 1000000 4 0 1000000
1000000 1000000 99 1000000 1000000 0

6 0 4 2 1000000 1000000 1000000 1000000 0 5 10 1000000 1000000 1000000 1000000 0 1000000 3 1000000 1000000 1000000 1000000 0 1000000 11 1000000 1000000 1000000 4 0 1000000 1000000 1000000 99 1000000 1000000 0

 */


void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
                   int my_rank, int loc_n) {
    int loc_v;

    if (my_rank == 0)
        loc_known[0] = 1;
    else
        loc_known[0] = 0;

    for (loc_v = 1; loc_v < loc_n; loc_v++)
        loc_known[loc_v] = 0;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        loc_dist[loc_v] = loc_mat[0 * loc_n + loc_v];
        loc_pred[loc_v] = 0;
    }
}






void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
              MPI_Comm comm) {

    int i, loc_v, loc_u, glbl_u, new_dist, my_rank, dist_glbl_u;
    int *loc_known;
    int my_min[2];
    int glbl_min[2];

    MPI_Comm_rank(comm, &my_rank);
    int p=0;
    if(n%4!=0){
        p = 4 - n%4;
    }
    n=n+p;

    loc_known =(int *) malloc(loc_n * sizeof(int));

    Dijkstra_Init(loc_mat, loc_pred, loc_dist, loc_known, my_rank, loc_n);


    for (i = 0; i < n - 1; i++) {
        loc_u = Find_min_dist(loc_dist, loc_known, loc_n);

        if (loc_u != -1) {
            my_min[0] = loc_dist[loc_u];
            my_min[1] = loc_u + my_rank * loc_n;
        }
        else {
            my_min[0] = INFINITY;
            my_min[1] = -1;
        }



        MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);

        dist_glbl_u = glbl_min[0];
        glbl_u = glbl_min[1];


        if (glbl_u == -1)
            break;

        if ((glbl_u / loc_n) == my_rank) {
            loc_u = glbl_u % loc_n;
            loc_known[loc_u] = 1;
        }

        for (loc_v = 0; loc_v < loc_n; loc_v++) {
            if (!loc_known[loc_v]) {
                new_dist = dist_glbl_u + loc_mat[glbl_u * loc_n + loc_v];
                if (new_dist < loc_dist[loc_v]) {
                    loc_dist[loc_v] = new_dist;
                    loc_pred[loc_v] = glbl_u;
                }
            }
        }
    }
    free(loc_known);
}





int Find_min_dist(int loc_dist[], int loc_known[], int loc_n) {
    int loc_u = -1, loc_v;
    int shortest_dist = INFINITY;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (!loc_known[loc_v]) {
            if (loc_dist[loc_v] < shortest_dist) {
                shortest_dist = loc_dist[loc_v];
                loc_u = loc_v;
            }
        }
    }
    return loc_u;
}


vector<int> Print_paths(int global_pred[], int n) {
    int v, w, *path, count, i;

    path = (int *) malloc(n * sizeof(int));
    vector<int> ans;

    for (v = mp.size()-1; v < mp.size(); v++) {
        // printf("%3d:    ", v);
        count = 0;
        w = v;
        while (w != 0) {
            path[count] = w;
            count++;
            w = global_pred[w];
        }
        // printf("0 ");
        
        for (i = count-1; i >= 0; i--){
            // printf("%d ", path[i]);
            ans.push_back(path[i]);
            // cout<<path[i]<<"("<<mp[path[i]].first<<","<<mp[path[i]].second<<")";
        }
        // printf("\n");

    }

    // cout<<endl;
    // free(path);
    return ans;

}