
#include <bits/stdc++.h>
using namespace std;
// g++ -std=c++11 adj.cpp -o adj

void solve(vector<vector<int>>& maze){
    int n= maze.size();
    cout<<n<<endl;
    int c=0;
    for(int i=0;i<n;i++){
        for(int j=0; j<n;j++){
            if(maze[i][j]==0){
                maze[i][j]= -1;
            }
            if(maze[i][j]==1){
                maze[i][j]= c;
                c++;
            }
        }
    }
    // 0=wall

    // c=100;
    cout<<"c= "<<c<<endl;
    vector<vector<int>> adj(c, vector<int> (c, 0));
    cout<<"-----------------"<<endl;
    for(int i=0;i<n;i++){
        for(int j=0; j<n;j++){
            if(maze[i][j]!=-1){
                if(i+1<n && maze[i+1][j]!=-1){
                    cout<<maze[i][j]<<" , "<<maze[i+1][j]<<endl;
                    cout<<"i= "<<maze[i][j]<<" j= "<<maze[i+1][j]<<endl;
                    adj[maze[i][j]][maze[i+1][j]]=1;
                }
                if(j+1<n && maze[i][j+1]!=-1){
                    cout<<maze[i][j]<<" ,, "<<maze[i][j+1]<<endl;
                    cout<<"i= "<<maze[i][j]<<" j= "<<maze[i][j+1]<<endl;
                    adj[maze[i][j]][maze[i][j+1]]=1;
                }
                if(i-1>=0 && maze[i-1][j]!=-1 ){
                    cout<<maze[i][j]<<" ,,, "<<maze[i-1][j]<<endl;
                    cout<<"i= "<<maze[i][j]<<" j= "<<maze[i-1][j]<<endl;
                    adj[maze[i][j]][maze[i-1][j]]=1;
                }
                if(j-1>=0 && maze[i][j-1]!=-1){
                    cout<<maze[i][j]<<" ,,,, "<<maze[i][j-1]<<endl;
                    cout<<"i= "<<maze[i][j]<<" j= "<<maze[i][j-1]<<endl;
                    adj[maze[i][j]][maze[i][j-1]]=1;
                }
                // c--;
                // cout<<c<<endl;
            }
        }
    }
    cout<<"-----------------"<<endl;    
    for(int i=0;i<c;i++){
        for(int j=0;j<c;j++){
            cout<<adj[i][j]<<" ";
        }
        cout<<endl;
    }

}

signed main(){
    // ios_base::sync_with_stdio(false); cin.tie(0); mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    vector<vector<int>> maze;
    // make a 10x10 maze, 1 is path, 0 is wall
    maze.push_back({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    maze.push_back({1, 0, 0, 0, 0, 0, 0, 0, 0, 1});
    maze.push_back({1, 0, 1, 1, 1, 1, 1, 1, 0, 1});
    maze.push_back({1, 0, 1, 0, 0, 0, 0, 1, 0, 1});
    maze.push_back({1, 0, 1, 0, 1, 1, 0, 1, 0, 1});
    maze.push_back({1, 0, 1, 0, 1, 1, 0, 1, 0, 1});
    maze.push_back({1, 0, 0, 0, 0, 0, 0, 0, 0, 1});
    maze.push_back({1, 1, 0, 0, 0, 0, 0, 0, 0, 1});
    maze.push_back({1, 1, 1, 1, 1, 0, 0, 0, 0, 1});
    maze.push_back({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    solve(maze);
    return 0;
}
/*
1. Check borderline constraints. Can a variable you are dividing by be 0?
2. Use ll while using bitshifts
3. Do not erase from set while iterating it
4. Initialise everything
5. Read the task carefully, is something unique, sorted, adjacent, guaranteed??
6. DO NOT use if(!mp[x]) if you want to iterate the map later
7. Are you using i in all loops? Are the i's conflicting?
8. Use iterator to iterate thorugh maps if you want to changes the values
9. Use vector in place of pair to speed up typing
10. Try to make function outside all the loops in open space in order to reduce numerous compiling
11. Try not  use to INT_MAX because of out of bounds issues
*/ 