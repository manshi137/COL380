Subtask1:

    gcc assignment2_subtask1.cpp -o assignment2_subtask1 -std=c++11
    ./assignment2_subtask1 <taskid> ....
    if taskid==1
        ./assignment2_subtask1 1 <N> <M> <P> <matrix in rowmajor of size N> <kernel in rowmajor of size M>
    if taskid==2
        ./assignment2_subtask1 2 <choice> <N> <M> <matrix in rowmajor of size N*M>
    if taskid==3
        ./assignment2_subtask1 3 <choice> <poolsize> <N> <matrix in rowmajor of size N*N>
    if taskid==4
        ./assignment2_subtask1 4 <choice> <N> <vector of size N>

Subtask2:

    nvcc assignment2_subtask2.cu
    ./a.out <taskid> ....
    if taskid==1
        ./a.out 1 <N> <M> <P> <matrix in rowmajor of size N> <kernel in rowmajor of size M>
    if taskid==2
        ./a.out 2 <choice> <N> <M> <matrix in rowmajor of size N*M>
    if taskid==3
        ./a.out 3 <choice> <poolsize> <N> <matrix in rowmajor of size N*N>
    if taskid==4
        ./a.out 4 <choice> <N> <vector of size N>

Subtask3:

    nvcc assignment2_subtask3.cu
    ./a.out

Subtask4:

    nvcc assignment2_subtask4.cu
    ./a.out <choice=0 for no streams and choice=1 for with streams>
