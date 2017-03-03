#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"


int main(int argc, char* argv[]){

  // MPI Initialize
  int rank, total;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total);

  int x=200/(total-1);
  int data[200+x][200];
  int middle[200+x][198];
  int local[x][200];
  int result[x][198];
  int temporary[2][200];
  int local2[x][198];
  int result2[x][196];
  int temporary2[2][198];
  int middle2[200+x][196];


  int first[3][3]  = { {-1,-1,-1}, {2,2,2}, {-1,-1,-1} };
  int second[3][3] = { {-1,2,-1}, {-1,2,-1}, {-1,2,-1} };
  int third[3][3]  = { {-1,-1,2}, {-1,2,-1}, {2,-1,-1} };
  int fourth[3][3] = { {2,-1,-1}, {-1,2,-1}, {-1,-1,2} };

  int threshold;
  sscanf(argv[3], "%d", &threshold);    // threshold initialize

  // READING THE FILE AND KEEP IT IN THE MASTER PROCESSOR
  if(rank==0){
    FILE *cin = fopen(argv[1], "r");
    int i=0;
    for(; i<x; i++){
       int j=0;
       for(; j<200; j++){
          data[i][j] = 0;
       }
    } 
    i=x;
    for(; i<200+x; i++){
      int j=0;
      int temp=0;
      for(; j<200; j++){
        fscanf(cin,"%d", &temp);
        data[i][j] = temp;
      }
    }
    fclose(cin);
  }


  // distribute the data among processors
  MPI_Scatter(data,x*200,MPI_INT,local,x*200,MPI_INT,0,MPI_COMM_WORLD);





  // MEAN VALUES, SMOOTHING
  if (rank!=0) {
    int i=0;
    for(; i<x-2; i++){
      int j=0;
      for(; j<198; j++){
        int avg = 0;
        int k=i;
        for(; k<=i+2; k++){
          int l=j;
          for(; l<=j+2; l++){
            avg += local[k][l];
          }
        }
        avg/=9;
        result[i][j]= avg;
      }
    }

    if(rank!=1){
      if(x>1)
        MPI_Send(&(local[0][0]), 2*200 , MPI_INT, rank-1, 0, MPI_COMM_WORLD);
      else if(x==1){
        MPI_Send(&(local[0][0]), 200 , MPI_INT, rank-1, 0, MPI_COMM_WORLD);
        if(rank>2)
          MPI_Send(&(local[0][0]), 200 , MPI_INT, rank-2, 0, MPI_COMM_WORLD);
      }
    }
    if(rank!=total-1){
      if(x>1)
        MPI_Recv(&(temporary[0][0]), 2*200, MPI_INT, rank+1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      else if(x==1){
        MPI_Recv(&(temporary[0][0]), 200, MPI_INT, rank+1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank<total-2)
          MPI_Recv(&(temporary[1][0]), 200, MPI_INT, rank+2, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
    }
    if(rank!=total-1){
      for(; i<x; i++){
        int j=0;
        for(; j<198; j++){
          int avg=0;
          int m=j;
          for(; m<=j+2; m++){
            avg += local[i][m] + temporary[0][m];
            if(i==x-1)
              avg+= temporary[1][m];
            else if(i==x-2)
              avg+= local[i+1][m];
          }
          avg /= 9;
          result[i][j] = avg;
        }
      }
    }
  }

  // collecting smoothed data from processors to the master processor
  MPI_Gather(result,x*198,MPI_INT,middle,x*198,MPI_INT,0,MPI_COMM_WORLD);

  // distribute the smoothed data among processors
  MPI_Scatter(middle,x*198,MPI_INT,local2,x*198,MPI_INT,0,MPI_COMM_WORLD);



  // THRESHOLDING
  if(rank!=0){
    int i=0;
    for(; i<x-2; i++){
      int j=0;
      for(; j<196; j++){
        int a;
        int counter=0;
        for(a=1; a<=4; a++){
          int avg = 0;
          int k=i;
          for(; k<=i+2; k++){
            int l=j;
            for(; l<=j+2; l++){
              if(a==1)
                avg += local2[k][l]*first[k-i][l-j];
              else if(a==2)
                avg += local2[k][l]*second[k-i][l-j];
              else if(a==3)
                avg += local2[k][l]*third[k-i][l-j];
              else if(a==4)
                avg += local2[k][l]*fourth[k-i][l-j];
            }
          }
          if(avg>threshold){
            counter++;
            break;
          }
        }
        if(counter==1)
          result2[i][j] = 255;
        else if(counter==0)
          result2[i][j] = 0;
      }
    }

    if(rank!=1){
      if(x>1)
        MPI_Send(&(local2[0][0]), 2*198 , MPI_INT, rank-1, 0, MPI_COMM_WORLD);
      else if(x==1){
        MPI_Send(&(local2[0][0]), 198 , MPI_INT, rank-1, 0, MPI_COMM_WORLD);
        if(rank>2)
          MPI_Send(&(local2[0][0]), 198 , MPI_INT, rank-2, 0, MPI_COMM_WORLD);
      }
    }
    if(rank!=total-1){
      if(x>1)
        MPI_Recv(&(temporary2[0][0]), 2*198, MPI_INT, rank+1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      else if(x==1){
        MPI_Recv(&(temporary2[0][0]), 198, MPI_INT, rank+1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank<total-2)
          MPI_Recv(&(temporary2[1][0]), 198, MPI_INT, rank+2, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
    }

    if(rank!=total-1){
      for(; i<x; i++){
        int j=0;
        for(; j<198; j++){
          int counter = 0;
          int a;
          for(a=1; a<=4; a++){
            int m=j;
            int avg=0;
            for(; m<=j+2; m++){
              if(i==x-1){
                if(a==1)
                  avg += local2[i][m]*first[0][m-j] + temporary2[0][m]*first[1][m-j] + temporary2[1][m]*first[2][m-j];
                else if(a==2)
                  avg += local2[i][m]*second[0][m-j] + temporary2[0][m]*second[1][m-j] + temporary2[1][m]*second[2][m-j];
                else if(a==3)
                  avg += local2[i][m]*third[0][m-j] + temporary2[0][m]*third[1][m-j] + temporary2[1][m]*third[2][m-j];
                else if(a==4)
                  avg += local2[i][m]*fourth[0][m-j] + temporary2[0][m]*fourth[1][m-j] + temporary2[1][m]*fourth[2][m-j];
              }
              else if(i==x-2){
                if(a==1)
                  avg += local2[i][m]*first[0][m-j] + local2[i+1][m]*first[1][m-j] + temporary2[0][m]*first[2][m-j];
                else if(a==2)
                  avg += local2[i][m]*second[0][m-j] + local2[i+1][m]*second[1][m-j] + temporary2[0][m]*second[2][m-j];
                else if(a==3)
                  avg += local2[i][m]*third[0][m-j] + local2[i+1][m]*third[1][m-j] + temporary2[0][m]*third[2][m-j];
                else if(a==4)
                  avg += local2[i][m]*fourth[0][m-j] + local2[i+1][m]*fourth[1][m-j] + temporary2[0][m]*fourth[2][m-j];
              }
            }
            if(avg>threshold){
              counter++;
              break;
            }
          }

          if(counter==1)
            result2[i][j] = 255;
          else if(counter==0)
            result2[i][j] = 0;

        }


      }

    }
  }

  // collecting end values from slaves to master
  MPI_Gather(result2,x*196,MPI_INT,middle2,x*196,MPI_INT,0,MPI_COMM_WORLD);

  // write in the output file
  if(rank==0){
    FILE *cout = fopen(argv[2], "w");
    int i;
    for(i=x; i<196+x; i++){
      int j;
      for(j=0; j<196; j++){
        int temp = middle2[i][j];
        fprintf(cout, "%d ", temp);
      }
      fprintf(cout, "\n");
    }
    fclose(cout);
  }

  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}