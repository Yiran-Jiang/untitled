#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <omp.h>

#define Matrix_A_file "M_A"
#define Matrix_B_file "M_B"
#define Matrix_C_file "M_C"
#define Block_Matrix_A_file "M_A_Block"
#define Block_Matrix_B_file "M_B_Block"
#define Block_Matrix_C_file "M_C_Block"
#define OPENMP 0
#define threads_n 2

typedef struct {
    int       num_of_process,row_coords_process,col_coords_process,order_grid,world_rank;
    MPI_Comm  com,com_row,com_col;
} GRID;

#define MAX 65536000  // Maximum capacity of the array which holds the matrix
typedef struct{
    int     n_order_block;
    double  entries[MAX];
    #define Entry(A,i,j) (*(((A)->entries) + ((A)->n_order_block)*(i) + (j)))
} Block_Matrix;

void             C_zero_initialized(Block_Matrix* block_A);
void             readMatrix(Block_Matrix* block_A, GRID* grid_global, int n, char* filename);
void             Build_matrix_type(Block_Matrix* block_A);
void             Block_matrix_multiply(Block_Matrix* block_A, Block_Matrix* block_B, Block_Matrix* block_C);
void             Write_matrix_C(Block_Matrix* block_C, GRID* grid_global, int n,char* filename);
void             writeMatrixBlock(Block_Matrix* matrix, GRID* grid_global, char* filename);
MPI_Datatype     block_matrix_mpi_dt;
Block_Matrix*    temp_mat;

int main(int argc, char* argv[]) {
    FILE             *fp;
    GRID      grid_global;
    Block_Matrix*  block_A = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    Block_Matrix*  block_B = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    Block_Matrix*  block_C = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    int              n_order=1,n_order_block,content,world_rank;
    double           start_t_MPI_Wtime,end_t_MPI_Wtime;
    clock_t          time_s, time_e;

    void Init_grid(GRID*  grid_global);
    void Fox(GRID* grid_global, Block_Matrix* block_A, Block_Matrix* block_B, Block_Matrix* block_C);

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Set OpenMP threads
    if (OPENMP==1)omp_set_num_threads(threads_n);

    Init_grid(&grid_global);

    //Count the number of rows as mix order
    if (world_rank == 0) {
        fp = fopen(Matrix_A_file,"r");
        while((content = fgetc(fp)) != EOF)
        {
            if(content =='\n') n_order++;
        }
        fclose(fp);
        printf("The order is %d\n", n_order);
    }

    n_order_block = n_order/grid_global.order_grid;//block the order of matrix
    MPI_Bcast(&n_order, 1, MPI_INT, 0, MPI_COMM_WORLD);//broadcast the order

    //read matrix and assign order
    block_A->n_order_block= n_order_block;
    readMatrix(block_A, &grid_global, n_order,Matrix_A_file);
    block_B->n_order_block= n_order_block;
    readMatrix(block_B, &grid_global, n_order,Matrix_B_file);
    Build_matrix_type(block_A);// Build block_A's MPI matrix data type
    temp_mat = (Block_Matrix*) malloc(sizeof(Block_Matrix));//temporary matrix
    block_C->n_order_block= n_order_block;

    // Main multiplication by using Fox Algorithm and it is parallel
    MPI_Barrier(MPI_COMM_WORLD);
    time_s = clock();
    start_t_MPI_Wtime = MPI_Wtime();
    Fox(&grid_global, block_A, block_B, block_C);
    end_t_MPI_Wtime = MPI_Wtime();
    time_e = clock();
    MPI_Barrier(MPI_COMM_WORLD);

    //Write relevant files
    Write_matrix_C(block_C, &grid_global, n_order, Matrix_C_file);
    writeMatrixBlock(block_A, &grid_global,Block_Matrix_A_file);
    writeMatrixBlock(block_B, &grid_global,Block_Matrix_B_file);
    writeMatrixBlock(block_C, &grid_global,Block_Matrix_C_file);

    //free resources
    free(block_A);
    free(block_B);
    free(block_C);

    //print time
    if(world_rank == 0){
        printf("The time for Parallel Fox algorithm execution(calculate from MPI_Wtime()): %30.20E seconds\n", end_t_MPI_Wtime-start_t_MPI_Wtime);
        printf("The time for Parallel Fox algorithm execution(calculate from clock()): %f seconds\n", (double)(time_e - time_s) / CLOCKS_PER_SEC);
    }

    MPI_Finalize();
    exit(0);
}


/*********************************************************/
void Init_grid(GRID*  grid_global) {
    int i,reorder=1,ndims_cart_grid=2;
    int row_col_dim[ndims_cart_grid],periodic_dim[ndims_cart_grid],coords_process[ndims_cart_grid];


    for(i=0;i<ndims_cart_grid;i++)
    {
        periodic_dim[i]=1;
    }

    MPI_Comm_rank(grid_global->com, &(grid_global->world_rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(grid_global->num_of_process));
    grid_global->order_grid = (int) sqrt((double) grid_global->num_of_process);
    for(i=0;i<ndims_cart_grid;i++)
    {
        row_col_dim[i]=grid_global->order_grid;
    }

    MPI_Cart_create(MPI_COMM_WORLD, ndims_cart_grid, row_col_dim,periodic_dim, reorder, &(grid_global->com));
    MPI_Cart_coords(grid_global->com, grid_global->world_rank, ndims_cart_grid,coords_process);
    grid_global->row_coords_process = coords_process[0];
    grid_global->col_coords_process = coords_process[1];

    int keep_dims_row[ndims_cart_grid];
    for(i=0;i<ndims_cart_grid;i++)
    {
        keep_dims_row[i]=i;
    }

    int keep_dims_col[ndims_cart_grid];
    for(i=ndims_cart_grid-1;i>-1;i--)
    {
        keep_dims_col[i]=i;
    }

    MPI_Cart_sub(grid_global->com, keep_dims_row,&(grid_global->com_row));
    MPI_Cart_sub(grid_global->com, keep_dims_col,&(grid_global->com_col));
}

void Fox(GRID* grid_global, Block_Matrix*  block_A, Block_Matrix*  block_B, Block_Matrix*  block_C) {
    Block_Matrix*  tempA_sub = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    int root, curr_step,source,destination,numcount=1,tag=0;
    MPI_Status status;

    C_zero_initialized(block_C);

    source = (grid_global->row_coords_process + 1) % grid_global->order_grid;
    destination = (grid_global->row_coords_process + grid_global->order_grid - 1) % grid_global->order_grid;

    for (curr_step = 0; curr_step < grid_global->order_grid; curr_step++) {
        root = (grid_global->row_coords_process + curr_step) % grid_global->order_grid;
        if (root != grid_global->col_coords_process) {
            MPI_Bcast(tempA_sub, numcount, block_matrix_mpi_dt,root, grid_global->com_row);
            Block_matrix_multiply(tempA_sub, block_B,block_C);
        } else {
            MPI_Bcast(block_A, numcount, block_matrix_mpi_dt,root, grid_global->com_row);
            Block_matrix_multiply(block_A, block_B,block_C);
        }
        MPI_Sendrecv_replace(block_B, numcount, block_matrix_mpi_dt, destination, tag, source, tag, grid_global->com_col, &status);
    } 

}




/* Read and distribute matrix for matrix A:
 *     foreach global row_Matrix of the matrix,
 *         foreach grid_global column
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void readMatrix(Block_Matrix*  Block_M, GRID* grid_global, int n, char* filename) {
    FILE *fp;
    int        row_Matrix, col_Matrix, row_Grid, col_Grid, destination, coords[2];
    double*     temp;
    MPI_Status status;

    if (grid_global->world_rank == 0) {
        fp = fopen(filename,"r");
        temp = (double*) malloc(Block_M->n_order_block*sizeof(double));
        fflush(stdout);
        for (row_Matrix = 0;  row_Matrix < n; row_Matrix++) {
            row_Grid = row_Matrix/Block_M->n_order_block;
            coords[0] = row_Grid;
            for (col_Grid = 0; col_Grid < grid_global->order_grid; col_Grid++) {
                coords[1] = col_Grid;
                MPI_Cart_rank(grid_global->com, coords, &destination);
                if (destination == 0) {
                    for (col_Matrix = 0; col_Matrix < Block_M->n_order_block; col_Matrix++)
                        fscanf(fp, "%lf",
                               (Block_M->entries)+row_Matrix*Block_M->n_order_block+col_Matrix);

                } else {
                    for(col_Matrix = 0; col_Matrix < Block_M->n_order_block; col_Matrix++)
                        fscanf(fp,"%lf", temp + col_Matrix);
                    MPI_Send(temp, Block_M->n_order_block, MPI_DOUBLE, destination, 0,
                             grid_global->com);
                }
            }
        }
        free(temp);
        fclose(fp);
        printf("Read from file %s finish\n", filename);
    } else {  // Other processess receive matrix from process 0
        for (row_Matrix = 0; row_Matrix < Block_M->n_order_block; row_Matrix++)
            MPI_Recv(&Entry(Block_M, row_Matrix, 0), Block_M->n_order_block,
                     MPI_DOUBLE, 0, 0, grid_global->com, &status);
    }

}

/*********************************************************/
/* Recive and Write Matrix C into a file:
 *     foreach global row_Matrix of the matrix,
 *         foreach grid_global column
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them
 */
void Write_matrix_C(
        Block_Matrix*  block_C  /* out */,
        GRID*     grid_global     /* in  */,
        int              n        /* in  */,
        char*            filename    /* in  */) {

    FILE      *fp;
    int        row_Matrix, col_Matrix;
    int        row_Grid, col_Grid;
    int        source;
    int        coords[2];
    double*     temp;
    MPI_Status status;

    if (grid_global->world_rank == 0) {
        fp = fopen(filename, "w+");
        temp = (double*) malloc(block_C->n_order_block*sizeof(double));

        for (row_Matrix = 0;  row_Matrix < n; row_Matrix++) {
            row_Grid = row_Matrix/block_C->n_order_block;
            coords[0] = row_Grid;
            for (col_Grid = 0; col_Grid < grid_global->order_grid; col_Grid++) {
                coords[1] = col_Grid;
                MPI_Cart_rank(grid_global->com, coords, &source);
                if (source == 0) {
                    for(col_Matrix = 0; col_Matrix < block_C->n_order_block; col_Matrix++)
                        fprintf(fp, "%20.15E ", Entry(block_C, row_Matrix, col_Matrix));
                    // printf("%20.15E ", Entry(block_A, row_Matrix, col_Matrix));
                } else {
                    MPI_Recv(temp, block_C->n_order_block, MPI_DOUBLE, source, 0,
                             grid_global->com, &status);
                    for(col_Matrix = 0; col_Matrix < block_C->n_order_block; col_Matrix++)
                        fprintf(fp, "%20.15E ", temp[col_Matrix]);
                    // printf("%20.15E ", temp[col_Matrix]);
                }
            }
            fprintf(fp,"\n");
        }
        free(temp);
        fclose(fp);
        printf("Write to file %s finish\n", filename);
    } else {
        for (row_Matrix = 0; row_Matrix < block_C->n_order_block; row_Matrix++)
            MPI_Send(&Entry(block_C, row_Matrix, 0), block_C->n_order_block,
                     MPI_DOUBLE, 0, 0, grid_global->com);
    }

}  /* Write_matrix_C */

void writeMatrixBlock(Block_Matrix* matrix,GRID* grid_global, char* filename){
    FILE        *fp;
    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;

    // print by process No.0 in process mesh
    if (grid_global->world_rank == 0) {
        fp = fopen(filename,"w+");
        fprintf(fp,"Process %d > row_Grid = %d, col_Grid = %d\n",
                grid_global->world_rank, grid_global->row_coords_process, grid_global->col_coords_process);
        for (i = 0; i < matrix->n_order_block; i++) {
            for (j = 0; j < matrix->n_order_block; j++)
                fprintf(fp,"%20.15E ", Entry(matrix,i,j));
            fprintf(fp, "\n");
        }
        for (source = 1; source < grid_global->num_of_process; source++) {
            MPI_Recv(temp_mat, 1, block_matrix_mpi_dt, source, 0,
                     grid_global->com, &status);
            MPI_Cart_coords(grid_global->com, source, 2, coords);
            fprintf(fp, "Process %d > row_Grid = %d, col_Grid = %d\n",
                    source, coords[0], coords[1]);
            for (i = 0; i < temp_mat->n_order_block; i++) {
                for (j = 0; j < temp_mat->n_order_block; j++)
                    fprintf(fp, "%20.15E ", Entry(temp_mat,i,j));
                fprintf(fp, "\n");
            }
        }
        fflush(stdout);
        fclose(fp);
        printf("Write to file %s finish\n",filename);
    } else {
        MPI_Send(matrix, 1, block_matrix_mpi_dt, 0, 0, grid_global->com);
    }
}

void C_zero_initialized(Block_Matrix*  matrix) {
    int i, j;
    for (i = 0; i < matrix->n_order_block; i++)
        for (j = 0; j < matrix->n_order_block; j++)
            Entry(matrix,i,j) = 0.0E0;
}


/*********************************************************/
void Build_matrix_type(Block_Matrix*  block_A) {
    MPI_Datatype  temp_mpi_t;
    int           block_lengths[2];
    MPI_Aint      displacements[2];
    MPI_Datatype  typelist[2];
    MPI_Aint      start_address;
    MPI_Aint      address;

    MPI_Type_contiguous(block_A->n_order_block*block_A->n_order_block,
                        MPI_DOUBLE, &temp_mpi_t);                         // Creates a contiguous datatype
    /*
    Synopsis
           int MPI_Type_contiguous(int count,
                             MPI_Datatype oldtype,
                             MPI_Datatype *newtype)
    Input Parameters
    count
           replication count (nonnegative integer)
    oldtype
           old datatype (handle)
    */
    block_lengths[0] = block_lengths[1] = 1;

    typelist[0] = MPI_INT;
    typelist[1] = temp_mpi_t;

    MPI_Get_address(block_A, &start_address);                 // Gets the address of a location in caller's memory
    MPI_Get_address(&(block_A->n_order_block), &address);
    /*
    Synopsis
           int MPI_Address(const void *location, MPI_Aint *address)

    Input Parameters
    location
           location in caller memory (choice)

    Output Parameters
           address
           address of location (address integer)
    */
    displacements[0] = address - start_address;

    MPI_Get_address(block_A->entries, &address);
    displacements[1] = address - start_address;

    MPI_Type_create_struct(2, block_lengths, displacements,
                    typelist, &block_matrix_mpi_dt);                   // Creates a struct datatype
    /*
    Synopsis
    int MPI_Type_struct(int count,
                      const int *array_of_blocklengths,
                      const MPI_Aint *array_of_displacements,
                      const MPI_Datatype *array_of_types,
                      MPI_Datatype *newtype)

    Input Parameters
    count
        number of blocks (integer) -- also number of entries in arrays array_of_types , array_of_displacements and array_of_blocklengths
    array_of_blocklengths
        number of elements in each block (array)
    array_of_displacements
        byte displacement of each block (array)
    array_of_types
        type of elements in each block (array of handles to datatype objects)

    Output Parameters

    newtype
        new datatype (handle)
    */
    MPI_Type_commit(&block_matrix_mpi_dt);                 // Commits the datatype
    /*
    Synopsis
    int MPI_Type_commit(MPI_Datatype *datatype)

    Input Parameters
    datatype
        datatype (handle)
    */
}


/*********************************************************/
/* local matrix multiplication function
*  withing OpenMP Thread Acceleration
*/
void Block_matrix_multiply(
        Block_Matrix*  block_A  /* in  */,
        Block_Matrix*  block_B  /* in  */,
        Block_Matrix*  block_C  /* out */) {

    int i, j, k;
#pragma omp parallel for private(i, j, k) shared(block_A, block_B, block_C) num_threads(threads_n)       // Threads acceleration upgrade, parallel task split
    for (i = 0; i < block_A->n_order_block; i++) {
        for (j = 0; j < block_A->n_order_block; j++)
            for (k = 0; k < block_B->n_order_block; k++)
                Entry(block_C,i,j) = Entry(block_C,i,j)             // switch rows and colums in block_B, for column major storage
                                     + Entry(block_A,i,k)*Entry(block_B,j,k);        // continuous memory access, local matrix multiplication A(i,k)*B^T(j,k)
    }
}




