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
    int       p;             /* Total number of processes    */
    MPI_Comm  comm;          /* Communicator for entire grid */
    MPI_Comm  row_comm;      /* Communicator for my row      */
    MPI_Comm  col_comm;      /* Communicator for my col      */
    int       q;             /* Order of grid                */
    int       my_row;        /* My row number                */
    int       my_col;        /* My column number             */
    int       world_rank;       /* My rank in the grid comm     */
} GRID_INFO_T;

#define MAX 65536000  // Maximum capacity of the array which holds the matrix
typedef struct{
    int     n_order_block;
    double  entries[MAX];
    #define Entry(A,i,j) (*(((A)->entries) + ((A)->n_order_block)*(i) + (j)))
} Block_Matrix;

void             C_zero_initialized(Block_Matrix* block_A);
void             readMatrix(Block_Matrix* block_A, GRID_INFO_T* grid, int n, char* filename);
void             Build_matrix_type(Block_Matrix* block_A);
void             Block_matrix_multiply(Block_Matrix* block_A, Block_Matrix* block_B, Block_Matrix* block_C);
void             Write_matrix_C(Block_Matrix* block_C, GRID_INFO_T* grid, int n,char* filename);
void             writeMatrixBlock(Block_Matrix* matrix, GRID_INFO_T* grid, char* filename);
MPI_Datatype     block_matrix_mpi_dt;
Block_Matrix*    temp_mat;

int main(int argc, char* argv[]) {
    FILE             *fp;
    GRID_INFO_T      grid;
    Block_Matrix*  block_A = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    Block_Matrix*  block_B = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    Block_Matrix*  block_C = (Block_Matrix*) malloc(sizeof(Block_Matrix));
    int              n_order=1,n_order_block,content,world_rank;
    double           start_t_MPI_Wtime,end_t_MPI_Wtime;
    clock_t          start_t, end_t;

    void Setup_grid(GRID_INFO_T*  grid);
    void Fox(int n, GRID_INFO_T* grid, Block_Matrix* block_A, Block_Matrix* block_B, Block_Matrix* block_C);

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Set OpenMP threads
    if (OPENMP==1)omp_set_num_threads(threads_n);

    Setup_grid(&grid);

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

    n_order_block = n_order/grid.q;//block the order of matrix
    MPI_Bcast(&n_order, 1, MPI_INT, 0, MPI_COMM_WORLD);//broadcast the order

    //read matrix and assign order
    block_A->n_order_block= n_order_block;
    readMatrix(block_A, &grid, n_order,Matrix_A_file);
    block_B->n_order_block= n_order_block;
    readMatrix(block_B, &grid, n_order,Matrix_B_file);
    Build_matrix_type(block_A);// Build block_A's MPI matrix data type
    temp_mat = (Block_Matrix*) malloc(sizeof(Block_Matrix));//temporary matrix
    block_C->n_order_block= n_order_block;

    // Main multiplication by using Fox Algorithm and it is parallel
    MPI_Barrier(MPI_COMM_WORLD);
    start_t = clock();
    start_t_MPI_Wtime = MPI_Wtime();
    Fox(n_order, &grid, block_A, block_B, block_C);
    end_t_MPI_Wtime = MPI_Wtime();
    end_t = clock();
    MPI_Barrier(MPI_COMM_WORLD);

    //Write relevant files
    Write_matrix_C(block_C, &grid, n_order, Matrix_C_file);
    writeMatrixBlock(block_A, &grid,Block_Matrix_A_file);
    writeMatrixBlock(block_B, &grid,Block_Matrix_B_file);
    writeMatrixBlock(block_C, &grid,Block_Matrix_C_file);

    //free resources
    free(block_A);
    free(block_B);
    free(block_C);

    //print time
    if(world_rank == 0){
        printf("The time for Parallel Fox algorithm execution(calculate from MPI_Wtime()): %30.20E seconds\n", end_t_MPI_Wtime-start_t_MPI_Wtime);
        printf("The time for Parallel Fox algorithm execution(calculate from clock()): %f seconds\n", (double)(end_t - start_t) / CLOCKS_PER_SEC);
    }

    MPI_Finalize();
    exit(0);
}


/*********************************************************/
void Setup_grid(
        GRID_INFO_T*  grid  /* out */) {
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /* Set up Global Grid Information */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    /* We assume p is a perfect square */     // but what if it's not a perfect square
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;

    /* We want a circular shift in second dimension. */
    /* Don't care about first                        */
    wrap_around[0] = wrap_around[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions,
                    wrap_around, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->world_rank));
    MPI_Cart_coords(grid->comm, grid->world_rank, 2,
                    coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* Set up row communicators */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords,
                 &(grid->row_comm));

    /* Set up column communicators */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords,
                 &(grid->col_comm));
} /* Setup_grid */

void Fox(
        int              n         /* in  */,
        GRID_INFO_T*     grid      /* in  */,
        Block_Matrix*  block_A   /* in  */,
        Block_Matrix*  block_B   /* in  */,
        Block_Matrix*  block_C   /* out */) {

    Block_Matrix*  temp_A; /* Storage for the sub-    */
    /* matrix of A used during */
    /* the current stage       */
    int              stage;
    int              bcast_root;
    int              n_bar;  /* n/sqrt(p)               */
    int              source;
    int              dest;
    MPI_Status       status;

    n_bar = n/grid->q;
    C_zero_initialized(block_C);

    /* Calculate addresses for row circular shift of B */
    source = (grid->my_row + 1) % grid->q;
    dest = (grid->my_row + grid->q - 1) % grid->q;

    /* Set aside storage for the broadcast block of A */
    temp_A = (Block_Matrix*) malloc(sizeof(Block_Matrix));

    for (stage = 0; stage < grid->q; stage++) {
        bcast_root = (grid->my_row + stage) % grid->q;
        if (bcast_root == grid->my_col) {
            MPI_Bcast(block_A, 1, block_matrix_mpi_dt,
                      bcast_root, grid->row_comm);
            Block_matrix_multiply(block_A, block_B,
                                  block_C);
        } else {
            MPI_Bcast(temp_A, 1, block_matrix_mpi_dt,
                      bcast_root, grid->row_comm);
            Block_matrix_multiply(temp_A, block_B,
                                  block_C);
        }
        MPI_Sendrecv_replace(block_B, 1, block_matrix_mpi_dt,
                             dest, 0, source, 0, grid->col_comm, &status);
    } /* for */

} /* Fox */




/* Read and distribute matrix for matrix A:
 *     foreach global row of the matrix,
 *         foreach grid column
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void readMatrix(
        Block_Matrix*  Block_M  /* out */,
        GRID_INFO_T*     grid     /* in  */,
        int              n        /* in  */,
        char*            filename   /* in  */) {

    FILE *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    double*     temp;
    MPI_Status status;

    if (grid->world_rank == 0) {  // Process 0 read matrix input from stdin and send them to other processess
        fp = fopen(filename,"r");
        temp = (double*) malloc(Block_M->n_order_block*sizeof(double));
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Block_M->n_order_block;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (mat_col = 0; mat_col < Block_M->n_order_block; mat_col++)
                        fscanf(fp, "%lf",
                               (Block_M->entries)+mat_row*Block_M->n_order_block+mat_col);

                } else {
                    for(mat_col = 0; mat_col < Block_M->n_order_block; mat_col++)
                        fscanf(fp,"%lf", temp + mat_col);
                    MPI_Send(temp, Block_M->n_order_block, MPI_DOUBLE, dest, 0,
                             grid->comm);
                }
            }
        }
        free(temp);
        fclose(fp);
        printf("Read from file %s finish\n", filename);
    } else {  // Other processess receive matrix from process 0
        for (mat_row = 0; mat_row < Block_M->n_order_block; mat_row++)
            MPI_Recv(&Entry(Block_M, mat_row, 0), Block_M->n_order_block,
                     MPI_DOUBLE, 0, 0, grid->comm, &status);
    }

}  /* Read_matrix */

/*********************************************************/
/* Read and distribute matrix for matrix A:
 *     foreach global row of the matrix,
 *         foreach grid column
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void readA(
        char*            prompt   /* in  */,
        Block_Matrix*  block_A  /* out */,
        GRID_INFO_T*     grid     /* in  */,
        int              n        /* in  */) {

    FILE *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    double*     temp;
    MPI_Status status;

    if (grid->world_rank == 0) {  // Process 0 read matrix input from stdin and send them to other processess
        fp = fopen(Matrix_A_file,"r");
        temp = (double*) malloc(block_A->n_order_block*sizeof(double));
        printf("%s\n", prompt);
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/block_A->n_order_block;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (mat_col = 0; mat_col < block_A->n_order_block; mat_col++)
                        fscanf(fp, "%lf",
                               (block_A->entries)+mat_row*block_A->n_order_block+mat_col);

                } else {
                    for(mat_col = 0; mat_col < block_A->n_order_block; mat_col++)
                        fscanf(fp,"%lf", temp + mat_col);
                    MPI_Send(temp, block_A->n_order_block, MPI_DOUBLE, dest, 0,
                             grid->comm);
                }
            }
        }
        free(temp);
        fclose(fp);
    } else {  // Other processess receive matrix from process 0
        for (mat_row = 0; mat_row < block_A->n_order_block; mat_row++)
            MPI_Recv(&Entry(block_A, mat_row, 0), block_A->n_order_block,
                     MPI_DOUBLE, 0, 0, grid->comm, &status);
    }

}  /* Read_matrix */


/*********************************************************/
/* Read and distribute matrix for local matrix B's transpose:
 *     foreach global row of the matrix,
 *         foreach grid column
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */

void readB(
        char*            prompt   /* in  */,
        Block_Matrix*  block_B  /* out */,
        GRID_INFO_T*     grid     /* in  */,
        int              n        /* in  */) {

    FILE       *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    double      *temp;
    MPI_Status status;

    if (grid->world_rank == 0) {  // Process 0 read matrix input from stdin and send them to other processess
        fp = fopen(Matrix_B_file,"r");
        temp = (double*) malloc(block_B->n_order_block*sizeof(double));
        printf("%s\n", prompt);
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/block_B->n_order_block;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {                                                    // process 0 (local)
                    for (mat_col = 0; mat_col < block_B->n_order_block; mat_col++)
                        fscanf(fp, "%lf",
                               (block_B->entries)+mat_col*block_B->n_order_block+mat_row);       // switch rows and colums in block_B, for column major storage
                    /* scanf("%lf",
                      (block_B->entries)+mat_col*Order(block_B)+mat_row);       // switch rows and colums in block_B, for column major storage
                    */
                    /* scanf("%lf",
                      (block_A->entries)+mat_row*Order(block_A)+mat_col); */
                } else {
                    for(mat_col = 0; mat_col < block_B->n_order_block; mat_col++)
                        fscanf(fp, "%lf", temp + mat_col);
                    // scanf("%lf", temp + mat_col);
                    MPI_Send(temp, block_B->n_order_block, MPI_DOUBLE, dest, 0,
                             grid->comm);
                }
            }
        }
        free(temp);
        fclose(fp);
    } else {  // Other processess receive matrix from process 0
        temp = (double*) malloc(block_B->n_order_block*sizeof(double));               // switch rows and colums in block_B, for column major storage
        for (mat_col = 0; mat_col < block_B->n_order_block; mat_col++) {
            MPI_Recv(temp, block_B->n_order_block,
                     MPI_DOUBLE, 0, 0, grid->comm, &status);                      // switch rows and colums in block_B, for column major storage
            for(mat_row = 0; mat_row < block_B->n_order_block; mat_row++)
                Entry(block_B, mat_row, mat_col) = *(temp + mat_row);       // switch rows and colums in block_B, for column major storage

            /* MPI_Recv(&Entry(block_A, mat_row, 0), Order(block_A),
                MPI_DOUBLE, 0, 0, grid->comm, &status); */
        }
        free(temp);
    }

}  /* readB */







/*********************************************************/
/* Recive and Write Matrix C into a file:
 *     foreach global row of the matrix,
 *         foreach grid column
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them
 */
void Write_matrix_C(
        Block_Matrix*  block_C  /* out */,
        GRID_INFO_T*     grid     /* in  */,
        int              n        /* in  */,
        char*            filename    /* in  */) {

    FILE      *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    double*     temp;
    MPI_Status status;

    if (grid->world_rank == 0) {
        fp = fopen(filename, "w+");
        temp = (double*) malloc(block_C->n_order_block*sizeof(double));

        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/block_C->n_order_block;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < block_C->n_order_block; mat_col++)
                        fprintf(fp, "%20.15E ", Entry(block_C, mat_row, mat_col));
                    // printf("%20.15E ", Entry(block_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, block_C->n_order_block, MPI_DOUBLE, source, 0,
                             grid->comm, &status);
                    for(mat_col = 0; mat_col < block_C->n_order_block; mat_col++)
                        fprintf(fp, "%20.15E ", temp[mat_col]);
                    // printf("%20.15E ", temp[mat_col]);
                }
            }
            fprintf(fp,"\n");
        }
        free(temp);
        fclose(fp);
        printf("Write to file %s finish\n", filename);
    } else {
        for (mat_row = 0; mat_row < block_C->n_order_block; mat_row++)
            MPI_Send(&Entry(block_C, mat_row, 0), block_C->n_order_block,
                     MPI_DOUBLE, 0, 0, grid->comm);
    }

}  /* Write_matrix_C */

void writeMatrixBlock(Block_Matrix* matrix,GRID_INFO_T* grid, char* filename){
    FILE        *fp;
    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;

    // print by process No.0 in process mesh
    if (grid->world_rank == 0) {
        fp = fopen(filename,"w+");
        fprintf(fp,"Process %d > grid_row = %d, grid_col = %d\n",
                grid->world_rank, grid->my_row, grid->my_col);
        for (i = 0; i < matrix->n_order_block; i++) {
            for (j = 0; j < matrix->n_order_block; j++)
                fprintf(fp,"%20.15E ", Entry(matrix,i,j));
            fprintf(fp, "\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, block_matrix_mpi_dt, source, 0,
                     grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
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
        MPI_Send(matrix, 1, block_matrix_mpi_dt, 0, 0, grid->comm);
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




