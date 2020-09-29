#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

#define Matrix_A_file "M_A"
#define Matrix_B_file "M_B"
#define Matrix_C_file "M_C"
#define threads_n 2
#define MAX 65536000

typedef struct {
    int       num_of_process,row_coords_process,col_coords_process,order_grid,world_rank;
    MPI_Comm  com,com_row,com_col;
} GRID;

typedef struct {
    int     n_order_block;
    double  entries[MAX];
#define Entry(A,i,j) (*(((A)->entries) + ((A)->n_order_block)*(i) + (j)))
} Block_Matrix;
int              check(int N,char* filenameA,char* filenameB,char* filenameC);
void             readMatrixA(Block_Matrix*  block_A,GRID* grid_global,int n, char* filename);
void             readMatrixB(Block_Matrix* block_B,GRID* grid_global, int n,char* filename);
void             C_zero_initialized(Block_Matrix* matrix);
void             Block_matrix_multiply(Block_Matrix* block_A,Block_Matrix* block_B, Block_Matrix* block_C);
void             create_A_type_MPI_dt(Block_Matrix* block_M);
void             Write_matrix_C(Block_Matrix*  block_C ,GRID* grid_global,int n,char* filename);
void             Init_grid(GRID* grid_global);
void             Fox(GRID* grid_global, Block_Matrix*  block_A, Block_Matrix*  block_B, Block_Matrix*  block_C);
MPI_Datatype     block_matrix_mpi_dt;

int main() {
    FILE             *fp;
    GRID             grid_global;
    Block_Matrix*  block_A=(Block_Matrix*) malloc(sizeof(Block_Matrix));
    Block_Matrix*  block_B=(Block_Matrix*) malloc(sizeof(Block_Matrix));
    Block_Matrix*  block_C=(Block_Matrix*) malloc(sizeof(Block_Matrix));
    int              world_rank,recv,n_order=1,n_order_block;
    double           start_t_MPI_Wtime, end_t_MPI_Wtime;
    clock_t          time_s, time_e;

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initial OpenMP Environment
    omp_set_num_threads(threads_n);

    Init_grid(&grid_global);
    if (world_rank == 0) {
        fp = fopen(Matrix_A_file,"r");
        while((recv = fgetc(fp)) != EOF)
        {
            if(recv =='\n') n_order++;
        }
        fclose(fp);
        printf("The order is %d\n", n_order);
    }
    MPI_Bcast(&n_order, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n_order_block = n_order/grid_global.order_grid;

    block_A->n_order_block = n_order_block;
    readMatrixA(block_A, &grid_global, n_order,Matrix_A_file);

    block_B->n_order_block = n_order_block;
    readMatrixB(block_B, &grid_global, n_order,Matrix_B_file);

    block_C->n_order_block = n_order_block;
    create_A_type_MPI_dt(block_A);

    MPI_Barrier(MPI_COMM_WORLD);
    time_s = clock();
    start_t_MPI_Wtime = MPI_Wtime();
    Fox(&grid_global, block_A, block_B, block_C);
    end_t_MPI_Wtime = MPI_Wtime();
    time_e = clock();
    MPI_Barrier(MPI_COMM_WORLD);

    Write_matrix_C( block_C, &grid_global, n_order, Matrix_C_file);

    free(block_A);
    free(block_B);
    free(block_C);

    if(world_rank == 0){
        printf("The time for Parallel Fox algorithm execution(calculate from MPI_Wtime()): %30.20E seconds\n", end_t_MPI_Wtime-start_t_MPI_Wtime);
        printf("The time for Parallel Fox algorithm execution(calculate from clock()): %f seconds\n", (double)(time_e - time_s) / CLOCKS_PER_SEC);
    }
    MPI_Finalize();
    int res = check(n_order,Matrix_A_file,Matrix_B_file,Matrix_C_file);
    printf("%d\n",res);

    exit(0);
}

void Init_grid(GRID* grid_global) {
    int i,reorder=1,ndims_cart_grid=2;
    int row_col_dim[ndims_cart_grid],periodic_dim[ndims_cart_grid],coords_process[ndims_cart_grid];
    int keep_dims_row[ndims_cart_grid];
    int keep_dims_col[ndims_cart_grid];

    MPI_Comm_size(MPI_COMM_WORLD, &(grid_global->num_of_process));

    grid_global->order_grid = (int) sqrt((double) grid_global->num_of_process);
    for(i=0;i<ndims_cart_grid;i++)
    {
        row_col_dim[i]=grid_global->order_grid;
        keep_dims_row[i]=i;
        periodic_dim[i]=1;
        keep_dims_col[i]=1-i;
    }
    periodic_dim[0] = periodic_dim[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, ndims_cart_grid, row_col_dim,periodic_dim, reorder, &(grid_global->com));
    MPI_Comm_rank(grid_global->com, &(grid_global->world_rank));
    MPI_Cart_coords(grid_global->com, grid_global->world_rank, ndims_cart_grid, coords_process);
    grid_global->row_coords_process = coords_process[0];
    grid_global->col_coords_process = coords_process[1];

    MPI_Cart_sub(grid_global->com, keep_dims_row,&(grid_global->com_row));
    MPI_Cart_sub(grid_global->com, keep_dims_col,&(grid_global->com_col));
}

void Fox(GRID* grid_global, Block_Matrix*  block_A, Block_Matrix*  block_B, Block_Matrix*  block_C) {
    Block_Matrix*  temp_A =(Block_Matrix*) malloc(sizeof(Block_Matrix));
    int root, curr_step,source,destination,count=1,tag=0;
    MPI_Status       status;

    C_zero_initialized(block_C);

    source = (grid_global->row_coords_process + 1) % grid_global->order_grid;
    destination = (grid_global->row_coords_process + grid_global->order_grid - 1) % grid_global->order_grid;

    for (curr_step = 0; curr_step < grid_global->order_grid; curr_step++) {
        root = (grid_global->row_coords_process + curr_step) % grid_global->order_grid;
        if (root == grid_global->col_coords_process) {
            MPI_Bcast(block_A, count, block_matrix_mpi_dt, root, grid_global->com_row);
            Block_matrix_multiply(block_A, block_B,block_C);
        } else {
            MPI_Bcast(temp_A, count, block_matrix_mpi_dt, root, grid_global->com_row);
            Block_matrix_multiply(temp_A, block_B,block_C);
        }
        MPI_Sendrecv_replace(block_B, count, block_matrix_mpi_dt, destination, tag, source, tag, grid_global->com_col, &status);
    }

}

void readMatrixA(Block_Matrix*  block_A,GRID* grid_global,int n, char* filename) {
    FILE *fp;
    int        row_Matrix, col_Matrix, row_Grid, col_Grid, destination, row_col_coords[2];
    double*     temp;
    MPI_Status status;

    if (grid_global->world_rank == 0) {
        fp = fopen(filename,"r");
        temp = (double*) malloc(block_A->n_order_block*sizeof(double));
        fflush(stdout);
        for (row_Matrix = 0;  row_Matrix < n; row_Matrix++) {
            row_Grid = row_Matrix/block_A->n_order_block;
            row_col_coords[0] = row_Grid;
            for (col_Grid = 0; col_Grid < grid_global->order_grid; col_Grid++) {
                row_col_coords[1] = col_Grid;
                MPI_Cart_rank(grid_global->com, row_col_coords, &destination);
                if (destination == 0) {
                    for (col_Matrix = 0; col_Matrix < block_A->n_order_block; col_Matrix++)
                        fscanf(fp, "%lf",
                               (block_A->entries)+row_Matrix*block_A->n_order_block+col_Matrix);
                } else {
                    for(col_Matrix = 0; col_Matrix < block_A->n_order_block; col_Matrix++)
                        fscanf(fp,"%lf", temp + col_Matrix);
                    MPI_Send(temp, block_A->n_order_block, MPI_DOUBLE, destination, 0,
                             grid_global->com);
                }
            }
        }
        free(temp);
        fclose(fp);
        printf("Read from file %s finish\n", filename);
    } else {
        for (row_Matrix = 0; row_Matrix < block_A->n_order_block; row_Matrix++)
            MPI_Recv(&Entry(block_A, row_Matrix, 0), block_A->n_order_block,
                     MPI_DOUBLE, 0, 0, grid_global->com, &status);
    }
}

void readMatrixB(Block_Matrix*  block_B,GRID* grid_global,int n, char* filename) {
    FILE *fp;
    int        row_Matrix, col_Matrix, row_Grid, col_Grid, destination, row_col_coords[2];
    double*     temp;
    MPI_Status status;

    if (grid_global->world_rank == 0) {
        fp = fopen(filename,"r");
        temp = (double*) malloc(block_B->n_order_block*sizeof(double));
        fflush(stdout);
        for (row_Matrix = 0;  row_Matrix < n; row_Matrix++) {
            row_Grid = row_Matrix/block_B->n_order_block;
            row_col_coords[0] = row_Grid;
            for (col_Grid = 0; col_Grid < grid_global->order_grid; col_Grid++) {
                row_col_coords[1] = col_Grid;
                MPI_Cart_rank(grid_global->com, row_col_coords, &destination);
                if (destination == 0) {
                    for (col_Matrix = 0; col_Matrix < block_B->n_order_block; col_Matrix++)
                        fscanf(fp, "%lf",(block_B->entries)+col_Matrix*block_B->n_order_block+row_Matrix);
                } else {
                    for(col_Matrix = 0; col_Matrix < block_B->n_order_block; col_Matrix++)
                        fscanf(fp, "%lf", temp + col_Matrix);
                    MPI_Send(temp, block_B->n_order_block, MPI_DOUBLE, destination, 0,grid_global->com);
                }
            }
        }
        free(temp);
        fclose(fp);
        printf("Read from file %s finish\n", filename);
    } else {
        temp = (double *) malloc(block_B->n_order_block * sizeof(double));
        for (col_Matrix = 0; col_Matrix < block_B->n_order_block; col_Matrix++) {
            MPI_Recv(temp, block_B->n_order_block, MPI_DOUBLE, 0, 0, grid_global->com, &status);
            for (row_Matrix = 0; row_Matrix < block_B->n_order_block; row_Matrix++)
                Entry(block_B, row_Matrix, col_Matrix) = *(temp + row_Matrix);
        }
        free(temp);
    }
}

void Write_matrix_C(Block_Matrix*  block_C ,GRID* grid_global,int n,char* filename) {
    FILE      *fp;
    int        row_Matrix, col_Matrix, row_Grid, col_Grid, source, row_col_coords[2];
    double*     temp;
    MPI_Status status;

    if (grid_global->world_rank == 0) {
        fp = fopen(filename, "w+");
        temp = (double*) malloc(block_C->n_order_block*sizeof(double));
        for (row_Matrix = 0;  row_Matrix < n; row_Matrix++) {
            row_Grid = row_Matrix/block_C->n_order_block;
            row_col_coords[0] = row_Grid;
            for (col_Grid = 0; col_Grid < grid_global->order_grid; col_Grid++) {
                row_col_coords[1] = col_Grid;
                MPI_Cart_rank(grid_global->com, row_col_coords, &source);
                if (source == 0) {
                    for(col_Matrix = 0; col_Matrix < block_C->n_order_block; col_Matrix++)
                        fprintf(fp, "%20.15E ", Entry(block_C, row_Matrix, col_Matrix));
                } else {
                    MPI_Recv(temp, block_C->n_order_block, MPI_DOUBLE, source, 0,
                             grid_global->com, &status);
                    for(col_Matrix = 0; col_Matrix < block_C->n_order_block; col_Matrix++)
                        fprintf(fp, "%20.15E ", temp[col_Matrix]);
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
}

void C_zero_initialized(Block_Matrix *  matrix) {
    int i, j;
    for (i = 0; i < matrix->n_order_block; i++)
        for (j = 0; j < matrix->n_order_block; j++)
            Entry(matrix,i,j) = 0.0E0;
}

void create_A_type_MPI_dt(Block_Matrix*  block_M) {
    MPI_Datatype  new_mpi_dt;
    int num_blocks=2,i;

    int           blocklengths[num_blocks];
    MPI_Aint      displacements[num_blocks];
    MPI_Datatype  types[num_blocks];

    MPI_Aint      start_address;
    MPI_Aint      end_address;

    MPI_Type_contiguous((int)pow(block_M->n_order_block,2),MPI_DOUBLE, &new_mpi_dt);

    for(i=0;i<num_blocks;i++) {
        blocklengths[i] = 1;
    }

    types[0] = MPI_INT;
    types[1] = new_mpi_dt;

    MPI_Get_address(block_M, &start_address);

    MPI_Get_address(&(block_M->n_order_block), &end_address);
    displacements[0] = end_address - start_address;
    MPI_Get_address(block_M->entries, &end_address);
    displacements[1] = end_address - start_address;

    MPI_Type_create_struct(num_blocks, blocklengths, displacements,types, &block_matrix_mpi_dt);
    MPI_Type_commit(&block_matrix_mpi_dt);
}

/* local matrix multiplication function
*  withing OpenMP Thread Acceleration
*/
void Block_matrix_multiply(Block_Matrix*  block_A, Block_Matrix*  block_B,Block_Matrix*  block_C) {
    int i, j, k,world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

#pragma omp parallel for private(i, j, k) shared(block_A, block_B, block_C) num_threads(threads_n)
    for (i = 0; i < block_A->n_order_block; i++) {
        //printf("In Fox multiply,Process id: %d, Thread id: %d\n",world_rank,omp_get_thread_num());
        for (j = 0; j < block_A->n_order_block; j++)
            for (k = 0; k < block_B->n_order_block; k++)
                Entry(block_C,i,j) = Entry(block_C,i,j)+ Entry(block_A,i,k)*Entry(block_B,j,k);//switch rows and columns continuous memory access
    }
}

int check(int N,char* filenameA,char* filenameB,char* filenameC){

    FILE *fp_A;
    FILE *fp_B;
    FILE *fp_C;
    double a[N][N],b[N][N],c[N][N];
    int i,j,k;
    fp_A=fopen(filenameA,"r");
    fp_B=fopen(filenameB,"r");
    fp_C=fopen(filenameC,"r");

    printf("Read Matrix:\n");
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            fscanf(fp_A,"%lf",&a[i][j]);
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            fscanf(fp_B,"%lf",&b[i][j]);
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            fscanf(fp_C,"%lf",&c[i][j]);

    //double r[N][1];
    double **r;
    r = (double **)malloc(N*sizeof(double*));
    for(int i = 0; i < N; i++){
        r[i] = (double*)malloc(sizeof(double));
    }
    for (i = 0; i < N; i++)
    {
        r[i][0] = rand() % 2;
    }

    //ouble br[N][1];
    double **br;
    br = (double **)malloc(N*sizeof(double*));
    for(int i = 0; i < N; i++){
        br[i] = (double*)malloc(sizeof(double));
    }
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < 1; j++)
        {
            for (k = 0; k < N; k++)
            {
                br[i][j] = br[i][j] + b[i][k] * r[k][j];
            }
        }
    }
    //double cr[N][1];
    double **cr;
    cr = (double **)malloc(N*sizeof(double*));
    for(int i = 0; i < N; i++){
        cr[i] = (double*)malloc(sizeof(double));
    }
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < 1; j++)
        {
            for (k = 0; k < N; k++)
            {
                cr[i][j] = cr[i][j] + c[i][k] * r[k][j];
            }
        }
    }
    //double abr[N][1];
    double **abr;
    abr = (double **)malloc(N*sizeof(double*));
    for(int i = 0; i < N; i++){
        abr[i] = (double*)malloc(sizeof(double));
    }
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < 1; j++)
        {
            for (k = 0; k < N; k++)
            {
                abr[i][j] = abr[i][j] + a[i][k] * br[k][j];
            }
        }
    }
    for (i = 0; i < N; i++)
    {
        abr[i][0] -= cr[i][0];
    }
    int flag = 1;
    for (i = 0; i < N; i++)
    {
        if (abr[i][0] <= 0.000005)
            continue;
        else
            flag = 0;
    }
    return flag;
}

