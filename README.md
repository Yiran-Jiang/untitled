## Compile
 mpicc -O2 -fopenmp main.c -o main -lm

## Usage

  #!/bin/bash
  #SBATCH --ntasks=16
  #SBATCH --nodes=16
  #SBATCH --cpus-per-task=16
  #SBATCH --mem-per-cpu=8GB
  #SBATCH --time=0:20:00

  module load gcc/8.3.0
  module load openmpi/4.0.2
  module load pmix

  mpirun -np 16 main
