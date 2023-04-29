#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "monte_carlo.h"
#include "util.h"

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process)
{
  int count = 0;
  int num_points_per_proc = num_points / mpi_world_size;

  // scatter the points to all processes
  double *x_buf = (double *)malloc(num_points_per_proc * sizeof(double));
  double *y_buf = (double *)malloc(num_points_per_proc * sizeof(double));
  MPI_Scatter(xs, num_points_per_proc, MPI_DOUBLE, x_buf, num_points_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(ys, num_points_per_proc, MPI_DOUBLE, y_buf, num_points_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// set up OpenMP parallel region
#pragma omp parallel num_threads(threads_per_process)
  {
    int i;
    int local_count = 0;
    double x, y;

// loop over the points assigned to this thread
#pragma omp for schedule(static)
    for (i = 0; i < num_points_per_proc; i++)
    {
      x = x_buf[i];
      y = y_buf[i];

      if (x * x + y * y <= 1)
        local_count++;
    }

// reduce the counts from all threads in this process
#pragma omp atomic
    count += local_count;
  }

  // gather the counts from all processes and compute the total count
  int *counts = NULL;
  if (mpi_rank == 0)
  {
    counts = (int *)malloc(mpi_world_size * sizeof(int));
  }
  MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // compute the total count and return the estimated PI value
  if (mpi_rank == 0)
  {
    int total_count = 0;
    for (int i = 0; i < mpi_world_size; i++)
    {
      total_count += counts[i];
    }
    free(counts);
    free(x_buf);
    free(y_buf);
    return (double)4 * total_count / num_points;
  }
  else
  {
    free(x_buf);
    free(y_buf);
    return 0.0;
  }
}
