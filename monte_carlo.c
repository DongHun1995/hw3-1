#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "monte_carlo.h"
#include "util.h"

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process)
{
  int count = 0;
  // 프로세스로 num_point 쪼갬
  int num_points_per_process = num_points / mpi_world_size;

  // 쪼갠 프로세스별 num_point 메모리 할당
  double *x_buf = (double *)malloc(num_points_per_process * sizeof(double));
  double *y_buf = (double *)malloc(num_points_per_process * sizeof(double));

  // 각 프로세스별로 xs, ys scatter
  MPI_Scatter(xs, num_points_per_process, MPI_DOUBLE, x_buf, num_points_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(ys, num_points_per_process, MPI_DOUBLE, y_buf, num_points_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// openMP 적용
#pragma omp parallel num_threads(threads_per_process)
  {
    int i;
    int local_count = 0;
    double x, y;
// 반복문 병렬 처리
#pragma omp for schedule(static)
    for (i = 0; i < num_points_per_process; i++)
    {
      x = x_buf[i];
      y = y_buf[i];

      if (x * x + y * y <= 1)
        local_count++;
    }

// count 합치기
#pragma omp atomic
    count += local_count;
  }

  // scatter해서 count 계산한거 gather 하기
  int *counts = NULL;
  if (mpi_rank == 0)
  {
    counts = (int *)malloc(mpi_world_size * sizeof(int));
  }
  MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // 전체 확률 계산후 memory free
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
