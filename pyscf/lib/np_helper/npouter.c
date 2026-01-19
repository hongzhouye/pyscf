/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */
#include <stdlib.h>
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "np_helper/np_helper.h"

/*
 * Performs the operation
 *   C[i, j] += A[i] * B[j]
 * where A and B are real vectors and C is a real matrix.
 */
void NPomp_douter(const size_t m, const size_t n,
                  const double *__restrict__ a,
                  const double *__restrict__ b,
                  double *__restrict__ c)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        const double ai = a[i];
        double *ci = c + i * n;

        #pragma omp simd
        for (size_t j = 0; j < n; ++j) {
            ci[j] += ai * b[j];
        }
    }
}

/*
 * Performs the operation
 *   C[i, j] += A[i] * B[j]
 * where A and B are complex vectors and C is a complex matrix.
 */
void NPomp_zouter(const size_t m, const size_t n,
                  const double complex *__restrict__ a,
                  const double complex *__restrict__ b,
                  double complex *__restrict__ c)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        const double complex ai = a[i];
        double complex *ci = c + i * n;

        #pragma omp simd
        for (size_t j = 0; j < n; ++j) {
            ci[j] += ai * b[j];
        }
    }
}
