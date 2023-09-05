/* Copyright 2014-2021 The PySCF Developers. All Rights Reserved.

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
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "mp/mp2.h"


/*  Calculate KRMP2 energy with density fitting for a given k-point quad (ki,kj,ka,kb) where ka!=kb
    and a given AO range (i0,i0+nocci,j0,j0+noccj)

    Math:
        oovv(ki,kj,ka,kb) = einsum('iaL,jbL->ijab', ovL(ki,ka), ovL(kj,kb))
        oovv(ki,kj,kb,ka) = einsum('ibL,jaL->ijba', ovL(ki,kb), ovL(kj,ka))
        denom(ki,kj,ka,kb) = ei(ki)+ej(kj)-ea(ka)-eb(kb)
        t2(ki,kj,ka,kb) = conj(oovv(ki,kj,ka,kb)) / denom(ki,kj,ka,kb)
        t2(ki,kj,kb,ka) = conj(oovv(ki,kj,kb,ka)) / denom(ki,kj,ka,kb)
        ed_out += einsum('ijab,ijab->', t2(ki,kj,ka,kb), oovv(ki,kj,ka,kb))
        ed_out += einsum('ijba,ijba->', t2(ki,kj,kb,ka), oovv(ki,kj,kb,ka))
        ex_out -= einsum('ijab,ijba->', t2(ki,kj,ka,kb), oovv(ki,kj,kb,ka)) * 2
*/
void KMP2_contract_kaneqkb(double *ed_out, double *ex_out,
                           const double *batch_iaLR, const double *batch_iaLI,
                           const double *batch_ibLR, const double *batch_ibLI,
                           const double *batch_jbLR, const double *batch_jbLI,
                           const double *batch_jaLR, const double *batch_jaLI,
                           const int i0, const int j0, const int nocci, const int noccj,
                           const int nvir, const int naux,
                           const double *moeoo, const double *moevv)
{
    const int I1 = 1;
    const double D0 = 0;
    const double D1 = 1;
    const double Dm1 = -1;
    const char TRANS_Y = 'T';
    const char TRANS_N = 'N';

    const int nvv = nvir*nvir;
    const int nvx = nvir*naux;

    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci*noccj);
    size_t njob = _MP2_gen_jobs(jobs, 0, i0, j0, nocci, noccj);

    const double **parr_iaLR = _gen_ptr_arr(batch_iaLR, nocci, nvx);
    const double **parr_iaLI = _gen_ptr_arr(batch_iaLI, nocci, nvx);
    const double **parr_ibLR = _gen_ptr_arr(batch_ibLR, nocci, nvx);
    const double **parr_ibLI = _gen_ptr_arr(batch_ibLI, nocci, nvx);
    const double **parr_jbLR = _gen_ptr_arr(batch_jbLR, noccj, nvx);
    const double **parr_jbLI = _gen_ptr_arr(batch_jbLI, noccj, nvx);
    const double **parr_jaLR = _gen_ptr_arr(batch_jaLR, noccj, nvx);
    const double **parr_jaLI = _gen_ptr_arr(batch_jaLI, noccj, nvx);

#pragma omp parallel default(none) \
        shared(njob, jobs, batch_iaLR, batch_iaLI, batch_ibLR, batch_ibLI, batch_jbLR, batch_jbLI, batch_jaLR, batch_jaLI, parr_iaLR, parr_iaLI, parr_ibLR, parr_ibLI, parr_jbLR, parr_jbLI, parr_jaLR, parr_jaLI, moeoo, moevv, naux, nvir, nvv, noccj, D0, D1, Dm1, I1, TRANS_N, TRANS_Y, ed_out, ex_out)
{
    double *cache = malloc(sizeof(double) * nvv*6);
    double *vabR = cache;
    double *vabI = vabR + nvv;
    double *vabTR = vabI + nvv;
    double *vabTI = vabTR + nvv;
    double *tabR = vabTI + nvv;
    double *tabI = tabR + nvv;
    double eij;

    const double *iaLR, *iaLI, *ibLR, *ibLI, *jbLR, *jbLI, *jaLR, *jaLI;
    size_t i,j,a,m;
    double ed=0, ex=0, fac;

#pragma omp for schedule (dynamic, 4)

    for (m = 0; m < njob; ++m) {
        i = jobs[m].i;
        j = jobs[m].j;
        fac = jobs[m].fac;

        iaLR = parr_iaLR[i]; iaLI = parr_iaLI[i];
        ibLR = parr_ibLR[i]; ibLI = parr_ibLI[i];
        jbLR = parr_jbLR[j]; jbLI = parr_jbLI[j];
        jaLR = parr_jaLR[j]; jaLI = parr_jaLI[j];
        eij = moeoo[i*noccj+j];

        // einsum([i]aL,[j]bL) -> [i][j]ab
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbLR, &naux, iaLR, &naux,
               &D0, vabR, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &Dm1, jbLI, &naux, iaLI, &naux,
               &D1, vabR, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbLR, &naux, iaLI, &naux,
               &D0, vabI, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbLI, &naux, iaLR, &naux,
               &D1, vabI, &nvir);
        // einsum([j]aL,[i]bL) -> [i][j]ab
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, ibLR, &naux, jaLR, &naux,
               &D0, vabTR, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &Dm1, ibLI, &naux, jaLI, &naux,
               &D1, vabTR, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, ibLR, &naux, jaLI, &naux,
               &D0, vabTI, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, ibLI, &naux, jaLR, &naux,
               &D1, vabTI, &nvir);
        // tab = vab / eijab
        for (a = 0; a < nvv; ++a) {
            tabR[a] =  vabR[a] / (eij - moevv[a]);
            tabI[a] = -vabI[a] / (eij - moevv[a]);
        }
        // vab, tab -> ed
        ed += ddot_(&nvv, vabR, &I1, tabR, &I1) * fac;
        ed -= ddot_(&nvv, vabI, &I1, tabI, &I1) * fac;
        // vab_ex, tab -> ex
        ex -= ddot_(&nvv, vabTR, &I1, tabR, &I1) * fac;
        ex += ddot_(&nvv, vabTI, &I1, tabI, &I1) * fac;

        // tab = vab / eijab
        for (a = 0; a < nvv; ++a) {
            tabR[a] =  vabTR[a] / (eij - moevv[a]);
            tabI[a] = -vabTI[a] / (eij - moevv[a]);
        }
        // vab, tab -> ed
        ed += ddot_(&nvv, vabTR, &I1, tabR, &I1) * fac;
        ed -= ddot_(&nvv, vabTI, &I1, tabI, &I1) * fac;
    }
    free(cache);

#pragma omp critical
{
    *ed_out += ed;
    *ex_out += ex * 2;
}

} // parallel

    free(jobs);
    free(parr_iaLR); free(parr_iaLI);
    free(parr_ibLR); free(parr_ibLI);
    free(parr_jbLR); free(parr_jbLI);
    free(parr_jaLR); free(parr_jaLI);
}


/*  Driver for KRMP2
*/
void KMP2_contract_drv(double *ed_out, double *ex_out,
                       const double *batch_iaLR, const double *batch_iaLI,
                       const double *batch_ibLR, const double *batch_ibLI,
                       const double *batch_jbLR, const double *batch_jbLI,
                       const double *batch_jaLR, const double *batch_jaLI,
                       const int ki, const int kj, const int ka, const int kb,
                       const int i0, const int j0,
                       const int nocci, const int noccj,
                       const int nvir, const int naux,
                       const double *moeoo, const double *moevv)
{
    if (ka == kb) {
        const int s2symm = (ki==kj)?(1):(0);
        MP2_contract_c(
            ed_out, ex_out, s2symm,
            batch_iaLR, batch_iaLI, batch_jbLR, batch_jbLI,
            i0, j0, nocci, noccj, nvir, naux, moeoo, moevv
        );
    } else {
        KMP2_contract_kaneqkb(
            ed_out, ex_out,
            batch_iaLR, batch_iaLI, batch_ibLR, batch_ibLI,
            batch_jbLR, batch_jbLI, batch_jaLR, batch_jaLI,
            i0, j0, nocci, noccj, nvir, naux, moeoo, moevv
        );
    }
}
