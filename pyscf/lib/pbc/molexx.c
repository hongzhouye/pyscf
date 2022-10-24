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
#include <math.h>
#include "cint.h"
#include "np_helper/np_helper.h"

#include <stdio.h>

#include <sys/time.h>
double timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}


int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
    env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
    env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
    env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

void precompute_q_cond(int (*intor)(), double *q_cond, CINTOpt *cintopt,
                       int nimgs, double *Ls,
                       int *ao_loc, int *atm, int natm,
                       int *bas, int nbas, double *env, int nenv)
{
    int shls_slice[] = {0, nbas};
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                             atm, natm, bas, nbas, env);
#pragma omp parallel
{
    double qtmp, tmp;
    size_t ij, i, j, di, dj, dij, diji, ish, jsh, jshr, iL, jptrxyz;
    size_t Nbas = nbas / 2;
    int shls[4];
    double *env_loc = malloc(sizeof(double) * nenv);
    NPdcopy(env_loc, env, nenv);
    double *cache = malloc(sizeof(double) * cache_size);
    di = 0;
    for (ish = 0; ish < Nbas; ish++) {
        dj = ao_loc[ish+1] - ao_loc[ish];
        di = MAX(di, dj);
    }
    double *buf = malloc(sizeof(double) * di*di*di*di);
    double *qij;
#pragma omp for schedule(dynamic, 4)
    for (ij = 0; ij < Nbas*Nbas; ij++) {
        ish = ij/Nbas;
        jshr = ij%Nbas;
        jsh = jshr + Nbas;
        di = ao_loc[ish+1] - ao_loc[ish];
        dj = ao_loc[jsh+1] - ao_loc[jsh];
        dij = di * dj;
        diji = dij * di;
        shls[0] = ish;
        shls[1] = jsh;
        shls[2] = ish;
        shls[3] = jsh;
        jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        qij = q_cond + (ish*Nbas+jshr)*nimgs;
        for (iL = 0; iL < nimgs; iL++) {
            shift_bas(env_loc, env, Ls, jptrxyz, iL);
            qtmp = 1e-100;
            if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env_loc,
                              cintopt, cache)) {
                for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                    tmp = buf[i+di*j+dij*i+diji*j];
                    qtmp += tmp*tmp;
                } }
                qtmp = sqrt(sqrt(qtmp));
            }
            qij[iL] = qtmp;
        }
    }
    free(buf);
    free(cache);
}
}

/* pseudo C-code for molexx

    Math:
        K(i 0, j w)
            = \sum_{k,l} \sum_{t1,t2,T} (i 0, l t1 | k (t1+t2), j w+T) P(l 0, k t2)
            = \sum_{k,l} \sum_{t2} P(l 0, k t2) *
                                   [ \sum_{t1,T} (i 0, l t1 | k (t1+t2), j w+T) ]

    Params:
        thr_K

    Precompute:
        ws, wdm
        dm_ts, dtdm
        ovlp_ts, otext, otq, otR

    loop ish
      loop jsh
        loop w \in ws
          con_wdm = max(wdm[w,ish,jsh])
          # BEGIN fill
          # input: ish, jsh, w, dtdm, thr_dtdm, dm_ts, ovlp_ts
          # output: wvk[w,ish,jsh]
          loop lsh
            il_otext = otext[ish,lsh]
            il_otR = otR[ish,lsh]
            il_otq = otq[ish,lsh]
            loop ksh
              kj_otext = otext[ksh,jsh]
              kj_otR = otR[ksh,jsh]
              kj_otq = otq[ksh,jsh]
              loop t2 \in dm_ts
                con_dm = max(dtdm[lsh,ksh,t2]) * con_wdm
                eri = 0.
                loop it1,t1 \in enumerate(ovlp_ts)
                  Ts, Ts_idx = find_Ts(ovlp_ts, t1+t2-w)
                  loop T,iT in zip(Ts,Ts_idx)
                    R = calc_dist(il_otR[it1], kj_otR[iT])
                    ext = il_otext[it1] + kj_otext[iT]
                    denom = max(R - ext, 1.)
                    num = il_otq[it1] * kj_otq[iT] * con_dm
                    if num/denom < thr_K
                      continue
                    eri += calc_eri(ish,lsh,ksh,jsh,0,t1,t1+t2,w+T)
                wvk[w,ish,jsh] += einsum('psrq,sr->pq', eri,dtdm[t2,lsh,ksh])
          # END fill
*/

void row_absmax(double *vec, const double *mat, size_t nrow, size_t ncol)
{
    size_t i,j,ij;
    double tmp;
    for (ij = i = 0; i < nrow; i++) {
        tmp = -1e100;
        for (j = 0; j < ncol; j++, ij++) {
            tmp = MAX(tmp, fabs(mat[ij]));
        }
        vec[i] = tmp;
    }
}

void check_row_int(char *isint, const double *mat, const size_t nrow, const size_t ncol,
                   const double thr)
{
    size_t i,j,ij;
    char isint_row;
    for (ij = i = 0; i < nrow; i++) {
        isint_row = 1;
        for (j = 0; j < ncol; j++, ij++) {
            isint_row &= fabs( round(mat[ij]) - mat[ij] ) < thr;
        }
        isint[i] = isint_row;
    }
}

inline void vec3_add(double *vout, const double *v1, const double *v2, const double a)
/* vo = a*v1+v2
*/
{
    vout[0] = a*v1[0] + v2[0];
    vout[1] = a*v1[1] + v2[1];
    vout[2] = a*v1[2] + v2[2];
}

void mat_rowvec3_add(double *matout, const double *mat, const double *vec,
                     const size_t nrow)
{
    size_t i;
    for (i = 0; i < nrow; i++) {
        matout[i*3+0] = mat[i*3+0] + vec[0];
        matout[i*3+1] = mat[i*3+1] + vec[1];
        matout[i*3+2] = mat[i*3+2] + vec[2];
    }
}

inline void mat_vec3_mul(double *vout, const double *mat, const double *vec)
{
    vout[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
    vout[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
    vout[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline double vec3_dist(const double *v1, const double *v2)
{
    double tmp, d;
    d = 0.;
    tmp = v1[0]-v2[0];
    d += tmp*tmp;
    tmp = v1[1]-v2[1];
    d += tmp*tmp;
    tmp = v1[2]-v2[2];
    d += tmp*tmp;
    d = sqrt(d);
    return d;
}

void contract_eri_dm_shl(double *vk, const double *eri, const double *dm,
                         const int di, const int dl, const int dk, const int dj,
                         const int nao)
/* vk[i,j] += \sum_{k,l} eri[i,l,k,j] dm[l,k]
*/
{
    int i,j,k,l,ilkj;
    double e, d;
    const double *dmk;
    double *vkj;
    for (ilkj = j = 0; j < dj; j++) {
        vkj = vk + j;
        for (k = 0; k < dk; k++) {
            dmk = dm+k;
            for (l = 0; l < dl; l++) {
                d = dmk[l*nao];
                for (i = 0; i < di; i++, ilkj++) {
                    e = eri[ilkj];
                    vkj[i*nao] += e*d;
                }
            }
        }
    }
}

void contract_eri_dm(int (*intor)(), double *ij_out, CINTOpt *cintopt,
                     double *q_cond, double *ext_cond, double *R_cond,
                     double *mdm_cond, double thresh_K,
                     double *buf, double *bvk_b, double *bvk_L,
                     int mdm_nimgs, double *mdm_Ls, double *mdm,
                     int nimgs, double *Ls, double *Lds,
                     int ishr, int jshr,
                     int *ao_loc, int *atm, int natm,
                     int *bas, int nbas, double *env, int nenv,
                     double *env_loc)
{
    const size_t Nbas = nbas/4;
    const size_t Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t lsh0 = Nbas;
    const size_t ksh0 = Nbas*2;
    const size_t jsh0 = Nbas*3;
    const size_t ish = ishr + ish0;
    const size_t jsh = jshr + jsh0;
    int shls_slice[] = {ish0, lsh0, lsh0, ksh0, ksh0, jsh0, jsh0, nbas};
    int shls[] = {ish, 0, 0, jsh};
    const size_t jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    size_t kptrxyz, lptrxyz;
    const size_t di = ao_loc[ish+1] - ao_loc[ish];
    const size_t dj = ao_loc[jsh+1] - ao_loc[jsh];
    const size_t dij = di*dj;
    size_t dk, dl, dilkj;
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t dilkjmax = di*dj*dkmax*dlmax;
    double *eri = buf + dilkjmax;
    double *Tds = eri + dilkjmax;
    double *cache = Tds + nimgs*3;
    double *lk_mdm;

    size_t lsh, lshr, laor, ksh, kshr, kaor, il_shift, kj_shift, mdm_iL, iL, jL, i;
    const double *il_q_cond, *il_ext_cond, *il_R_cond;
    const double *kj_q_cond, *kj_ext_cond, *kj_R_cond;
    const double *lk_mdm_cond;
    const double *cond_R_bra, *cond_R_ket;
    double cond_mdm, R_bra_ket, denom, numer;

    double *mdm_pL, *pL, *pL2;
    double vtmp1[3], vtmp2[3], vtmp3[3];
    char *Lds_mask = malloc(sizeof(char) * nimgs);

    /* debug timing
    */
    struct timeval tick;
    struct timeval tock;
    double timespent[] = {0., 0., 0., 0., 0.};

    for (lshr = 0; lshr < Nbas; lshr++) {
        lsh = lshr + lsh0;
        shls[1] = lsh;
        lptrxyz = atm[PTR_COORD+bas[ATOM_OF+lsh*BAS_SLOTS]*ATM_SLOTS];
        laor = ao_loc[lsh] - ao_loc[lsh0];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        il_shift = (ishr*Nbas+lshr) * nimgs;
        il_q_cond = q_cond + il_shift;
        il_ext_cond = ext_cond + il_shift;
        il_R_cond = R_cond + il_shift*3;
        for (kshr = 0; kshr < Nbas; kshr++) {
            ksh = kshr + ksh0;
            shls[2] = ksh;
            kptrxyz = atm[PTR_COORD+bas[ATOM_OF+ksh*BAS_SLOTS]*ATM_SLOTS];
            kaor = ao_loc[ksh] - ao_loc[ksh0];
            dk = ao_loc[ksh+1] - ao_loc[ksh];
            dilkj = dij*dl*dk;
            kj_shift = (kshr*Nbas+jshr) * nimgs;
            kj_q_cond = q_cond + kj_shift;
            kj_ext_cond = ext_cond + kj_shift;
            kj_R_cond = R_cond + kj_shift*3;
            lk_mdm_cond = mdm_cond + (lshr*Nbas+kshr) * mdm_nimgs;
            for (mdm_iL = 0; mdm_iL < mdm_nimgs; mdm_iL++) {
                mdm_pL = mdm_Ls + mdm_iL*3;
                cond_mdm = lk_mdm_cond[mdm_iL];
                for (i = 0; i < dilkj; i++) {
                    eri[i] = 0.;
                }
                for (iL = 0; iL < nimgs; iL++) {
                    pL = Ls + iL*3;
                    // v1 = mdm_L + L
                    // v2 = mdm_L + L - bvk_L
                    // v3 = v2d
                    gettimeofday(&tick, 0);
                    vec3_add(vtmp1, mdm_pL, pL, 1.);
                    vec3_add(vtmp2, bvk_L, vtmp1, -1.);
                    mat_vec3_mul(vtmp3, bvk_b, vtmp2);
                    mat_rowvec3_add(Tds, Lds, vtmp3, nimgs);
                    check_row_int(Lds_mask, Tds, nimgs, 3, 1e-6);
                    gettimeofday(&tock, 0);
                    timespent[0] += timedifference_msec(tick, tock);
                    // lsh -> L
                    shift_bas(env_loc, env, Ls, lptrxyz, iL);
                    // ksh -> mdm_L + L = v1
                    shift_bas(env_loc, env, vtmp1, kptrxyz, 0);
                    for (jL = 0; jL < nimgs; jL++) {
                        if (!Lds_mask[jL]) {
                            continue;
                        }
                        // qqr cond
                        gettimeofday(&tick, 0);
                        cond_R_bra = il_R_cond + iL*3;
                        cond_R_ket = kj_R_cond + jL*3;
                        R_bra_ket = vec3_dist(cond_R_bra, cond_R_ket);
                        denom = MAX(R_bra_ket-il_ext_cond[iL]-kj_ext_cond[jL], 1.);
                        numer = il_q_cond[iL] * kj_q_cond[jL] * cond_mdm;
                        gettimeofday(&tock, 0);
                        timespent[1] += timedifference_msec(tick, tock);
                        if (numer/denom < thresh_K) {
                            continue;
                        }
                        gettimeofday(&tick, 0);
                        // v3 = mdm_L + L + Lj = v1 + Lj
                        pL2 = Ls + jL*3;
                        vec3_add(vtmp3, vtmp1, pL2, 1.);
                        // jsh -> v3
                        shift_bas(env_loc, env, vtmp3, jptrxyz, 0);
                        // calc eri
                        if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env_loc,
                                          cintopt, cache)) {
                            for (i = 0; i < dilkj; i++) {
                                eri[i] += buf[i];
                            }
                        }
                        gettimeofday(&tock, 0);
                        timespent[2] += timedifference_msec(tick, tock);
                    }
                } // iL

                gettimeofday(&tick, 0);
                lk_mdm = mdm + mdm_iL*Nao*Nao + laor*Nao + kaor;
                contract_eri_dm_shl(ij_out, eri, lk_mdm, di, dl, dk, dj, Nao);
                gettimeofday(&tock, 0);
                timespent[3] += timedifference_msec(tick, tock);
            } // mdm_iL
        } // kshr
    } // lshr

    double tsum = 0.;
    for (i = 0; i < 4; i++) {
        tsum += timespent[i];
    }
    for (i = 0; i < 4; i++) {
        printf("time spent %d: %10.3f sec  %5.1f%%\n",
               i, timespent[i]/1e3, timespent[i]/tsum*100);
    }
}

void molexx_drv(int (*intor)(), void (*contract)(), double *out, CINTOpt *cintopt,
                int mdm_nimgs, double *mdm_Ls, double *mdm,
                int bvk_nimgs, double *bvk_Ls, double *bvk_b,
                int nimgs, double *Ls, double *Lds,
                double *q_cond, double *ext_cond, double *R_cond,
                double *mdm_cond, double thresh_K,
                int *ao_loc, int *atm, int natm,
                int *bas, int nbas, double *env, int nenv)
{
    const int Nbas = nbas/4;
    const int Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t jsh0 = Nbas*3;
    int shls_slice[] = {0, Nbas, Nbas, Nbas*2, Nbas*2, Nbas*3, Nbas*3, Nbas*4};
    const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                atm, natm, bas, nbas, env);
    const size_t dimax = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t djmax = GTOmax_shell_dim(ao_loc, shls_slice+6, 1);

#pragma omp parallel
{
    size_t ij, ish, ishr, jsh, jshr, ijao, bvk_iL;
    // buf = [eribuf1, eribuf2, Lsbuf, cache]
    double *buf = malloc(sizeof(double) * (dimax*dlmax*dkmax*djmax*2+nimgs*3+cache_size));
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);
    double *bvk_L, *bvk_ij_out;
#pragma omp for schedule(dynamic)
    for (ij = 0; ij < Nbas*Nbas; ij++) {
        ishr = ij/Nbas;
        jshr = ij%Nbas;
        ish = ishr + ish0;
        jsh = jshr + jsh0;
        ijao = (ao_loc[ish]-ao_loc[ish0])*Nao + ao_loc[jsh]-ao_loc[jsh0];
        for (bvk_iL = 0; bvk_iL < bvk_nimgs; bvk_iL++) {
            bvk_L = bvk_Ls + bvk_iL*3;
            bvk_ij_out = out + bvk_iL*Nao*Nao + ijao;
            (*contract)(intor, bvk_ij_out, cintopt,
                        q_cond, ext_cond, R_cond, mdm_cond, thresh_K,
                        buf, bvk_b, bvk_L,
                        mdm_nimgs, mdm_Ls, mdm,
                        nimgs, Ls, Lds,
                        ishr, jshr,
                        ao_loc, atm, natm, bas, nbas,
                        env, nenv, env_loc);
        } // bvk_iL
    } // ij
    free(buf);
    free(env_loc);
}
}

/* Version 2: Don't compute commensurate Ts; rely on integral screening
*/

void contract_eri_dm_2(int (*intor)(), double *ij_out, CINTOpt *cintopt,
                       double *q_cond, double *ext_cond, double *R_cond,
                       double *mdm_cond, double thresh_K,
                       double *buf,
                       int mdm_nimgs, double *mdm_Ls, double *mdm,
                       int nimgs, double *Ls,
                       int bvk_nimgs, int *bvk_cell_loc, int *bbvk_loc,
                       int *bvk_mdm_loc, int *mdm_cell_loc,
                       int ishr, int jshr,
                       int *ao_loc, int *atm, int natm,
                       int *bas, int nbas, double *env, int nenv,
                       double *env_loc)
{
    const size_t Nbas = nbas/4;
    const size_t Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t lsh0 = Nbas;
    const size_t ksh0 = Nbas*2;
    const size_t jsh0 = Nbas*3;
    const size_t ish = ishr + ish0;
    const size_t jsh = jshr + jsh0;
    int shls_slice[] = {ish0, lsh0, lsh0, ksh0, ksh0, jsh0, jsh0, nbas};
    int shls[] = {ish, 0, 0, jsh};
    const size_t jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    size_t kptrxyz, lptrxyz;
    const size_t di = ao_loc[ish+1] - ao_loc[ish];
    const size_t dj = ao_loc[jsh+1] - ao_loc[jsh];
    const size_t dij = di*dj;
    size_t dk, dl, dilkj;
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t dilkjmax = di*dj*dkmax*dlmax;
    double *eri = buf + dilkjmax;
    double *cache = eri + dilkjmax;
    double *lk_mdm, *bvk_ij_out;

    size_t lsh, lshr, laor, ksh, kshr, kaor, il_shift, kj_shift, i;
    size_t mdm_iL, iL, jL, bvk_iL, bvk_jL, bvk_kL, idx_iL, bvk_iL_mdm_shift;
    const double *il_q_cond, *il_ext_cond, *il_R_cond;
    const double *kj_q_cond, *kj_ext_cond, *kj_R_cond;
    const double *lk_mdm_cond;
    const double *cond_R_bra, *cond_R_ket;
    double cond_mdm, R_bra_ket, denom, numer;

    double *mdm_pL, *pL, *pL2;
    double vtmp1[3], vtmp2[3], vtmp3[3];

    for (lshr = 0; lshr < Nbas; lshr++) {
        lsh = lshr + lsh0;
        shls[1] = lsh;
        lptrxyz = atm[PTR_COORD+bas[ATOM_OF+lsh*BAS_SLOTS]*ATM_SLOTS];
        laor = ao_loc[lsh] - ao_loc[lsh0];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        il_shift = (ishr*Nbas+lshr) * nimgs;
        il_q_cond = q_cond + il_shift;
        il_ext_cond = ext_cond + il_shift;
        il_R_cond = R_cond + il_shift*3;
        for (kshr = 0; kshr < Nbas; kshr++) {
            ksh = kshr + ksh0;
            shls[2] = ksh;
            kptrxyz = atm[PTR_COORD+bas[ATOM_OF+ksh*BAS_SLOTS]*ATM_SLOTS];
            kaor = ao_loc[ksh] - ao_loc[ksh0];
            dk = ao_loc[ksh+1] - ao_loc[ksh];
            dilkj = dij*dl*dk;
            kj_shift = (kshr*Nbas+jshr) * nimgs;
            kj_q_cond = q_cond + kj_shift;
            kj_ext_cond = ext_cond + kj_shift;
            kj_R_cond = R_cond + kj_shift*3;
            lk_mdm_cond = mdm_cond + (lshr*Nbas+kshr) * mdm_nimgs;
            for (bvk_jL = 0; bvk_jL < bvk_nimgs; bvk_jL++) {
                bvk_ij_out = ij_out + bvk_jL*Nao*Nao;
                for (bvk_iL = 0; bvk_iL < bvk_nimgs; bvk_iL++) {
                    bvk_kL = bbvk_loc[bvk_jL*bvk_nimgs+bvk_iL];
                    bvk_iL_mdm_shift = bvk_iL * (mdm_nimgs+1);
                    for (mdm_iL = 0; mdm_iL < mdm_nimgs; mdm_iL++) {
                        mdm_pL = mdm_Ls + mdm_iL*3;
                        cond_mdm = lk_mdm_cond[mdm_iL];
                        for (i = 0; i < dilkj; i++) {
                            eri[i] = 0.;
                        }
                        for (idx_iL = bvk_mdm_loc[bvk_iL_mdm_shift+mdm_iL];
                             idx_iL < bvk_mdm_loc[bvk_iL_mdm_shift+mdm_iL+1]; idx_iL++) {
                            iL = mdm_cell_loc[idx_iL];
                            pL = Ls + iL*3;
                            // lsh -> L
                            shift_bas(env_loc, env, Ls, lptrxyz, iL);
                            // v1 = mdm_L + Li
                            vec3_add(vtmp1, mdm_pL, pL, 1.);
                            shift_bas(env_loc, env, vtmp1, kptrxyz, 0);
                            for (jL = bvk_cell_loc[bvk_kL];
                                 jL < bvk_cell_loc[bvk_kL+1]; jL++) {
                                // qqr cond
                                cond_R_bra = il_R_cond + iL*3;
                                cond_R_ket = kj_R_cond + jL*3;
                                R_bra_ket = vec3_dist(cond_R_bra, cond_R_ket);
                                denom = MAX(R_bra_ket-il_ext_cond[iL]-kj_ext_cond[jL],
                                            1.);
                                numer = il_q_cond[iL] * kj_q_cond[jL] * cond_mdm;
                                if (numer/denom < thresh_K) {
                                    continue;
                                }
                                pL2 = Ls + jL*3;
                                // v2 = mdm_L + Li + Lj  =  bvk_Lj + T
                                vec3_add(vtmp2, vtmp1, pL2, 1.);
                                shift_bas(env_loc, env, vtmp2, jptrxyz, 0);
                                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env_loc,
                                                  cintopt, cache)) {
                                    for (i = 0; i < dilkj; i++) {
                                        eri[i] += buf[i];
                                    }
                                }
                            } // jL
                        } // idx_iL

                        lk_mdm = mdm + mdm_iL*Nao*Nao + laor*Nao + kaor;
                        contract_eri_dm_shl(bvk_ij_out, eri, lk_mdm, di, dl, dk, dj, Nao);

                    } // mdm_iL
                } // bvk_iL
            } // bvk_jL
        } // kshr
    } // lshr
}

void molexx_drv_2(int (*intor)(), void (*contract)(), double *out, CINTOpt *cintopt,
                  int mdm_nimgs, double *mdm_Ls, double *mdm,
                  int nimgs, double *Ls,
                  int bvk_nimgs, int *bvk_cell_loc, int *bbvk_loc,
                  int *bvk_mdm_loc, int *mdm_cell_loc,
                  double *q_cond, double *ext_cond, double *R_cond,
                  double *mdm_cond, double thresh_K,
                  int *ao_loc, int *atm, int natm,
                  int *bas, int nbas, double *env, int nenv)
{
    const int Nbas = nbas/4;
    const int Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t jsh0 = Nbas*3;
    int shls_slice[] = {0, Nbas, Nbas, Nbas*2, Nbas*2, Nbas*3, Nbas*3, Nbas*4};
    const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                atm, natm, bas, nbas, env);
    const size_t dimax = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t djmax = GTOmax_shell_dim(ao_loc, shls_slice+6, 1);

#pragma omp parallel
{
    size_t ij, ish, ishr, jsh, jshr, ijao;
    // buf = [eribuf1, eribuf2, Lsbuf, cache]
    double *buf = malloc(sizeof(double) * (dimax*dlmax*dkmax*djmax*2+cache_size));
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);
    double *ij_out;
#pragma omp for schedule(dynamic)
    for (ij = 0; ij < Nbas*Nbas; ij++) {
        ishr = ij/Nbas;
        jshr = ij%Nbas;
        ish = ishr + ish0;
        jsh = jshr + jsh0;
        ijao = (ao_loc[ish]-ao_loc[ish0])*Nao + ao_loc[jsh]-ao_loc[jsh0];
        ij_out = out + ijao;
        (*contract)(intor, ij_out, cintopt,
                    q_cond, ext_cond, R_cond, mdm_cond, thresh_K,
                    buf,
                    mdm_nimgs, mdm_Ls, mdm,
                    nimgs, Ls,
                    bvk_nimgs, bvk_cell_loc, bbvk_loc,
                    bvk_mdm_loc, mdm_cell_loc,
                    ishr, jshr,
                    ao_loc, atm, natm, bas, nbas,
                    env, nenv, env_loc);
    } // ij
    free(buf);
    free(env_loc);
}
}

/* Version 3: Don't compute commensurate Ts; rely on integral screening
*/

void contract_eri_dm_3(int (*intor)(), double *ij_out, CINTOpt *cintopt,
                       double *q_cond, double *ext_cond, double *R_cond,
                       double *mdm_cond, double thresh_K,
                       double *buf,
                       int mdm_nimgs, double *mdm_Ls, double *mdm,
                       int nimgs, double *Ls,
                       int bvk_nimgs, int *bvk_cell_loc, int *bbvk_loc,
                       int *bvkidx_by_mdmcell,
                       int ishr, int jshr,
                       int *ao_loc, int *atm, int natm,
                       int *bas, int nbas, double *env, int nenv,
                       double *env_loc)
{
    const size_t Nbas = nbas/4;
    const size_t Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t lsh0 = Nbas;
    const size_t ksh0 = Nbas*2;
    const size_t jsh0 = Nbas*3;
    const size_t ish = ishr + ish0;
    const size_t jsh = jshr + jsh0;
    int shls_slice[] = {ish0, lsh0, lsh0, ksh0, ksh0, jsh0, jsh0, nbas};
    int shls[] = {ish, 0, 0, jsh};
    const size_t jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    size_t kptrxyz, lptrxyz;
    const size_t di = ao_loc[ish+1] - ao_loc[ish];
    const size_t dj = ao_loc[jsh+1] - ao_loc[jsh];
    const size_t dij = di*dj;
    size_t dk, dl, dilkj;
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t dilkjmax = di*dj*dkmax*dlmax;
    double *eri = buf + dilkjmax;
    double *cache = eri + dilkjmax;
    double *lk_mdm, *bvk_ij_out;

    size_t lsh, lshr, laor, ksh, kshr, kaor, il_shift, kj_shift, i;
    size_t mdm_iL, iL, jL, bvk_iL, bvk_jL, bvk_kL;
    const double *il_q_cond, *il_ext_cond, *il_R_cond;
    const double *kj_q_cond, *kj_ext_cond, *kj_R_cond;
    const double *lk_mdm_cond;
    const double *cond_R_bra, *cond_R_ket;
    double cond_mdm, il_q_dm_cond, R_bra_ket, denom, numer;

    double *mdm_pL, *pL, *pL2;
    double vtmp1[3], vtmp2[3];

    for (lshr = 0; lshr < Nbas; lshr++) {
        lsh = lshr + lsh0;
        shls[1] = lsh;
        lptrxyz = atm[PTR_COORD+bas[ATOM_OF+lsh*BAS_SLOTS]*ATM_SLOTS];
        laor = ao_loc[lsh] - ao_loc[lsh0];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        il_shift = (ishr*Nbas+lshr) * nimgs;
        il_q_cond = q_cond + il_shift;
        il_ext_cond = ext_cond + il_shift;
        il_R_cond = R_cond + il_shift*3;
        for (kshr = 0; kshr < Nbas; kshr++) {
            ksh = kshr + ksh0;
            shls[2] = ksh;
            kptrxyz = atm[PTR_COORD+bas[ATOM_OF+ksh*BAS_SLOTS]*ATM_SLOTS];
            kaor = ao_loc[ksh] - ao_loc[ksh0];
            dk = ao_loc[ksh+1] - ao_loc[ksh];
            dilkj = dij*dl*dk;
            kj_shift = (kshr*Nbas+jshr) * nimgs;
            kj_q_cond = q_cond + kj_shift;
            kj_ext_cond = ext_cond + kj_shift;
            kj_R_cond = R_cond + kj_shift*3;
            lk_mdm_cond = mdm_cond + (lshr*Nbas+kshr) * mdm_nimgs;
            for (bvk_jL = 0; bvk_jL < bvk_nimgs; bvk_jL++) {
                bvk_ij_out = ij_out + bvk_jL*Nao*Nao;
                for (mdm_iL = 0; mdm_iL < mdm_nimgs; mdm_iL++) {
                    mdm_pL = mdm_Ls + mdm_iL*3;
                    // mdm cond
                    cond_mdm = lk_mdm_cond[mdm_iL];
                    if (cond_mdm < thresh_K) {
                        continue;
                    }
                    for (i = 0; i < dilkj; i++) {
                        eri[i] = 0.;
                    }
                    for (iL = 0; iL < nimgs; iL++) {
                        // q_bra cond
                        if (il_q_cond[iL] < thresh_K) {
                            continue;
                        }
                        cond_R_bra = il_R_cond + iL*3;
                        il_q_dm_cond = il_q_cond[iL] * cond_mdm;
                        bvk_iL = bvkidx_by_mdmcell[mdm_iL*nimgs+iL];
                        bvk_kL = bbvk_loc[bvk_jL*bvk_nimgs+bvk_iL];
                        pL = Ls + iL*3;
                        // lsh -> L
                        shift_bas(env_loc, env, Ls, lptrxyz, iL);
                        // v1 = mdm_L + Li
                        vec3_add(vtmp1, mdm_pL, pL, 1.);
                        shift_bas(env_loc, env, vtmp1, kptrxyz, 0);
                        for (jL = bvk_cell_loc[bvk_kL];
                             jL < bvk_cell_loc[bvk_kL+1]; jL++) {
                            // q_ket cond
                            if (kj_q_cond[jL] < thresh_K) {
                                continue;
                            }
                            // qqr cond
                            cond_R_ket = kj_R_cond + jL*3;
                            R_bra_ket = vec3_dist(cond_R_bra, cond_R_ket);
                            denom = MAX(R_bra_ket-il_ext_cond[iL]-kj_ext_cond[jL],
                                        1.);
                            numer = il_q_dm_cond * kj_q_cond[jL];
                            if (numer/denom < thresh_K) {
                                continue;
                            }
                            pL2 = Ls + jL*3;
                            // v2 = mdm_L + Li + Lj  =  bvk_Lj + T
                            vec3_add(vtmp2, vtmp1, pL2, 1.);
                            shift_bas(env_loc, env, vtmp2, jptrxyz, 0);
                            if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                              env_loc, cintopt, cache)) {
                                for (i = 0; i < dilkj; i++) {
                                    eri[i] += buf[i];
                                }
                            }
                        } // jL
                    } // iL

                    lk_mdm = mdm + mdm_iL*Nao*Nao + laor*Nao + kaor;
                    contract_eri_dm_shl(bvk_ij_out, eri, lk_mdm, di, dl, dk, dj, Nao);

                } // mdm_iL
            } // bvk_jL
        } // kshr
    } // lshr
}

void molexx_drv_3(int (*intor)(), void (*contract)(), double *out, CINTOpt *cintopt,
                  int mdm_nimgs, double *mdm_Ls, double *mdm,
                  int nimgs, double *Ls,
                  int bvk_nimgs, int *bvk_cell_loc, int *bbvk_loc,
                  int *bvkidx_by_mdmcell,
                  double *q_cond, double *ext_cond, double *R_cond,
                  double *mdm_cond, double thresh_K,
                  int *ao_loc, int *atm, int natm,
                  int *bas, int nbas, double *env, int nenv)
{
    const int Nbas = nbas/4;
    const int Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t jsh0 = Nbas*3;
    int shls_slice[] = {0, Nbas, Nbas, Nbas*2, Nbas*2, Nbas*3, Nbas*3, Nbas*4};
    const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                atm, natm, bas, nbas, env);
    const size_t dimax = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t djmax = GTOmax_shell_dim(ao_loc, shls_slice+6, 1);

#pragma omp parallel
{
    size_t ij, ish, ishr, jsh, jshr, ijao;
    // buf = [eribuf1, eribuf2, Lsbuf, cache]
    double *buf = malloc(sizeof(double) * (dimax*dlmax*dkmax*djmax*2+cache_size));
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);
    double *ij_out;
#pragma omp for schedule(dynamic)
    for (ij = 0; ij < Nbas*Nbas; ij++) {
        ishr = ij/Nbas;
        jshr = ij%Nbas;
        ish = ishr + ish0;
        jsh = jshr + jsh0;
        ijao = (ao_loc[ish]-ao_loc[ish0])*Nao + ao_loc[jsh]-ao_loc[jsh0];
        ij_out = out + ijao;
        (*contract)(intor, ij_out, cintopt,
                    q_cond, ext_cond, R_cond, mdm_cond, thresh_K,
                    buf,
                    mdm_nimgs, mdm_Ls, mdm,
                    nimgs, Ls,
                    bvk_nimgs, bvk_cell_loc, bbvk_loc,
                    bvkidx_by_mdmcell,
                    ishr, jshr,
                    ao_loc, atm, natm, bas, nbas,
                    env, nenv, env_loc);
    } // ij
    free(buf);
    free(env_loc);
}
}
