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
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include "cint.h"
#include "np_helper/np_helper.h"

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

enum {
    TDM_DM_TEST,
    TDM_DM_SKIP,
    TDM_BRA_TEST,
    TDM_BRA_SKIP,
    TDM_KET_TEST,
    TDM_KET_SKIP,
    TDM_QQR_TEST,
    TDM_QQR_SKIP,
    TDM_INTOR_CALL,
    TDM_INTOR_NONZERO,
    TDM_CONTRACT_CALL,
    TDM_NCOUNTS
};

enum {
    TDM_TOTAL_TIME,
    TDM_INTOR_TIME,
    TDM_CONTRACT_TIME,
    TDM_NTIMES
};

#define TDM_COUNT(profile, index) \
    do { if (profile != NULL) { profile[index]++; } } while (0)

static inline double wall_time(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
    env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
    env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
    env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

void PBCtdm_q_cond(int (*intor)(), double *q_cond, CINTOpt *cintopt,
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

/* Pseudo C-code for the TDM K build

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

static inline void vec3_add(double *vout, const double *v1, const double *v2,
                            const double a)
/* vo = a*v1+v2
*/
{
    vout[0] = a*v1[0] + v2[0];
    vout[1] = a*v1[1] + v2[1];
    vout[2] = a*v1[2] + v2[2];
}

static inline double vec3_dist(const double *v1, const double *v2)
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

static void contract_eri_dm_shl(double *vk, const double *eri, const double *dm,
                                const int di, const int dl, const int dk,
                                const int dj, const int nao)
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

void PBCtdm_contract_eri_dm(int (*intor)(), double *ij_out,
                            CINTOpt *cintopt, double *q_cond,
                            double *ext_cond, double *R_cond,
                            double *dm_cond, double thresh_K,
                            double *buf,
                            int dm_nimgs, double *dm_Ls, double *dm,
                            int nimgs, double *Ls,
                            int bvk_nimgs, int *bvk_cell_loc,
                            int *bvkadd_loc,
                            int *bvkidx_by_dmcell,
                            int ishr, int jshr,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env, int nenv,
                            double *env_loc, uint64_t *profile_counts,
                            double *profile_times)
{
    const size_t Nbas = nbas/4;
    const size_t Nao = ao_loc[Nbas] - ao_loc[0];
    const size_t ish0 = 0;
    const size_t lsh0 = Nbas;
    const size_t ksh0 = Nbas*2;
    const size_t jsh0 = Nbas*3;
    const size_t ish = ishr + ish0;
    const size_t jsh = jshr + jsh0;
    int shls_slice[] = {ish0, lsh0, lsh0, ksh0,
                        ksh0, jsh0, jsh0, nbas};
    int shls[] = {ish, 0, 0, jsh};
    const size_t jptrxyz =
        atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    size_t kptrxyz, lptrxyz;
    const size_t di = ao_loc[ish+1] - ao_loc[ish];
    const size_t dj = ao_loc[jsh+1] - ao_loc[jsh];
    const size_t dij = di*dj;
    size_t dk, dl, dilkj;
    const size_t dlmax = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    const size_t dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    const size_t dilkjmax = di*dj*dkmax*dlmax;
    double *eri_bvk = malloc(sizeof(double) * dilkjmax*bvk_nimgs);
    double *cache = buf + dilkjmax;
    double *eri, *lk_dm, *bvk_ij_out;

    size_t lsh, lshr, laor, ksh, kshr, kaor, il_shift, kj_shift, i;
    size_t dm_iL, iL, jL, bvk_iL, bvk_jL, bvk_kL;
    const double *il_q_cond, *il_ext_cond, *il_R_cond;
    const double *kj_q_cond, *kj_ext_cond, *kj_R_cond;
    const double *lk_dm_cond;
    const double *cond_R_bra, *cond_R_ket;
    double cond_dm, il_q_dm_cond, R_bra_ket, denom, numer;

    double *dm_pL, *pL, *pL2;
    double vtmp1[3], vtmp2[3];
    double tick;
    int intor_nonzero;

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
            lk_dm_cond = dm_cond + (lshr*Nbas+kshr) * dm_nimgs;
            for (dm_iL = 0; dm_iL < dm_nimgs; dm_iL++) {
                dm_pL = dm_Ls + dm_iL*3;
                cond_dm = lk_dm_cond[dm_iL];
                TDM_COUNT(profile_counts, TDM_DM_TEST);
                if (cond_dm < thresh_K) {
                    TDM_COUNT(profile_counts, TDM_DM_SKIP);
                    continue;
                }
                for (bvk_jL = 0; bvk_jL < bvk_nimgs; bvk_jL++) {
                    eri = eri_bvk + bvk_jL*dilkjmax;
                    for (i = 0; i < dilkj; i++) {
                        eri[i] = 0.;
                    }
                }
                for (iL = 0; iL < nimgs; iL++) {
                    TDM_COUNT(profile_counts, TDM_BRA_TEST);
                    if (il_q_cond[iL] < thresh_K) {
                        TDM_COUNT(profile_counts, TDM_BRA_SKIP);
                        continue;
                    }
                    cond_R_bra = il_R_cond + iL*3;
                    il_q_dm_cond = il_q_cond[iL] * cond_dm;
                    bvk_iL = bvkidx_by_dmcell[dm_iL*nimgs+iL];
                    pL = Ls + iL*3;
                    shift_bas(env_loc, env, Ls, lptrxyz, iL);
                    vec3_add(vtmp1, dm_pL, pL, 1.);
                    shift_bas(env_loc, env, vtmp1, kptrxyz, 0);
                    for (bvk_kL = 0; bvk_kL < bvk_nimgs; bvk_kL++) {
                        bvk_jL =
                            bvkadd_loc[bvk_kL*bvk_nimgs+bvk_iL];
                        eri = eri_bvk + bvk_jL*dilkjmax;
                        for (jL = bvk_cell_loc[bvk_kL];
                             jL < bvk_cell_loc[bvk_kL+1]; jL++) {
                            TDM_COUNT(profile_counts, TDM_KET_TEST);
                            if (kj_q_cond[jL] < thresh_K) {
                                TDM_COUNT(profile_counts, TDM_KET_SKIP);
                                continue;
                            }
                            TDM_COUNT(profile_counts, TDM_QQR_TEST);
                            cond_R_ket = kj_R_cond + jL*3;
                            vec3_add(vtmp2, vtmp1, cond_R_ket, 1.);
                            R_bra_ket = vec3_dist(cond_R_bra, vtmp2);
                            denom = MAX(R_bra_ket-il_ext_cond[iL]
                                        -kj_ext_cond[jL], 1.);
                            numer = il_q_dm_cond * kj_q_cond[jL];
                            if (numer/denom < thresh_K) {
                                TDM_COUNT(profile_counts, TDM_QQR_SKIP);
                                continue;
                            }
                            pL2 = Ls + jL*3;
                            vec3_add(vtmp2, vtmp1, pL2, 1.);
                            shift_bas(env_loc, env, vtmp2, jptrxyz, 0);
                            TDM_COUNT(profile_counts, TDM_INTOR_CALL);
                            if (profile_times != NULL) {
                                tick = wall_time();
                            }
                            intor_nonzero = (*intor)(
                                buf, NULL, shls, atm, natm, bas, nbas,
                                env_loc, cintopt, cache);
                            if (profile_times != NULL) {
                                profile_times[TDM_INTOR_TIME] +=
                                    wall_time() - tick;
                            }
                            if (intor_nonzero != 0) {
                                TDM_COUNT(profile_counts, TDM_INTOR_NONZERO);
                                for (i = 0; i < dilkj; i++) {
                                    eri[i] += buf[i];
                                }
                            }
                        } // jL
                    } // bvk_kL
                } // iL

                lk_dm = dm + dm_iL*Nao*Nao + laor*Nao + kaor;
                for (bvk_jL = 0; bvk_jL < bvk_nimgs; bvk_jL++) {
                    bvk_ij_out = ij_out + bvk_jL*Nao*Nao;
                    eri = eri_bvk + bvk_jL*dilkjmax;
                    TDM_COUNT(profile_counts, TDM_CONTRACT_CALL);
                    if (profile_times != NULL) {
                        tick = wall_time();
                    }
                    contract_eri_dm_shl(bvk_ij_out, eri, lk_dm,
                                        di, dl, dk, dj, Nao);
                    if (profile_times != NULL) {
                        profile_times[TDM_CONTRACT_TIME] +=
                            wall_time() - tick;
                    }
                } // bvk_jL
            } // dm_iL
        } // kshr
    } // lshr
    free(eri_bvk);
}

static void tdm_k_drv(int (*intor)(), double *out,
                      CINTOpt *cintopt, int dm_nimgs, double *dm_Ls, double *dm,
                      int nimgs, double *Ls,
                      int bvk_nimgs, int *bvk_cell_loc, int *bvkadd_loc,
                      int *bvkidx_by_dmcell,
                      double *q_cond, double *ext_cond, double *R_cond,
                      double *dm_cond, double thresh_K,
                      int *ao_loc, int *atm, int natm,
                      int *bas, int nbas, double *env, int nenv,
                      uint64_t *profile_counts, double *profile_times)
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
    size_t ip;

    if (profile_counts != NULL) {
        for (ip = 0; ip < TDM_NCOUNTS; ip++) {
            profile_counts[ip] = 0;
        }
    }
    if (profile_times != NULL) {
        for (ip = 0; ip < TDM_NTIMES; ip++) {
            profile_times[ip] = 0.;
        }
    }

#pragma omp parallel
{
    size_t ij, ish, jsh, ijao;
    uint64_t counts[TDM_NCOUNTS] = {0};
    double times[TDM_NTIMES] = {0.};
    double tick;
    int ishr, jshr;
    // buf = [eribuf, cache]
    double *buf = malloc(sizeof(double) *
                         (dimax*dlmax*dkmax*djmax+cache_size));
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);
    double *ij_out;
    if (profile_times != NULL) {
        tick = wall_time();
    }
#pragma omp for schedule(dynamic)
    for (ij = 0; ij < Nbas*Nbas; ij++) {
        ishr = ij/Nbas;
        jshr = ij%Nbas;
        ish = ishr + ish0;
        jsh = jshr + jsh0;
        ijao = (ao_loc[ish]-ao_loc[ish0])*Nao + ao_loc[jsh]-ao_loc[jsh0];
        ij_out = out + ijao;
        PBCtdm_contract_eri_dm(
            intor, ij_out, cintopt,
            q_cond, ext_cond, R_cond, dm_cond, thresh_K,
            buf,
            dm_nimgs, dm_Ls, dm,
            nimgs, Ls,
            bvk_nimgs, bvk_cell_loc, bvkadd_loc,
            bvkidx_by_dmcell,
            ishr, jshr,
            ao_loc, atm, natm, bas, nbas,
            env, nenv, env_loc,
            profile_counts == NULL ? NULL : counts,
            profile_times == NULL ? NULL : times);
    } // ij
    if (profile_times != NULL) {
        times[TDM_TOTAL_TIME] += wall_time() - tick;
    }
#pragma omp critical
    {
        if (profile_counts != NULL) {
            for (ip = 0; ip < TDM_NCOUNTS; ip++) {
                profile_counts[ip] += counts[ip];
            }
        }
        if (profile_times != NULL) {
            for (ip = 0; ip < TDM_NTIMES; ip++) {
                profile_times[ip] += times[ip];
            }
        }
    }
    free(buf);
    free(env_loc);
}
}

void PBCtdm_k_drv(int (*intor)(), double *out,
                  CINTOpt *cintopt, int dm_nimgs, double *dm_Ls, double *dm,
                  int nimgs, double *Ls,
                  int bvk_nimgs, int *bvk_cell_loc, int *bvkadd_loc,
                  int *bvkidx_by_dmcell,
                  double *q_cond, double *ext_cond, double *R_cond,
                  double *dm_cond, double thresh_K,
                  int *ao_loc, int *atm, int natm,
                  int *bas, int nbas, double *env, int nenv)
{
    tdm_k_drv(intor, out, cintopt,
              dm_nimgs, dm_Ls, dm, nimgs, Ls,
              bvk_nimgs, bvk_cell_loc, bvkadd_loc, bvkidx_by_dmcell,
              q_cond, ext_cond, R_cond, dm_cond, thresh_K,
              ao_loc, atm, natm, bas, nbas, env, nenv, NULL, NULL);
}

void PBCtdm_k_drv_profile(int (*intor)(), double *out,
                          CINTOpt *cintopt, int dm_nimgs, double *dm_Ls,
                          double *dm, int nimgs, double *Ls,
                          int bvk_nimgs, int *bvk_cell_loc, int *bvkadd_loc,
                          int *bvkidx_by_dmcell,
                          double *q_cond, double *ext_cond, double *R_cond,
                          double *dm_cond, double thresh_K,
                          int *ao_loc, int *atm, int natm,
                          int *bas, int nbas, double *env, int nenv,
                          uint64_t *profile_counts, double *profile_times)
{
    tdm_k_drv(intor, out, cintopt,
              dm_nimgs, dm_Ls, dm, nimgs, Ls,
              bvk_nimgs, bvk_cell_loc, bvkadd_loc, bvkidx_by_dmcell,
              q_cond, ext_cond, R_cond, dm_cond, thresh_K,
              ao_loc, atm, natm, bas, nbas, env, nenv,
              profile_counts, profile_times);
}
