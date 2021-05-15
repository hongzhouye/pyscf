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
#include <string.h>
#include <complex.h>
#include <assert.h>
#include "config.h"
#include "cint.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define ABS(X)          ((X>0)?(X):(-X))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static int shloc_partition(int *kshloc, int *ao_loc, int ksh0, int ksh1, int dkmax)
{
        int ksh;
        int nloc = 0;
        int loclast = ao_loc[ksh0];
        kshloc[0] = ksh0;
        for (ksh = ksh0+1; ksh < ksh1; ksh++) {
                assert(ao_loc[ksh+1] - ao_loc[ksh] < dkmax);
                if (ao_loc[ksh+1] - loclast > dkmax) {
                        nloc += 1;
                        kshloc[nloc] = ksh;
                        loclast = ao_loc[ksh];
                }
        }
        nloc += 1;
        kshloc[nloc] = ksh1;
        return nloc;
}


double get_dsqure(double *ri, double *rj)
{
    double dx = ri[0]-rj[0];
    double dy = ri[1]-rj[1];
    double dz = ri[2]-rj[2];
    return dx*dx+dy*dy+dz*dz;
}
void get_rc(double *rc, double *ri, double *rj, double ei, double ej) {
    double eij = ei+ej;
    rc[0] = (ri[0]*ei + rj[0]*ej) / eij;
    rc[1] = (ri[1]*ei + rj[1]*ej) / eij;
    rc[2] = (ri[2]*ei + rj[2]*ej) / eij;
}
size_t max_shlsize(int *ao_loc, int nbas)
{
    int i, dimax=0;
    for(i=0; i<nbas; ++i) {
        dimax = MAX(dimax,ao_loc[i+1]-ao_loc[i]);
    }
    return dimax;
}

void fill_sr2c2e_g(int (*intor)(), double *out,
                   int comp, CINTOpt *cintopt,
                   int *ao_loc, int *ao_locsup,
                   double *uniq_Rcuts, int *refuniqshl_map,
                   int *refsupshl_loc, int *refsupshl_map,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
{
    size_t IJstart, IJSH, ISH, JSH, Ish, Jsh, I0, I1, J0, J1;
    size_t jsh, jsh_, i, j, i0, jmax, iptrxyz, jptrxyz;
    size_t di, dj, dic, dijc;
    int shls[2];
    double Rcut2, Rij2;
    double *ri, *rj;
    const int dimax = max_shlsize(ao_loc, nbas);
    int shls_slice[4];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);
    double *buf = malloc(sizeof(double)*(dimax*dimax+cache_size));
    double *buf_L, *pbuf, *cache;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

    for(Ish=0; Ish<nbas; ++Ish) {
        ISH = refuniqshl_map[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        IJstart = I0*(I0+1)/2;
        di = I1 - I0;
        dic = di * comp;
        shls[1] = Ish;
        iptrxyz = atm[PTR_COORD+bas[ATOM_OF+Ish*BAS_SLOTS]*ATM_SLOTS];
        ri = env+iptrxyz;
        for(Jsh=0; Jsh<=Ish; ++Jsh) {
            JSH = refuniqshl_map[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dijc = dic * dj;
            buf_L = buf;
            pbuf = buf + dijc;
            cache = pbuf + dijc;
            for(j=0; j<dijc; ++j) {
                buf_L[j] = 0.;
            }
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            Rcut2 = uniq_Rcuts[IJSH]*uniq_Rcuts[IJSH];

            for(jsh_=refsupshl_loc[Jsh]; jsh_<refsupshl_loc[Jsh+1]; ++jsh_) {
                jsh = refsupshl_map[jsh_];
                shls[0] = jsh;
                jptrxyz = atmsup[PTR_COORD+bassup[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
                rj = envsup+jptrxyz;
                Rij2 = get_dsqure(ri,rj);
                if(Rij2 < Rcut2) {
                    if(safe) {
                        envsup[PTR_RANGE_OMEGA] = 0.;
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            for(j=0; j<dijc; ++j) {
                                buf_L[j] += pbuf[j];
                            }
                        }
                        envsup[PTR_RANGE_OMEGA] = omega;
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            for(j=0; j<dijc; ++j) {
                                buf_L[j] -= pbuf[j];
                            }
                        }
                    } else {
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            for(j=0; j<dijc; ++j) {
                                buf_L[j] += pbuf[j];
                            }
                        }
                    }
                }
            }

            i0 = IJstart;
            for(i=0; i<di; ++i) {
                jmax = (Ish==Jsh)?(i+1):(dj);
                for(j=0; j<jmax; ++j) {
                    out[i0+j] = buf_L[i*dj+j];
                }
                i0 += I0+i+1;
            }

            IJstart += dj;
        }
    }

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(buf);
}

void fill_sr2c2e_k(int (*intor)(), double complex *out,
                   int comp, CINTOpt *cintopt,
                   double complex *expLk, int nkpts,
                   int *ao_loc, int *ao_locsup,
                   double *uniq_Rcuts, int *refuniqshl_map,
                   int *refsupshl_loc, int *refsupshl_map,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
{
    double *expLk_r = malloc(sizeof(double) * natmsup*nkpts * OF_CMPLX);
    double *expLk_i = expLk_r + natmsup*nkpts;
    double *expLk_r_, *expLk_i_;
    double phi_r, phi_i, tmp;
    double complex tmpc;
    double complex *outk;
    int i;
    for (i = 0; i < natmsup*nkpts; i++) {
            expLk_r[i] = creal(expLk[i]);
            expLk_i[i] = cimag(expLk[i]);
    }

    size_t IJstart, IJSH, ISH, JSH, Ish, Jsh, I0, I1, J0, J1;
    size_t jsh, jsh_, j, i0, kk, jmax, iptrxyz, jptrxyz, jatm;
    size_t di, dj, dic, dijc;
    int shls[2];
    int nao2 = ao_loc[nbas]*(ao_loc[nbas]+1)/2;
    double Rcut2, Rij2;
    double *ri, *rj;
    const int dimax = max_shlsize(ao_loc, nbas);
    int shls_slice[4];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);
    double *buf = malloc(sizeof(double)*(dimax*dimax+cache_size));
    double *buf_L, *pbuf, *cache;
    char skip;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

    for(Ish=0; Ish<nbas; ++Ish) {
        ISH = refuniqshl_map[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        IJstart = I0*(I0+1)/2;
        di = I1 - I0;
        dic = di * comp;
        shls[1] = Ish;
        iptrxyz = atm[PTR_COORD+bas[ATOM_OF+Ish*BAS_SLOTS]*ATM_SLOTS];
        ri = env+iptrxyz;
        for(Jsh=0; Jsh<=Ish; ++Jsh) {
            JSH = refuniqshl_map[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dijc = dic * dj;
            buf_L = buf;
            pbuf = buf + dijc;
            cache = pbuf + dijc;
            for(j=0; j<dijc; ++j) {
                buf_L[j] = 0.;
            }
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            Rcut2 = uniq_Rcuts[IJSH]*uniq_Rcuts[IJSH];

            for(jsh_=refsupshl_loc[Jsh]; jsh_<refsupshl_loc[Jsh+1]; ++jsh_) {
                jsh = refsupshl_map[jsh_];
                shls[0] = jsh;
                jatm = bassup[ATOM_OF+jsh*BAS_SLOTS];
                expLk_r_ = expLk_r + jatm * nkpts;
                expLk_i_ = expLk_i + jatm * nkpts;
                jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                rj = envsup+jptrxyz;
                Rij2 = get_dsqure(ri,rj);
                if(Rij2 < Rcut2) {
                    skip = 1;
                    if(safe) {
                        envsup[PTR_RANGE_OMEGA] = 0.;
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            skip = 0;
                            envsup[PTR_RANGE_OMEGA] = omega;
                            (*intor)(buf_L, NULL, shls, atmsup, natmsup, bassup,
                                         nbassup, envsup, cintopt, cache);
                            for(j=0; j<dijc; ++j) {
                                pbuf[j] -= buf_L[j];
                            }
                        }
                    } else {
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            skip = 0;
                        }
                    }
                    if(!skip) {
                        for(kk=0; kk<nkpts; ++kk) {
                            phi_r = expLk_r_[kk];
                            phi_i = expLk_i_[kk];
                            outk = out + kk * nao2;
                            i0 = IJstart;
                            for(i=0; i<di; ++i) {
                                jmax = (Ish==Jsh)?(i+1):(dj);
                                for(j=0; j<jmax; ++j) {
                                    tmp = pbuf[i*dj+j];
                                    outk[i0+j] += tmp * phi_r +
                                                  tmp * phi_i * _Complex_I;
                                }
                                i0 += I0+i+1;
                            }
                        }
                    }
                }
            }

            IJstart += dj;
        }
    }

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(expLk_r);
    free(buf);
}

void fill_sr3c2e_g()
{
/*
Need
    refuniqshl_map
    supshlpr_loc[IJsh]
    supshlpr_lst

Pseudocode
    loop Ish
        Get ISH
        Get RI
        loop Jsh
            Get JSH
            Get IJSH <-- tril(ISH,JSH)
            Get RJ
            Calc dIJ
            Get idIJ <-- IJSH,dIJ
*/
    for(Ish=0; Ish<nbas; ++Ish) {
        ei = refexp[Ish];
        ISH = refuniqshl_map[Ish];
        Iptrxyz = atm[PTR_COORD+bas[ATOM_OF+Ish*BAS_SLOTS]*ATM_SLOTS];
        rI = env+Iptrxyz;
        for(Jsh=0; Jsh<=Ish; ++Jsh) {
            ej = refexp[Jsh];
            JSH = refuniqshl_map[Jsh];
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);

            Jptrxyz = atm[PTR_COORD+bas[ATOM_OF+Jsh*BAS_SLOTS]*ATM_SLOTS];
            rJ = env+Jptrxyz;
            idIJ = (int)sqrt(get_dsqure(rI,rJ));

            uniq_RcutsdIJ = uniq_Rcuts + idIJ*nbasaux;

            IJsh = Ish*(Ish+1)/2 + Jsh;
            ijsh0 = supshlpr_loc[IJsh];
            ijsh1 = supshlpr_loc[IJsh+1];
            for(ijsh=ijsh0; ijsh<ijsh1; ++ijsh) {
                ish = supshlpr_i_lst[ijsh];
                jsh = supshlpr_j_lst[ijsh];
                shls[0] = ish;
                shls[1] = jsh;

                iptrxyz = atmsup[PTR_COORD+
                                    bassup[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
                ri = envsup+iptrxyz;
                jptrxyz = atmsup[PTR_COORD+
                                    bassup[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
                rj = envsup+jptrxyz;
                get_rc(rc, ri, rj, ei, ej);

                for(Ksh=0; Ksh<nbasaux; ++Ksh) {
                    ksh = Ksh + nbassup;
                    shls[2] = ksh;

                    rk = envsup + ksh;
                    Rijk2 = get_dsqure(rc, rk);

                }
            }
        }
    }
}

void fill_sr3c2e_g()
{
/*
    atm,bas,env,ao_loc --> ref cell + ref auxcell
    atmsup,bassup,envsup --> supmol + ref auxcell
    uniqd_loc
    supuniqshlpr_loc
    supshlprij_lst (nsupshlpr)
    suprefshl_map
    refexp
    uniq_Rcuts
*/

    const int dimax = max_shlsize(ao_loc, nbas);
    const int djmax = max_shlsize(ao_loc, nbas);
    const int dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas, env);

    for(IJSH=0; IJSH<=nuniqbas2; ++IJSH) {
        for(idIJ=uniqd_loc[IJSH]; idIJ<uniqd_loc[IJSH+1]; ++idIJ) {
            ijsh0 = supuniqshlpr_loc[idIJ];
            ijsh1 = supuniqshlpr_loc[idIJ+1];
            uniq_RcutsdIJ = uniq_Rcuts + idIJ*nbasaux;
            for(ijsh=ijsh0; ijsh<ijsh1; ++ijsh) {
                ish = supshlpri_lst[ijsh];
                jsh = supshlprj_lst[ijsh];
                shls[0] = ish;
                shls[1] = jsh;
                Ish = suprefshl_map[ish];
                Jsh = suprefshl_map[jsh];
                IJsh = (Ish>=Jsh)?(Ish*(Ish+1)/2+Jsh):(Jsh*(Jsh+1)/2+Ish);

                I0 = ao_loc[Ish];
                I1 = ao_loc[Ish+1];
                di = I1 - I0;
                J0 = ao_loc[Jsh];
                J1 = ao_loc[Jsh+1];
                dj = J1 - J0;
                dijc = di*dj*comp;

                ei = refexp[Ish];
                ej = refexp[Jsh];
                iptrxyz = atmsup[PTR_COORD+
                                    bassup[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
                ri = envsup+iptrxyz;
                jptrxyz = atmsup[PTR_COORD+
                                    bassup[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
                rj = envsup+jptrxyz;
                get_rc(rc, ri, rj, ei, ej);
                for(ksh=0; ksh<nbasaux; ++ksh) {
                    kptrxyz = atmaux[PTR_COORD+
                                    basaux[ATOM_OF+ksh*BAS_SLOTS]*ATM_SLOTS];
                    rk = envaux+kptrxyz;
                    Rijk2 = get_dsqure(rc, rk);
                    Rcut2 = uniq_RcutsdIJ[ksh]*uniq_RcutsdIJ[ksh];
                    if(Rijk2 < Rcut2) {
                        dijkc = dijc*(ao_locaux[ksh+1] - ao_locaux[ksh]);
// ------------> add shift for ksh
                        shls[2] = ksh;
// <------------
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            if(Ish == Jsh) {
                                for(k=0; k<dk; ++k) {
                                    for(i=0; i<di; ++i) {
                                        for(j=0; j<dj; ++j) {
                                            out[ksh*nbas2+IJsh] +=
                                                            pbuf[k*dij+ij];
                                        }
                                    }
                                }
                            } else {
                                for(k=0; k<dk; ++k) {
                                    for(i=0; i<di; ++i) {
                                        for(j=0; j<dj; ++j) {
                                            out[ksh*nbas2+IJsh] +=
                                                            pbuf[k*dij+ij];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

/*
void PBCnr2c_drv(int (*intor)(), double *out,
                 int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int njsh = jsh1 - jsh0;
        double *expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
        double *expkL_i = expkL_r + nimgs*nkpts;
        int i;
        for (i = 0; i < nimgs*nkpts; i++) {
                expkL_r[i] = creal(expkL[i]);
                expkL_i[i] = cimag(expkL[i]);
        }
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                                 atm, natm, bas, nbas, env);

// #pragma omp parallel
// {
//         int jsh;
//         double *env_loc = malloc(sizeof(double)*nenv);
//         memcpy(env_loc, env, sizeof(double)*nenv);
//         size_t count = nkpts * OF_CMPLX + nimgs;
//         double *buf = malloc(sizeof(double)*(count*INTBUFMAX10*comp+cache_size));
// #pragma omp for schedule(dynamic)
//         for (jsh = 0; jsh < njsh; jsh++) {
//                 (*fill)(intor, out, nkpts, comp, nimgs, jsh,
//                         buf, env_loc, Ls, expkL_r, expkL_i,
//                         shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
//         }
//         free(buf);
//         free(env_loc);
// }
//         free(expkL_r);
}
*/
