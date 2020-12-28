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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include "cint.h"
#include "pbc/optimizer.h"

#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])

void PBCinit_optimizer(PBCOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        PBCOpt *opt0 = malloc(sizeof(PBCOpt));
        opt0->rrcut = NULL;
        opt0->fprescreen = &PBCnoscreen;
        *opt = opt0;
}

void PBCdel_optimizer(PBCOpt **opt)
{
        PBCOpt *opt0 = *opt;
        if (!opt0) {
                return;
        }

        if (!opt0->rrcut) {
                free(opt0->rrcut);
        }
        free(opt0);
        *opt = NULL;
}


int PBCnoscreen(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        return 1;
}

int PBCrcut_screen(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const double *ri = env + atm[bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double *rj = env + atm[bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        double rr = SQUARE(rirj);
        return (rr < opt->rrcut[ish] || rr < opt->rrcut[jsh]);
}

void PBCset_rcut_cond(PBCOpt *opt, double *rcut,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->rrcut) {
                free(opt->rrcut);
        }
        opt->rrcut = (double *)malloc(sizeof(double) * nbas);
        opt->fprescreen = &PBCrcut_screen;

        int i;
        for (i = 0; i < nbas; i++) {
                opt->rrcut[i] = rcut[i] * rcut[i];
        }
}

/* Single Rc_loc and R12_cut_lst determined from the most diffuse AOs and
   auxiliary basis */
void PBCinit_optimizer1(PBCOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
       PBCOpt *opt0 = malloc(sizeof(PBCOpt));
       opt0->rrcut = NULL;
       opt0->rc_cut = NULL;
       opt0->r12_cut = NULL;
       opt0->bas_exp = NULL;
       opt0->fprescreen = &PBCnoscreen;
       *opt = opt0;
}

void PBCdel_optimizer1(PBCOpt **opt)
{
       PBCOpt *opt0 = *opt;
       if (!opt0) {
               return;
       }

       if (!opt0->r12_cut) {
               free(opt0->r12_cut);
       }

       if (!opt0->rc_cut) {
               free(opt0->rc_cut);
       }

       if (!opt0->bas_exp) {
               free(opt0->bas_exp);
       }
       free(opt0);
       *opt = NULL;
}

int PBCrcut_screen1(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const int ksh = shls[2];
        const double *ri = env + atm[bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double *rj = env + atm[bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double ei = opt->bas_exp[ish];
        const double ej = opt->bas_exp[jsh];
        const double inveij = 1./(ei + ej);
        const double *rk = env + atm[bas[ATOM_OF+ksh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        double rcij[3];
        rcij[0] = (ri[0]*ei + rj[0]*ej)*inveij - rk[0];
        rcij[1] = (ri[1]*ei + rj[1]*ej)*inveij - rk[1];
        rcij[2] = (ri[2]*ei + rj[2]*ej)*inveij - rk[2];
        const double rcrc = SQUARE(rcij);
        if (rcrc > opt->rc_cut[0])
            return 0;

        int irc = (int)sqrt(rcrc);
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        double rr = SQUARE(rirj);
        return rr < opt->r12_cut[irc];
}

void PBCset_rcut_cond1(PBCOpt *opt, double Rc_cut, double *R12_cut_lst,
                       double *bas_exp,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        printf("\n****** Single Rc, R12_cut_lst Prescreening ******\n");
        if (opt->rc_cut) {
                free(opt->rc_cut);
        }
        if (opt->r12_cut) {
                free(opt->r12_cut);
        }
        if (opt->bas_exp) {
                free(opt->bas_exp);
        }
        int nR12 = (int)Rc_cut + 1;
        printf("nR12 = %d  Rc_cut = %4.1f\n", nR12, Rc_cut);
        opt->rc_cut = (double *)malloc(sizeof(double));
        opt->r12_cut = (double *)malloc(sizeof(double) * nR12);
        opt->bas_exp = (double *)malloc(sizeof(double) * nbas);
        opt->fprescreen = &PBCrcut_screen1;

        opt->rc_cut[0] = Rc_cut * Rc_cut;

        int i;
        for (i = 0; i < nR12; i++) {
                opt->r12_cut[i] = R12_cut_lst[i] * R12_cut_lst[i];
                // printf("%d %.3f\n", i, opt->r12_cut[i]);
        }

        for (i = 0; i < nbas; i++) {
                opt->bas_exp[i] = bas_exp[i];
                // printf("  %d %.6f\n", i, opt->bas_exp[i]);
        }

        opt->nbas = nbas;
}

/* Shellpair and aux shell specific Rc_cut and R12_cut_lst determined from the
   most diffuse AOs from the shellpair and shell. */
int PBCrcut_screen2(int *shls, PBCOpt *opt, int *atm, int *bas, double *env,
                    const double *Lc)
{
        if (!opt) {
                return 1; // no screen
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const int ksh = shls[2];
        const int jsh0 = jsh - opt->nbas;
        const int ksh0 = ksh - 2*opt->nbas;
        const double *rk = env + atm[bas[ATOM_OF+ksh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        double rcij[3];
        rcij[0] = Lc[0] - rk[0];
        rcij[1] = Lc[1] - rk[1];
        rcij[2] = Lc[2] - rk[2];
        const double rcrc = SQUARE(rcij);
        const double rc_cut = opt->rc_cut[ksh0*opt->nc_shift0 +
                                          ish*opt->nc_shift1 + jsh0];
        if (rcrc > rc_cut)
            return 0;

        const double *ri = env + atm[bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double *rj = env + atm[bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        int irc = (int)sqrt(rcrc);
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        const double rr = SQUARE(rirj);
        const double r12_cut = opt->r12_cut[ksh0*opt->n12_shift0 +
                                            ish*opt->n12_shift1 +
                                            jsh0*opt->n12_shift2 + irc];

        return rr < r12_cut;
}

void PBCset_rcut_cond2(PBCOpt *opt, int nbas_auxchg, int nc_max,
                       double *Rc_cut_mat, double *R12_cut_mat, double *bas_exp,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        printf("\n****** Shellpair & aux shell-specific Prescreening ******\n");
        if (opt->rc_cut) {
                free(opt->rc_cut);
        }
        if (opt->r12_cut) {
                free(opt->r12_cut);
        }
        if (opt->bas_exp) {
                free(opt->bas_exp);
        }
        int half_nbas = nbas / 2;
        int half_nbas2 = half_nbas * half_nbas;
        opt->nbas = half_nbas;
        // Given i/j/ksh, the index to find Rc_cut in Rc_cut_mat is
        // (ksh-2*half_nbas) * half_nbas2 + ish * half_nbas + jsh-half_nbas
        opt->nc_shift0 = half_nbas2;
        opt->nc_shift1 = half_nbas;
        // Given i/j/ksh, the index to find R12_cut in R12_cut_mat is
        // (ksh-2*half_nbas) * half_nbas2*nc_max + ish * half_nbas*nc_max + (jsh-half_nbas) * nc_max + irc
        opt->n12_shift0 = opt->nc_shift0*nc_max;
        opt->n12_shift1 = opt->nc_shift1*nc_max;
        opt->n12_shift2 = nc_max;
        opt->rc_cut = (double *)malloc(sizeof(double) * nbas_auxchg *
                                       half_nbas2);
        opt->r12_cut = (double *)malloc(sizeof(double) * nbas_auxchg *
                                        half_nbas2 * nc_max);
        opt->bas_exp = (double *)malloc(sizeof(double) * nbas);
        opt->fprescreen = &PBCrcut_screen2;

        int i;
        for (i = 0; i < nbas_auxchg * half_nbas2; i++) {
                opt->rc_cut[i] = Rc_cut_mat[i] * Rc_cut_mat[i];
        }

        for (i = 0; i < nbas_auxchg * half_nbas2 * nc_max; i++) {
                opt->r12_cut[i] = R12_cut_mat[i] * R12_cut_mat[i];
        }

        for (i = 0; i < nbas; i++) {
                opt->bas_exp[i] = bas_exp[i];
        }
}
