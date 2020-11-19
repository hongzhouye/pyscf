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
        opt0->rrcut_sp = NULL;
        opt0->LLcut_sp = NULL;
        opt0->ri_bas = NULL;
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
        if (!opt0->rrcut_sp) {
                free(opt0->rrcut_sp);
        }
        if (!opt0->LLcut_sp) {
                free(opt0->LLcut_sp);
        }
        if (!opt0->ri_bas) {
                free(opt0->ri_bas);
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

/* @@HY: screening by shell pair */
int PBCrcut_screen_sp(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const double *ri = env + atm[bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double *rj = env + atm[bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];

        /* Screen sp first: skip this shlpr if either shl lies outside a sphere.

        The distances being computed here (rri & rrj) are really just the lattice shift for the two shells. Ideally one should do this screening directly in e.g., pyscf/lib/pbc/fill_ints.c. However, the code there is of general-purpose, while this screening only applies to GDF/MDF.
        */
        const double *ri0 = opt->ri_bas + ish*3;
        double riri[3];
        riri[0] = ri[0] - ri0[0];
        riri[1] = ri[1] - ri0[1];
        riri[2] = ri[2] - ri0[2];
        double rri = SQUARE(riri);
        double rrcutij = opt->rrcut_sp[ish*opt->nbas+jsh];
        if (rri > rrcutij)
            return 0;
        const double *rj0 = opt->ri_bas + ish*3;
        double rjrj[3];
        rjrj[0] = rj[0] - rj0[0];
        rjrj[1] = rj[1] - rj0[1];
        rjrj[2] = rj[2] - rj0[2];
        double rrj = SQUARE(rjrj);
        if (rrj > rrcutij)
            return 0;

        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        double rr = SQUARE(rirj);
        return (rr < opt->rrcut[ish] || rr < opt->rrcut[jsh]);
}

void PBCset_rcut_cond_sp(PBCOpt *opt, double *rcut,
                         double *rcut_sp, double *Lcut_sp, int nL,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        opt->nbas = nbas;
        int nbas_hlf = nbas / 2;
        if (opt->rrcut) {
                free(opt->rrcut);
        }
        if (opt->rrcut_sp) {
                free(opt->rrcut_sp);
        }
        if (opt->LLcut_sp) {
                free(opt->LLcut_sp);
        }
        if (opt->ri_bas) {
                free(opt->ri_bas);
        }
        opt->rrcut = (double *)malloc(sizeof(double) * nbas);
        opt->rrcut_sp = (double *)malloc(sizeof(double) * nbas_hlf*nbas_hlf);
        opt->LLcut_sp = (double *)malloc(sizeof(double) * nL);
        opt->ri_bas = (double *)malloc(sizeof(double) * nbas*3);
        // opt->fprescreen = &PBCrcut_screen_sp;
        opt->fprescreen = &PBCrcut_screen;

        int i,j;
        for (i = 0; i < nbas; i++) {
                opt->rrcut[i] = rcut[i] * rcut[i];
                const double *ri0 = env +
                    atm[bas[ATOM_OF+i*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
                opt->ri_bas[i*3] = ri0[0];
                opt->ri_bas[i*3+1] = ri0[1];
                opt->ri_bas[i*3+2] = ri0[2];
        }
        for (i = 0; i < nbas_hlf; i++)
                for (j = 0; j < nbas_hlf; j++) {
                        double rij = rcut_sp[i*nbas_hlf+j];
                        opt->rrcut_sp[i*nbas_hlf+j] = rij * rij;
                }
        for (i = 0; i < nL; i++) {
                opt->LLcut_sp[i] = Lcut_sp[i] * Lcut_sp[i];
        }
}
