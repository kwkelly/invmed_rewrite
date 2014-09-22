#ifndef FUNCS_HPP
#define FUNCS_HPP

void eta_fn(const double* coord, int n, double* out);

void pt_sources_fn(const double* coord, int n, double* out);

void zero_fn(const double* coord, int n, double* out);

void one_fn(const double* coord, int n, double* out);

void eye_fn(const double* coord, int n, double* out);

void ctr_pt_sol_fn(const double* coord, int n, double* out);

void ctr_pt_sol_i_fn(const double* coord, int n, double* out);

void ctr_pt_sol_prod_fn(const double* coord, int n, double* out);

void sc_fn(const double* coord, int n, double* out);

void scc_fn(const double* coord, int n, double* out);

#endif
