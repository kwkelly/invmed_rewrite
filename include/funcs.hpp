#include <vector>
#ifndef FUNCS_HPP
#define FUNCS_HPP

void eta_fn(const double* coord, int n, double* out);

void eta_smooth_fn(const double* coord, int n, double* out);

void pt_sources_fn(const double* coord, int n, double* out);

void zero_fn(const double* coord, int n, double* out);

void one_fn(const double* coord, int n, double* out);

void eye_fn(const double* coord, int n, double* out);

void ctr_pt_sol_fn(const double* coord, int n, double* out);

void ctr_pt_sol_i_fn(const double* coord, int n, double* out);

void ctr_pt_sol_neg_conj_fn(const double* coord, int n, double* out);

void ctr_pt_sol_conj_fn(const double* coord, int n, double* out);

void ctr_pt_sol_prod_fn(const double* coord, int n, double* out);

void ctr_pt_sol_conj_prod_fn(const double* coord, int n, double* out);

void cs_fn(const double* coord, int n, double* out);

void sc_fn(const double* coord, int n, double* out);

void sc_osc_fn(const double* coord, int n, double* out);

void scc_fn(const double* coord, int n, double* out);

void sc2_fn(const double* coord, int n, double* out);

void twosin_fn(const double* coord, int n, double* out);

void ctr_pt_source_fn(const double* coord, int n, double* out);

void int_test_fn(const double* coord, int n, double* out);

void int_test_sol_fn(const double* coord, int n, double* out);

void poly_fn(const double* coord, int n, double* out);

void poly_prod_fn(const double* coord, int n, double* out);

void prod_fn(const double* coord, int n, double* out);

void linear_fn(const double* coord, int n, double* out);

void linear_prod_fn(const double* coord, int n, double* out);

void pt_src_sol_fn(const double* coord, int n, double* out, const double* src_coord);

void linear_comb_of_pt_src(const double* coord, int n, double* out, const std::vector<double> &coeffs, const std::vector<double> &src_coords);

void k2_fn(const double* coord, int n, double* out);

void eta_plus_k2_fn(const  double* coord, int n, double* out);

void mask_fn(const  double* coord, int n, double* out);

void cmask_fn(const  double* coord, int n, double* out);

void eta2_fn(const  double* coord, int n, double* out);

void two_pt_sol_fn(const double* coord, int n, double* out);

void eight_pt_sol_fn(const double* coord, int n, double* out);

void cos2pix_fn(const  double* coord, int n, double* out);

void sin2pix_fn(const  double* coord, int n, double* out);

void sc_osc_fn(const double* coord, int n, double* out);

#endif
