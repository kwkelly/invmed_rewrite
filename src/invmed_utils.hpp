#pragma once

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;


struct FMMData{
  const pvfmm::Kernel<double>* kernel;
  FMM_Mat_t* fmm_mat;
  FMM_Tree_t* tree;
  PetscInt m,n,M,N,l,L;
  std::vector<double> eta;
  std::vector<double> u_ref;
  Vec phi_0_vec;
  Vec phi_0;
  Vec eta_val;
  pvfmm::BoundaryType bndry;
};

int ptwise_coeff_mult(Vec &X, FMMData *fmm_data);

int eval_function_at_nodes(FMMData *fmm_data, void (*func)(double* coord, int n, double* out), std::vector<double> &func_vec);

int tree2vec(FMMData fmm_data, Vec& Y);

int vec2tree(Vec& Y, FMMData fmm_data);

int FMM_Init(MPI_Comm& comm, FMMData *fmm_data);

int FMMCreateShell(FMMData *fmm_data, Mat *A);

int FMMDestroy(FMMData *fmm_data);

int PtWiseTreeMult(FMMData &fmm_data, FMM_Tree_t &tree2);

int mult(Mat M, Vec U, Vec Y);

int eval_function_at_nodes(FMMData *fmm_data, void (*func)(double* coord, int n, double* out), std::vector<double> &func_vec);

int eval_cheb_at_nodes(FMMData *fmm_data, Vec &val_vec);

int CompPhiUsingBorn(Vec &true_sol, FMMData &fmm_data);
