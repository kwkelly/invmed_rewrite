#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char* argv[]){

	static char help[] = "\n\
												-eta        <Real>   Inf norm of \\eta\n\
												-ref_tol    <Real>   Tree refinement tolerance\n\
												-min_depth  <Int>    Minimum tree depth\n\
												-max_depth  <Int>    Maximum tree depth\n\
												-fmm_q      <Int>    Chebyshev polynomial degree\n\
												-fmm_m      <Int>    Multipole order (+ve even integer)\n\
												-gmres_tol  <Real>   GMRES residual tolerance\n\
												-gmres_iter <Int>    GMRES maximum iterations\n\
												";
	PetscInt  VTK_ORDER=0;
	PetscInt  INPUT_DOF=2;
	PetscReal  SCAL_EXP=1.0;
	PetscBool  PERIODIC=PETSC_FALSE;
	PetscBool TREE_ONLY=PETSC_FALSE;
	PetscInt  MAXDEPTH  =MAX_DEPTH;// Maximum tree depth
	PetscInt  MINDEPTH   =4;       // Minimum tree depth
	PetscReal       TOL  =1e-3;    // Tolerance
	PetscReal GMRES_TOL  =1e-6;    // Fine mesh GMRES tolerance
	PetscInt  CHEB_DEG  =14;       // Fine mesh Cheb. order
	PetscInt MUL_ORDER  =10;       // Fine mesh mult  order
	PetscInt MAX_ITER  =200;
	PetscReal f_max=1;
	PetscReal eta_=1;

  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,0,help);

  MPI_Comm comm=MPI_COMM_WORLD;
  PetscMPIInt    rank,size;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  // Get command line info! 
  PetscOptionsGetInt (NULL,  "-vtk_order",&VTK_ORDER  ,NULL);
  PetscOptionsGetInt (NULL,        "-dof",&INPUT_DOF  ,NULL);
  PetscOptionsGetReal(NULL,       "-scal",& SCAL_EXP  ,NULL);
  PetscOptionsGetBool(NULL,   "-periodic",& PERIODIC  ,NULL);
  PetscOptionsGetBool(NULL,       "-tree",& TREE_ONLY ,NULL);

  PetscOptionsGetInt (NULL, "-max_depth" ,&MAXDEPTH   ,NULL);
  PetscOptionsGetInt (NULL, "-min_depth" ,&MINDEPTH   ,NULL);
  PetscOptionsGetReal(NULL,   "-ref_tol" ,&      TOL  ,NULL);
  PetscOptionsGetReal(NULL, "-gmres_tol" ,&GMRES_TOL  ,NULL);

  PetscOptionsGetInt (NULL,   "-fmm_q"   ,& CHEB_DEG  ,NULL);
  PetscOptionsGetInt (NULL,   "-fmm_m"   ,&MUL_ORDER  ,NULL);

  PetscOptionsGetInt (NULL, "-gmres_iter",& MAX_ITER  ,NULL);

  PetscOptionsGetReal(NULL,       "-eta" ,&    eta_   ,NULL);

	// Define some stuff!
	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

	// Define new tree
	
	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);	
  const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->cheb_deg = CHEB_DEG;
	eta->mult_order = MUL_ORDER;
	eta->tol = TOL;
	eta->mindepth = MINDEPTH;
	eta->maxdepth = MAXDEPTH;
	eta->fn = eta_fn;
	eta->f_max = eta_;
	eta->dim = 3;
	eta->adap = true;
	eta->data_dof = 1;


	// initialize it
  pvfmm::Profile::Tic("Initialize eta",&comm,true);
		eta->Initialize();
		eta->Write2File("results/eta",0);
  pvfmm::Profile::Toc();
	// Copy data and initialize new tree
	
  pvfmm::Profile::Tic("Initialize pt_sources",&comm,true);
	InvMedTree<FMM_Mat_t> *pt_sources  = new InvMedTree<FMM_Mat_t>(comm);	
	//const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
	//pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	pt_sources->bndry = bndry;
	pt_sources->kernel = kernel;
	pt_sources->cheb_deg = CHEB_DEG;
	pt_sources->mult_order = MUL_ORDER;
	pt_sources->tol = TOL;
	pt_sources->mindepth = MINDEPTH;
	pt_sources->maxdepth = MAXDEPTH;
	pt_sources->fn = pt_sources_fn;
	pt_sources->f_max = 1;
	pt_sources->dim = 3;
	pt_sources->adap = true;
	pt_sources->data_dof = 2;
	pt_sources->Initialize();

	/////////////////////////////////////////

  pvfmm::Profile::Tic("Initialize zero",&comm,true);
	InvMedTree<FMM_Mat_t> *zero  = new InvMedTree<FMM_Mat_t>(comm);	
	//const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
	//pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	zero->bndry = bndry;
	zero->kernel = kernel;
	zero->cheb_deg = CHEB_DEG;
	zero->mult_order = MUL_ORDER;
	zero->tol = TOL;
	zero->mindepth = MINDEPTH;
	zero->maxdepth = MAXDEPTH;
	zero->fn = zero_fn;
	zero->f_max = 1;
	zero->dim = 3;
	zero->adap = true;
	zero->data_dof = 2;
	zero->Initialize();


	/////////////////////////////////////////
  pvfmm::Profile::Tic("Initialize Phi_0",&comm,true);
	InvMedTree<FMM_Mat_t> *phi_0  = new InvMedTree<FMM_Mat_t>(comm);	
	//const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
	//pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	phi_0->bndry = bndry;
	phi_0->kernel = kernel;
	phi_0->cheb_deg = CHEB_DEG;
	phi_0->mult_order = MUL_ORDER;
	phi_0->tol = TOL;
	phi_0->mindepth = MINDEPTH;
	phi_0->maxdepth = MAXDEPTH;
	phi_0->fn = pt_sources_fn;
	phi_0->f_max = 1;
	phi_0->dim = 3;
	phi_0->adap = true;
	phi_0->data_dof = 2;
	phi_0->Initialize();
	//phi_0->fmm_mat = new FMM_Mat_t;
	//phi_0->fmm_mat->Initialize(phi_0->mult_order,phi_0->cheb_deg,*(phi_0->Comm()),phi_0->kernel);

  phi_0->Write2File("results/pt_sources",0);
  pvfmm::Profile::Toc();

	phi_0->InitializeMat();
	phi_0->SetupFMM(phi_0->fmm_mat);
	phi_0->RunFMM();
	phi_0->Copy_FMMOutput();
  phi_0->Write2File("results/phi_0",0);
	return 0;
}
