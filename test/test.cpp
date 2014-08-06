#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <set>



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
	InvMedTree<FMM_Mat_t> *one= new InvMedTree<FMM_Mat_t>(comm);	
  const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	one->bndry = bndry;
	one->kernel = kernel;
	one->cheb_deg = CHEB_DEG;
	one->mult_order = MUL_ORDER;
	one->tol = TOL;
	one->mindepth = MINDEPTH;
	one->maxdepth = MAXDEPTH;
	one->fn = one_fn;
	one->f_max = 1;
	one->dim = 3;
	one->adap = false;
	one->data_dof = 2;

	// initialize one tree
  pvfmm::Profile::Tic("Initialize one",&comm,true);
		one->Initialize();
		one->Write2File("results/one",0);
  pvfmm::Profile::Toc();

	InvMedTree<FMM_Mat_t>::Multiply(*one,*one);
	one->Write2File("results/one_mult",0);

	InvMedTree<FMM_Mat_t>::Add(*one,*one);
	one->Write2File("results/one_add",0);
	//////////////////////////////////////////////////////
	//
	//
	//
	//
	
	InvMedTree<FMM_Mat_t> *pt_sources= new InvMedTree<FMM_Mat_t>(comm);	
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
	pt_sources->adap = false;
	pt_sources->data_dof = 2;

	// initialize one tree
  pvfmm::Profile::Tic("Initialize pt_sources",&comm,true);
		pt_sources->Initialize();
		pt_sources->Write2File("results/pt_sources",0);
  pvfmm::Profile::Toc();

	std::cout << "line 121" << std::endl;

	std::cout << InvMedTree<FMM_Mat_t>::m_instances.size() << std::endl;

	InvMedTree<FMM_Mat_t>::Multiply(*pt_sources,*one);
	pt_sources->Write2File("results/pt_source_times_one",0);

	InvMedTree<FMM_Mat_t>::Add(*pt_sources,*one);
	pt_sources->Write2File("results/pt_sources_plus_one",0);

	// Copy data and initialize new tree
/*	
  pvfmm::Profile::Tic("Initialize zero",&comm,true);
		InvMedTree<FMM_Mat_t> *zero  = new InvMedTree<FMM_Mat_t>(comm);	
		InvMedTree<FMM_Mat_t>::Copy(*zero, *eta);
		zero->fn = zero_fn;
		zero->data_dof = 2;
		zero->f_max = 0;
		zero->Initialize();
		zero->Write2File("results/zero",0);
  pvfmm::Profile::Toc();

	//delete zero;


  pvfmm::Profile::Tic("Initialize one",&comm,true);
		InvMedTree<FMM_Mat_t> *one  = new InvMedTree<FMM_Mat_t>(comm);	
		InvMedTree<FMM_Mat_t>::Copy(*one, *eta);
		one->fn = one_fn;
		one->data_dof = 2;
		one->f_max = 1;
		one->Initialize();
		one->InitializeMat();
		one->Write2File("results/one",0);
    one->SetupFMM(one->fmm_mat);
    one->RunFMM();
    one->Copy_FMMOutput();
		one->Write2File("results/one_fmm",0);
  pvfmm::Profile::Toc();


	InvMedTree<FMM_Mat_t>::Add(*one,*zero);
	one->Write2File("results/one_add",0);

	InvMedTree<FMM_Mat_t>::Add(*eta,*one);
	eta->Write2File("results/etaplusone",0);
*/
/*
  pvfmm::Profile::Tic("Initialize Phi_0",&comm,true);
		InvMedTree<FMM_Mat_t> *phi_0  = new InvMedTree<FMM_Mat_t>(comm);	
		InvMedTree<FMM_Mat_t>::Copy(*phi_0,*pt_sources);
		phi_0->Initialize();
		//phi_0->fmm_mat = new FMM_Mat_t;
		//phi_0->fmm_mat->Initialize(phi_0->mult_order,phi_0->cheb_deg,*(phi_0->Comm()),phi_0->kernel);
  pvfmm::Profile::Toc();
*/
	return 0;
}
