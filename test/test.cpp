#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <set>
//#include "petsc_utils.hpp"
#include "typedefs.hpp"

//BEGIN COPY FROM UTILS
struct InvMedData{
	InvMedTree<FMM_Mat_t>* phi_0;
	InvMedTree<FMM_Mat_t>* temp;
};

int mult(Mat M, Vec U, Vec Y);


template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y);

template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree);

#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	InvMedData invmed_data;
	MatShellGetContext(M, &invmed_data);
	//InvMedTree<FMM_Mat_t>* phi_0 = invmed_data.phi_0;
	//InvMedTree<FMM_Mat_t>* temp = invmed_data.temp;
	std::cout << "phi_0 size " << (((invmed_data.phi_0)->GetNodeList()).size()) << std::endl;

	std::cout << "temp size " << (((invmed_data.temp)->GetNodeList()).size()) << std::endl;
	const MPI_Comm* comm=invmed_data.phi_0->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int omp_p=omp_get_max_threads();
	vec2tree(U,invmed_data.temp);

	invmed_data.temp->Multiply(invmed_data.phi_0,1);

	// Run FMM ( Compute: G[ \eta * u ] )
	invmed_data.temp->ClearFMMData();
	invmed_data.temp->RunFMM();

	// Regularize
	tree2vec(invmed_data.temp,Y);

	PetscScalar alpha = (PetscScalar).00001;
	ierr = VecAXPY(Y,alpha,U);CHKERRQ(ierr);

	return 0;
}
		
#undef __FUNCT__
#define __FUNCT__ "tree2vec"
template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y){
	PetscErrorCode ierr;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());
		int SCAL_EXP = 1;

		PetscScalar *Y_ptr;
		ierr = VecGetArray(Y, &Y_ptr);

		#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				double s=std::pow(0.5,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++) Y_ptr[j+Y_offset]=coeff_vec[j]*s;
			}
		}
		ierr = VecRestoreArray(Y, &Y_ptr);
	}

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "vec2tree"
template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree){
	PetscErrorCode ierr;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());
		int SCAL_EXP = 1;

		const PetscScalar *Y_ptr;
		ierr = VecGetArrayRead(Y, &Y_ptr);

		#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++) coeff_vec[j]=PetscRealPart(Y_ptr[j+Y_offset])*s;
				nlist[i]->DataDOF()=data_dof;
			}
		}
	}

	return 0;
}

//END COPY FROM UTILS

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

	pvfmm::Profile::Enable(true);
	// Define some stuff!
//	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
//	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

  const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


//	InvMedTree<FMM_Mat_t>::bndry_ = bndry;
//	InvMedTree<FMM_Mat_t>::kernel_ = kernel;
	InvMedTree<FMM_Mat_t>::cheb_deg = CHEB_DEG;
	InvMedTree<FMM_Mat_t>::mult_order = MUL_ORDER;
	InvMedTree<FMM_Mat_t>::tol = TOL;
	InvMedTree<FMM_Mat_t>::mindepth = MINDEPTH;
	InvMedTree<FMM_Mat_t>::maxdepth = MAXDEPTH;
	InvMedTree<FMM_Mat_t>::adap = true;
	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;

	// Define new trees
	
	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);	
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = one_fn;
	one->f_max = 0;
	
	InvMedTree<FMM_Mat_t> *pt_sources= new InvMedTree<FMM_Mat_t>(comm);	
	pt_sources->bndry = bndry;
	pt_sources->kernel = kernel;
	pt_sources->fn = pt_sources_fn;
	pt_sources->f_max = 1;

	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);	
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->fn = pt_sources_fn;
	eta->f_max = 1;


	// Initialize the trees. We do this all at once so that they have the same
	// structure so that we may add them.
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	FMM_Mat_t* fmm_mat = new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	pt_sources->SetupFMM(fmm_mat);
	pt_sources->RunFMM();
	pt_sources->Copy_FMMOutput();

	Vec one_vec;
	Vec eta_vec;
  VecCreateMPI(comm,one->m,PETSC_DETERMINE,&one_vec);
  VecCreateMPI(comm,eta->m,PETSC_DETERMINE,&eta_vec);
	tree2vec(one, one_vec);
	tree2vec(eta, eta_vec);
	//petsc_utils::vec2tree(zero_vec, zero);

  // -------------------------------------------------------------------
  // Set up the linear system
  // -------------------------------------------------------------------
	// Set up the operator for CG;
	PetscInt m = pt_sources->m;
	PetscInt M = pt_sources->M;
	PetscInt n = pt_sources->n;
	PetscInt N = pt_sources->N;
	Mat A;
	InvMedData invmed_data;
	invmed_data.temp = one;
	invmed_data.phi_0 = pt_sources;
	std::cout << "In test.cpp " << (invmed_data.temp->GetNodeList()).size() << std::endl;
  MatCreateShell(comm,m,n,M,N,&invmed_data,&A);
  MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);
	//Vec sol ,rhs;
  //VecCreateMPI(comm,n,PETSC_DETERMINE,&sol);
  //VecCreateMPI(comm,n,PETSC_DETERMINE,&rhs);
	//petsc_utils::tree2vec(phi,rhs);
	//VecView(rhs, PETSC_VIEWER_STDOUT_SELF);
	
	std::vector<FMMNode_t*> nlist1 = (invmed_data.phi_0)->GetNodeList();
	std::cout << "phi_0 size before mult " <<  nlist1.size() << std::endl;
	std::vector<FMMNode_t*> nlist2 = (invmed_data.temp)->GetNodeList();
	std::cout << "temp size before mult" << nlist2.size() << std::endl;
	MatMult(A,one_vec,eta_vec);
/*
  KSP ksp;
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  KSPSetType(ksp  ,KSPGMRES);
  KSPSetNormType(ksp  , KSP_NORM_UNPRECONDITIONED);
  KSPSetTolerances(ksp  ,GMRES_TOL  ,PETSC_DEFAULT,PETSC_DEFAULT,MAX_ITER  );
  //KSPGMRESSetRestart(ksp  , MAX_ITER  );
  KSPGMRESSetRestart(ksp  , 100  );
  ierr = KSPSetFromOptions(ksp  );CHKERRQ(ierr);


	double time_ksp;
	int    iter_ksp;
  // -------------------------------------------------------------------
  // Solve the linear system
  // -------------------------------------------------------------------
	std::cout << "Outside the solve" << std::endl;
  pvfmm::Profile::Tic("KSPSolve",&comm,true);
  time_ksp=-omp_get_wtime();
  ierr = KSPSolve(ksp,one_vec,eta_vec);CHKERRQ(ierr);
  MPI_Barrier(comm);
  time_ksp+=omp_get_wtime();
  pvfmm::Profile::Toc();

  // View info about the solver
  KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // -------------------------------------------------------------------
  // Check solution and clean up
  // -------------------------------------------------------------------

  // Iterations
  PetscInt       its;
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its);CHKERRQ(ierr);
  iter_ksp=its;
*/
  { // Write output
    vec2tree(eta_vec, eta);
    eta->Write2File("results/sol",0);
  }

  // Free work space.  All PETSc objects should be destroyed when they
  // are no longer needed.
  //ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  //ierr = VecDestroy(&sol);CHKERRQ(ierr);
  //ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

	return 0;
}
