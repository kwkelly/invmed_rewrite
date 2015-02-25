#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <cstdlib>
#include <profile.hpp>
#include <kernel.hpp>
#include "funcs.hpp"
//#include "petsc_utils.hpp"
#include "typedefs.hpp"
#include <pvfmm.hpp>
//#include <pvfmm_common.hpp>
#include "invmed_utils.hpp"

// pt source locations
std::vector<double> pt_src_locs;
// random coefficients
std::vector<double> coeffs;
void phi_0_fn(const double* coord, int n, double* out);
void phi_0_fn(const double* coord, int n, double* out)
{
	linear_comb_of_pt_src(coord, n, out, coeffs, pt_src_locs);
}

//BEGIN COPY FROM UTILS

//END COPY FROM UTILS
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
												-tol				<Real>   GMRES/CG residual tolerance\n\
												-iter				<Int>    GMRES/CG maximum iterations\n\
												-obs				<Int>		 0 for partial, 1 for full\n\
												-alpha      <Real>	 Regularization parameter\n\
												";
	PetscInt  VTK_ORDER=0;
	PetscInt  INPUT_DOF=2;
	PetscReal  SCAL_EXP=1.0;
	PetscBool  PERIODIC=PETSC_FALSE;
	PetscBool TREE_ONLY=PETSC_FALSE;
	PetscInt  MAXDEPTH  =MAX_DEPTH;// Maximum tree depth
	PetscInt  MINDEPTH   =2;       // Minimum tree depth
	PetscReal   REF_TOL  =1e-3;    // Tolerance
	//PetscReal GMRES_TOL  =1e-6;   // Fine mesh GMRES tolerance
	PetscReal			TOL  =1e-6;			// Fine mesh GMRES/CG tolerance
	PetscInt  CHEB_DEG  =14;       // Fine mesh Cheb. order
	PetscInt MUL_ORDER  =10;       // Fine mesh mult  order
	PetscInt MAX_ITER  =200;
	PetscReal f_max=1;
	PetscReal eta_=1;
	PetscInt OBS = 1;
	char SCATTERED_FIELD[] = "solve";
	PetscReal ALPHA = .001;

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
	PetscOptionsGetReal(NULL,   "-ref_tol" ,&  REF_TOL  ,NULL);
	PetscOptionsGetReal(NULL, "-tol" ,						&TOL  ,NULL);
	PetscOptionsGetInt (NULL,   "-fmm_q"   ,& CHEB_DEG  ,NULL);
	PetscOptionsGetInt (NULL,   "-fmm_m"   ,&MUL_ORDER  ,NULL);
	PetscOptionsGetInt (NULL, "-iter",&       MAX_ITER  ,NULL);
	PetscOptionsGetReal(NULL,       "-eta" ,&    eta_   ,NULL);
	PetscOptionsGetInt (NULL, "-obs",&             OBS  ,NULL);
	PetscOptionsGetReal (NULL, "-alpha",&         ALPHA  ,NULL);
	PetscOptionsGetString (NULL, "-scattered", SCATTERED_FIELD ,6, NULL);
	if(strcmp(SCATTERED_FIELD, "solve") * strcmp(SCATTERED_FIELD, "born") != 0){
		std::cout << SCATTERED_FIELD << std::endl;
	//	std::cout << "Invalid option for computing the scattered field" << std::endl;
		return -1;
	}

	pvfmm::Profile::Enable(true);

	// Define some stuff!
	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

	//const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	// Define static variables
	InvMedTree<FMM_Mat_t>::cheb_deg = CHEB_DEG;
	InvMedTree<FMM_Mat_t>::mult_order = MUL_ORDER;
	InvMedTree<FMM_Mat_t>::tol = REF_TOL;
	InvMedTree<FMM_Mat_t>::mindepth = MINDEPTH;
	InvMedTree<FMM_Mat_t>::maxdepth = MAXDEPTH;
	InvMedTree<FMM_Mat_t>::adap = true;
	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;

	// Define new trees
	InvMedTree<FMM_Mat_t> *temp= new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	pt_src_locs = equisph(6,sqrt(2.0)/2);
	//pt_src_locs = randsph(20,1);

	for(int i=0;i<pt_src_locs.size()/3;i++){
		coeffs.push_back(1);
	}

	InvMedTree<FMM_Mat_t> *phi_0 = new InvMedTree<FMM_Mat_t>(comm);
	phi_0->bndry = bndry;
	phi_0->kernel = kernel;
	phi_0->fn = phi_0_fn;
	phi_0->f_max = 4; // not sure about this value reall

	// Set up
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	InvMedTree<FMM_Mat_t> *pt_sources= new InvMedTree<FMM_Mat_t>(comm);
	pt_sources->bndry = bndry;
	pt_sources->kernel = kernel;
	pt_sources->fn = pt_sources_fn;
	pt_sources->f_max = sqrt(500/M_PI);
	pt_sources->CreateTree(true);

	// would have to do this if I used points in the domain and wanted to integrate over them
	// compute phi_0 from the pt_sources
	//phi_0->SetupFMM(fmm_mat);
	//phi_0->RunFMM();
	//phi_0->Copy_FMMOutput();
	//phi_0->Write2File("results/phi_0",0);


	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->fn = eta_fn;
	eta->f_max = .01;
	eta->CreateTree(true);


	// Get the total number of chebyshev nodes.
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);

	// total num of nodes (non ghost, leaf);
	int n_tnodes = (eta->GetNGLNodes()).size();
	int total_size = n_nodes3*n_tnodes;
	std::cout << "total_size: " << total_size << std::endl;

	// Generate the DETECTOR locations
	//std::vector<double> src_coord;
	//std::vector<double> src_coord = randsph(total_size/10,0.5);
	//std::vector<double> src_coord = equisph(total_size/20,0.4);
	//std::vector<double> src_coord1 = equisph(total_size/20,0.3);
	//src_coord.insert(src_coord.end(),src_coord1.begin(),src_coord1.end());
	//std::vector<double> src_coord = randunif(total_size/5);
	std::vector<double> src_coord = equicube(50);
	//for(int i =0;i<src_coord.size();++i){
	//	std::cout << src_coord[i] << std::endl;
	//}
	//std::vector<double> src_coord = test_pts();
	std::cout << "size: " << src_coord.size() << std::endl;


	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	eta->Write2File("results/eta",0);
	phi_0->Write2File("results/pt_sources",0);

	std::vector<double> pt_sources_samples = phi_0->ReadVals(src_coord);
	temp->SetupFMM(fmm_mat);

	// Copy phi_0 to phi and and set it up for FMM
	InvMedTree<FMM_Mat_t> *phi = new InvMedTree<FMM_Mat_t>(comm);
	phi->Copy(phi_0);
	phi->SetupFMM(fmm_mat);

	// create the scattered field
	if(!strcmp(SCATTERED_FIELD,"born")){
		scatter_born(phi_0, eta, phi);
	}

	if(!strcmp(SCATTERED_FIELD,"solve")){
		ScatteredData scattered_data;
		scattered_data.eta = eta;
		scattered_data.temp = temp;
		scattered_data.alpha = ALPHA;

		ierr = scatter_solve(phi_0, scattered_data, phi);CHKERRQ(ierr);
	}

	// Turn phi into the RHS for the inverse medium problem
	phi->Add(phi_0,-1);
	phi->ScalarMultiply(-1);
	phi->Write2File("results/difference",0);

	// Sample phi at the points in src_coord, then apply the transpose
	// operator if we have only partial observations.
	std::vector<double> phi_samples = phi->ReadVals(src_coord);
	pvfmm::PtFMM_Tree* pt_tree = phi->CreatePtFMMTree(src_coord, phi_samples, kernel_conj);

	if(OBS == 0){
		std::vector<double> trg_value;
		pvfmm::PtFMM_Evaluate(pt_tree, trg_value);

		// Insert the values back in
		phi->Trg2Tree(trg_value);
		phi->ConjMultiply(phi_0,1);
		phi->Write2File("results/rhs_after_adj",0);
	}

	// -------------------------------------------------------------------
	// Set up the linear system
	// -------------------------------------------------------------------
	// Set up the operator for CG;
	PetscInt m = phi_0->m;
	PetscInt M = phi_0->M;
	PetscInt n = phi_0->n;
	PetscInt N = phi_0->N;
	Mat A;
	Mat I;
	InvMedData invmed_data;
	invmed_data.temp = temp;
	invmed_data.phi_0 = phi_0;
	invmed_data.alpha = ALPHA;
	if(OBS == 0){
		invmed_data.pt_tree = pt_tree;
		invmed_data.src_coord = src_coord;
		MatCreateShell(comm,m,n,M,N,&invmed_data,&A);
		MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);
		MatSetOption(A,MAT_SPD,PETSC_TRUE);
	}
	else{
		MatCreateShell(comm,m,n,M,N,&invmed_data,&A);
		MatShellSetOperation(A,MATOP_MULT,(void(*)(void))fullmult);
	}
	Vec sol ,rhs;
	VecCreateMPI(comm,n,PETSC_DETERMINE,&sol);
	VecCreateMPI(comm,n,PETSC_DETERMINE,&rhs);
	tree2vec(phi,rhs);
	//VecView(rhs, PETSC_VIEWER_STDOUT_SELF);

	KSP ksp;
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
	ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

	if(OBS==0){
		KSPSetType(ksp  ,KSPCG);
	}
	else{
		KSPSetType(ksp  ,KSPGMRES);
	}
	KSPSetNormType(ksp  , KSP_NORM_UNPRECONDITIONED);
	/*
	 * PetscErrorCode  KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits)
	 *
	 * ksp	- the Krylov subspace context
	 * rtol		- the relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm
	 * abstol		- the absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm
	 * dtol		- the divergence tolerance, amount (possibly preconditioned) residual norm can increase before KSPConvergedDefault() concludes that the method is diverging
	 * maxits		- maximum number of iterations to use
	 */
	ierr = KSPSetTolerances(ksp  ,TOL  ,PETSC_DEFAULT,PETSC_DEFAULT,MAX_ITER); CHKERRQ(ierr);

	// SET CG OR GMRES OPTIONS
	if(OBS == 0){
		//	ierr = KSPCGSetType(ksp,KSP_CG_SYMMETRIC); CHKERRQ(ierr);
	}
	else{
		ierr = KSPGMRESSetRestart(ksp  , MAX_ITER  ); CHKERRQ(ierr);
		ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization); CHKERRQ(ierr);
	}
	//PC pc;
	//ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
	//ierr = PCSetFromOptions(pc); CHKERRQ(ierr);
	//ierr = PCSetType(pc,PCNONE); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp  );CHKERRQ(ierr); // needs to be after PCSetFromOptions
	//ierr = KSPMonitorSet(ksp, KSPMonitorDefault, NULL, NULL); CHKERRQ(ierr);

	double time_ksp;
	int    iter_ksp;
	// -------------------------------------------------------------------
	// Solve the linear system
	// -------------------------------------------------------------------
	pvfmm::Profile::Tic("KSPSolve",&comm,true);
	time_ksp=-omp_get_wtime();
	ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);
	MPI_Barrier(comm);
	time_ksp+=omp_get_wtime();
	pvfmm::Profile::Toc();

	KSPConvergedReason reason;
	ierr=KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
	std::cout << reason << std::endl;
	ierr=PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);CHKERRQ(ierr);

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
	PetscReal s_min;
	PetscReal s_max;

	ierr = KSPComputeExtremeSingularValues(ksp, &s_max,&s_min); CHKERRQ(ierr);
	std::cout << "Min singular value: " << s_min << std::endl;
	std::cout << "Max singular value: " << s_max << std::endl;


	{ // Write output
		vec2tree(sol, phi);
		phi->Write2File("results/sol",0);
	}

	{ // check the solution. Probably need to reconsider the norms
		Vec eta_vec;
		VecCreateMPI(comm,n,PETSC_DETERMINE,&eta_vec);
		tree2vec(eta,eta_vec);

		PetscReal diffnorm;
		PetscReal etanorm;
		ierr = VecAXPY(sol,-1,eta_vec);CHKERRQ(ierr);
		ierr = VecNorm(sol,NORM_2,&diffnorm); CHKERRQ(ierr);
		ierr = VecNorm(eta_vec,NORM_2,&etanorm); CHKERRQ(ierr);

		std::cout << "Norms on coefficients" << std::endl;
		std::cout << "||eta||_2 = " << etanorm << std::endl;
		std::cout << "||sol - eta||_2 = " << diffnorm << std::endl;
		std::cout << "||sol - eta||_2/||eta||_2 = " << diffnorm/etanorm << std::endl;
	}

	{
		std::cout << "Norms on values" << std::endl;
		double etanorm = eta->Norm2();
		phi->Add(eta,-1);
		double diffnorm = phi->Norm2();

		std::cout << "||eta||_2 = " << etanorm << std::endl;
		std::cout << "||sol - eta||_2 = " << diffnorm << std::endl;
		std::cout << "||sol - eta||_2/||eta||_2 = " << diffnorm/etanorm << std::endl;
	}

	// Free work space.  All PETSc objects should be destroyed when they
	// are no longer needed.
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
	ierr = VecDestroy(&sol);CHKERRQ(ierr);
	ierr = VecDestroy(&rhs);CHKERRQ(ierr);
	ierr = MatDestroy(&A);CHKERRQ(ierr);

	return 0;
}
