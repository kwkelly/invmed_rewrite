#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include <kernel.hpp>
#include "funcs.hpp"
//#include "petsc_utils.hpp"
#include "typedefs.hpp"
#include <pvfmm.hpp>
#include <pvfmm_common.hpp>
//BEGIN COPY FROM UTILS
struct InvMedData{
	InvMedTree<FMM_Mat_t>* phi_0;
	InvMedTree<FMM_Mat_t>* temp;
	pvfmm::PtFMM_Tree* pt_tree;
	std::vector<double> src_coord;
};

#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y);

#undef __FUNCT__
#define __FUNCT__ "tree2vec"
template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y);

#undef __FUNCT__
#define __FUNCT__ "vec2tree"
template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree);

void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k);
void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k){
#ifndef __MIC__
	pvfmm::Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(24*dof));
#endif
	for(int t=0;t<trg_cnt;t++){
		for(int i=0;i<dof;i++){
			double p[2]={0,0};
			for(int s=0;s<src_cnt;s++){
				double dX_reg=r_trg[3*t ]-r_src[3*s ];
				double dY_reg=r_trg[3*t+1]-r_src[3*s+1];
				double dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
				double R = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
				if (R!=0){
					R = sqrt(R);
					double invR=1.0/R;
					invR = invR/(4*const_pi<double>());
					double G[2]={cos(k*R)*invR, sin(k*R)*invR};
					p[0] += v_src[(s*dof+i)*2+0]*G[0] - v_src[(s*dof+i)*2+1]*G[1];
					p[1] += v_src[(s*dof+i)*2+0]*G[1] + v_src[(s*dof+i)*2+1]*G[0];
				}
			}
			k_out[(t*dof+i)*2+0] += p[0];
			k_out[(t*dof+i)*2+1] += p[1];
		}
	}
}

void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, 1);
};

void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, -1);
};

const pvfmm::Kernel<double> helm_kernel=pvfmm::BuildKernel<double, helm_kernel_fn>("helm_kernel", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> helm_kernel_conj=pvfmm::BuildKernel<double, helm_kernel_conj_fn>("helm_kernel_conj", 3, std::pair<int,int>(2,2));

#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	// Get context
	InvMedData *invmed_data = NULL;
	MatShellGetContext(M, &invmed_data);
	InvMedTree<FMM_Mat_t>* phi_0 = invmed_data->phi_0;
	InvMedTree<FMM_Mat_t>* temp = invmed_data->temp;
	pvfmm::PtFMM_Tree* pt_tree = invmed_data->pt_tree;
	std::vector<double> src_coord = invmed_data->src_coord;

	const MPI_Comm* comm=invmed_data->phi_0->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	//int omp_p=omp_get_max_threads();
	vec2tree(U,temp);

	temp->Multiply(phi_0,1);

	// Run FMM ( Compute: G[ \eta * u ] )
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();

	// Sample at the points in src_coord, then apply the transpose
	// operator.
	std::vector<double> src_values = temp->ReadVals(src_coord);

//	std::cout << "src_values[i]" << std::endl;
//	for(int i=0;i<src_values.size();i++){
//		std::cout << src_values[i] << std::endl;
//	}
	InvMedTree<FMM_Mat_t>::SetSrcValues(src_coord,src_values,pt_tree);

	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value);

	// Insert the values back in
	temp->Trg2Tree(trg_value);
	
	// Ptwise multiply by the conjugate of phi_0
	temp->ConjMultiply(phi_0,1);


	// Regularize
	tree2vec(temp,Y);

	//PetscScalar alpha = (PetscScalar)1e-10;
	//ierr = VecAXPY(Y,alpha,U);CHKERRQ(ierr);

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
	//PetscReal GMRES_TOL  =1e-6;   // Fine mesh GMRES tolerance
	PetscReal CG_TOL  =1e-6;    	// Fine mesh CG tolerance
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
  //PetscOptionsGetReal(NULL, "-gmres_tol" ,&GMRES_TOL  ,NULL);
  PetscOptionsGetReal(NULL, "-cg_tol" ,&CG_TOL  ,NULL);

  PetscOptionsGetInt (NULL,   "-fmm_q"   ,& CHEB_DEG  ,NULL);
  PetscOptionsGetInt (NULL,   "-fmm_m"   ,&MUL_ORDER  ,NULL);

  PetscOptionsGetInt (NULL, "-gmres_iter",& MAX_ITER  ,NULL);

  PetscOptionsGetReal(NULL,       "-eta" ,&    eta_   ,NULL);

	pvfmm::Profile::Enable(true);

	// Define some stuff!

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
  //const pvfmm::Kernel<double>* kernel=&pvfmm::ker_helmholtz;
  const pvfmm::Kernel<double>* kernel=&helm_kernel;
  const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	// Set up the positions of the detectors (must be outside the object)

	std::vector<double> src_coord;
	src_coord.push_back(0.3);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.7);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.3);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.7);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.3);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.7);

	// Define static variables
	InvMedTree<FMM_Mat_t>::cheb_deg = CHEB_DEG;
	InvMedTree<FMM_Mat_t>::mult_order = MUL_ORDER;
	InvMedTree<FMM_Mat_t>::tol = TOL;
	InvMedTree<FMM_Mat_t>::mindepth = MINDEPTH;
	InvMedTree<FMM_Mat_t>::maxdepth = MAXDEPTH;
	InvMedTree<FMM_Mat_t>::adap = true;
	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;

	// Define new trees
	InvMedTree<FMM_Mat_t> *pt_sources= new InvMedTree<FMM_Mat_t>(comm);	
	pt_sources->bndry = bndry;
	pt_sources->kernel = kernel;
	pt_sources->fn = pt_sources_fn;
	pt_sources->f_max = sqrt(500/M_PI);

	InvMedTree<FMM_Mat_t> *temp= new InvMedTree<FMM_Mat_t>(comm);	
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);	
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->fn = eta_fn;
	eta->f_max = .01;

	InvMedTree<FMM_Mat_t> *phi_0 = new InvMedTree<FMM_Mat_t>(comm);	
	phi_0->bndry = bndry;
	phi_0->kernel = kernel;
	phi_0->fn = pt_sources_fn;
	phi_0->f_max = sqrt(500/M_PI);

	// Set up
	InvMedTree<FMM_Mat_t>::SetupInvMed();

  FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	eta->Write2File("results/eta",0);
	phi_0->Write2File("results/pt_sources",0);
	// compute phi_0
	phi_0->SetupFMM(fmm_mat);
	phi_0->RunFMM();
	phi_0->Copy_FMMOutput();
  phi_0->Write2File("results/phi_0",0);

	std::vector<double> pt_sources_samples = pt_sources->ReadVals(src_coord);
	temp->SetupFMM(fmm_mat);

  // Copy phi_0 to phi and and set it up for FMM	
	InvMedTree<FMM_Mat_t> *phi = new InvMedTree<FMM_Mat_t>(comm);	
	phi->Copy(phi_0);
  phi->Write2File("results/phi_0_copy",0);
	phi->SetupFMM(fmm_mat);

	
  // -------------------------------------------------------------------
  // Compute phi using the Born approximation
  // -------------------------------------------------------------------
	phi->Multiply(eta,-1);  
	phi->RunFMM();
	phi->Copy_FMMOutput();
	phi->Add(phi_0,1);
  phi->Write2File("results/phi",0);

	phi->Add(phi_0,-1);
	phi->ScalarMultiply(-1);
  phi->Write2File("results/born_difference",0);

	// Sample phi at the points in src_coord, then apply the transpose
	// operator.
	
	std::vector<double> phi_samples = phi->ReadVals(src_coord);
	pvfmm::PtFMM_Tree* pt_tree = phi->CreatePtFMMTree(src_coord, phi_samples, kernel_conj);

	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value);

	// Insert the values back in
	phi->Trg2Tree(trg_value);
	phi->ConjMultiply(phi_0,1);
	//phi->ScalarMultiply(10000000);
	phi->Write2File("results/rhs_after_adj",0);

  // -------------------------------------------------------------------
  // Set up the linear system
  // -------------------------------------------------------------------
	// Set up the operator for CG;
	PetscInt m = phi_0->m;
	PetscInt M = phi_0->M;
	PetscInt n = phi_0->n;
	PetscInt N = phi_0->N;
	Mat A;
	InvMedData invmed_data;
	invmed_data.temp = temp;
	invmed_data.phi_0 = phi_0;
	invmed_data.pt_tree = pt_tree;
	invmed_data.src_coord = src_coord;
  MatCreateShell(comm,m,n,M,N,&invmed_data,&A);
  MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);
	Vec sol ,rhs;
  VecCreateMPI(comm,n,PETSC_DETERMINE,&sol);
  VecCreateMPI(comm,n,PETSC_DETERMINE,&rhs);
	tree2vec(phi,rhs);
	//VecView(rhs, PETSC_VIEWER_STDOUT_SELF);

  KSP ksp;
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  //KSPSetType(ksp  ,KSPGMRES);
  KSPSetType(ksp  ,KSPCG);
  KSPSetNormType(ksp  , KSP_NORM_UNPRECONDITIONED);
  KSPSetTolerances(ksp  ,CG_TOL  ,PETSC_DEFAULT,PETSC_DEFAULT,MAX_ITER  );
  //KSPGMRESSetRestart(ksp  , MAX_ITER  );
  //KSPGMRESSetRestart(ksp  , 100  );
	//KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);
  ierr = KSPSetFromOptions(ksp  );CHKERRQ(ierr);
	ierr = KSPMonitorSet(ksp, KSPMonitorDefault, NULL, NULL); CHKERRQ(ierr);
	// What type of CG should this be... I think hermitian??
	ierr = KSPCGSetType(ksp,KSP_CG_SYMMETRIC); CHKERRQ(ierr);

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


  { // Write output
    vec2tree(sol, phi);
    phi->Write2File("results/sol",0);
  }


	// Some checks!!!!
	Vec eta_vec;
  VecCreateMPI(comm,n,PETSC_DETERMINE,&eta_vec);
	tree2vec(eta,eta_vec);
	phi->RunFMM();
	phi->Copy_FMMOutput();

	tree2vec(phi,sol);
	VecAXPY(sol,-1,rhs);
	PetscReal norm;
	VecNorm(sol,NORM_2,&norm);

	std::cout << "The norm is: " << norm << std::endl;


	eta->SetupFMM(fmm_mat);
	eta->RunFMM();
	eta->Copy_FMMOutput();
	tree2vec(eta,sol);
	VecAXPY(sol,-1,rhs);
	VecNorm(sol,NORM_2,&norm);

	std::cout << "The norm is: " << norm << std::endl;



  // Free work space.  All PETSc objects should be destroyed when they
  // are no longer needed.
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

	return 0;
}
