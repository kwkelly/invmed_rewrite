#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <pvfmm.hpp>
#include <set>
//#include "petsc_utils.hpp"
#include "typedefs.hpp"
#include "utils.hpp"
#include <mortonid.hpp>


void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k);
void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

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
	///////////////////////////////////////////////
	// SETUP
	//////////////////////////////////////////////

	// Define some stuff!
	typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > MPI_Node_t;
//	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

  const pvfmm::Kernel<double>* kernel=&helm_kernel;
  const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;

  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t>::cheb_deg = CHEB_DEG;
	InvMedTree<FMM_Mat_t>::mult_order = MUL_ORDER;
	InvMedTree<FMM_Mat_t>::tol = TOL;
	InvMedTree<FMM_Mat_t>::mindepth = MINDEPTH;
	InvMedTree<FMM_Mat_t>::maxdepth = MAXDEPTH;
	InvMedTree<FMM_Mat_t>::adap = true;
	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;

	// Define new trees
	
	InvMedTree<FMM_Mat_t> *zero = new InvMedTree<FMM_Mat_t>(comm);	
	zero->bndry = bndry;
	zero->kernel = kernel_conj;
	zero->fn = zero_fn;
	zero->f_max = 0;

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);	
	one->bndry = bndry;
	one->kernel = kernel_conj;
	one->fn = one_fn;
	one->f_max = 1;

	InvMedTree<FMM_Mat_t> *eye = new InvMedTree<FMM_Mat_t>(comm);	
	eye->bndry = bndry;
	eye->kernel = kernel_conj;
	eye->fn = eye_fn;
	eye->f_max = 1;
	
	InvMedTree<FMM_Mat_t> *pt_sources= new InvMedTree<FMM_Mat_t>(comm);	
	pt_sources->bndry = bndry;
	pt_sources->kernel = kernel;
	pt_sources->fn = pt_sources_fn;
	pt_sources->f_max = 16;

	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);	
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->fn = eta_fn;
	eta->f_max = 1;

	InvMedTree<FMM_Mat_t> *sc = new InvMedTree<FMM_Mat_t>(comm);	
	sc->bndry = bndry;
	sc->kernel = kernel;
	sc->fn = sc_fn;
	sc->f_max = 1;

	InvMedTree<FMM_Mat_t> *scc = new InvMedTree<FMM_Mat_t>(comm);	
	scc->bndry = bndry;
	scc->kernel = kernel;
	scc->fn = scc_fn;
	scc->f_max = 1;
//	InvMedTree<FMM_Mat_t> *ctr_pt_sol = new InvMedTree<FMM_Mat_t>(comm);	
//	ctr_pt_sol->bndry = bndry;
//	ctr_pt_sol->kernel = kernel_conj;
//	ctr_pt_sol->fn = ctr_pt_sol_fn;
//	ctr_pt_sol->f_max = 1;
//
//	InvMedTree<FMM_Mat_t> *ctr_pt_sol_i = new InvMedTree<FMM_Mat_t>(comm);	
//	ctr_pt_sol_i->bndry = bndry;
//	ctr_pt_sol_i->kernel = kernel_conj;
//	ctr_pt_sol_i->fn = ctr_pt_sol_i_fn;
//	ctr_pt_sol_i->f_max = 1;
//
//	InvMedTree<FMM_Mat_t> *ctr_pt_sol_prod = new InvMedTree<FMM_Mat_t>(comm);	
//	ctr_pt_sol_prod->bndry = bndry;
//	ctr_pt_sol_prod->kernel = kernel_conj;
//	ctr_pt_sol_prod->fn = ctr_pt_sol_prod_fn;
//	ctr_pt_sol_prod->f_max = 1;


	// Initialize the trees. We do this all at once so that they have the same
	// structure so that we may add them.
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	zero->Write2File("results/zero",0);
	//eta->Write2File("results/eta",0);
	one->Write2File("results/one",0);
	sc->Write2File("results/sc",0);
	scc->Write2File("results/scc",0);
	//pt_sources->Write2File("results/pt_sources",0);
	//ctr_pt_sol->Write2File("results/ctr_pt_sol",0);
	//ctr_pt_sol_i->Write2File("results/ctr_pt_sol_i",0);
	//ctr_pt_sol_prod->Write2File("results/ctr_pt_sol_prod",0);


	///////////////////////////////////////////////
	// TESTS
	//////////////////////////////////////////////
	
	// Test for particle tree
/*	
	{
		// Here we extract a value from another tree (should give the value 1)
		// then create a particle fmm tree using as evaluation points the Chebyshev
		// node points in the InvMedTree. The source point is directly in the middle.
		// We use here the conjugate of the Helmholtz kernel centered at (.5,.5,.5). 
		// After evaluating we compare to the analytical solution (just the conjugate Helmholtz
		// kernel function) and then examine the difference between the two.
		std::vector<double> src_coord;
		src_coord.push_back(0.5);
		src_coord.push_back(0.5);
		src_coord.push_back(0.5);

		// Has value of one at the center now.
		std::vector<double> val = one->ReadVals(src_coord);
		pvfmm::PtFMM_Tree* pt_tree = pt_sources->CreatePtFMMTree(src_coord, val, kernel_conj);
		std::vector<double> trg_value;
		pvfmm::PtFMM_Evaluate(pt_tree, trg_value);

		std::cout << "After evaluate" << std::endl;
		zero->Trg2Tree(trg_value);
		std::cout << "After trg2tree" << std::endl;
		zero->Write2File("results/pt_eval",0);
		std::cout << "After write" << std::endl;
		// Test the difference
		zero->Add(ctr_pt_sol,-1);
		zero->Write2File("results/pt_eval_diff",0);
		std::cout << "After write diff" << std::endl;

		//  Now we do the same test, but we set the source value to i*1 centered at the middle.
		val[0] = 0;
		val[1] = 1;
		InvMedTree<FMM_Mat_t>::SetSrcValues(src_coord,val,pt_tree);
		pvfmm::PtFMM_Evaluate(pt_tree, trg_value);
		zero->Trg2Tree(trg_value);
		zero->Write2File("results/pt_eval_i",0);
		zero->Add(ctr_pt_sol,-1);
		zero->Write2File("results/pt_eval_i_diff",0);
	
		// Reset the trg value.
		zero->Trg2Tree(trg_value);

	}
	*/

	// Test for Multiply, Add, ConjMultiply
	/*
	{
		eye->Add(one,1);
		eye->Write2File("results/sum",0);
		eye->Multiply(ctr_pt_sol,1);
		eye->Write2File("results/times",0);
		eye->Add(ctr_pt_sol_i,-1);
		eye->Add(ctr_pt_sol,-1);
		eye->Write2File("results/diff",0);

	}
*/
	// Test for Multiply, Add, ConjMultiply
	{
		sc->Multiply(scc,1);
		sc->Write2File("results/times",0);
		sc->Add(one,-1);
		sc->Write2File("results/diff",0);

	}
	
	//std::vector<MPI_Node_t*> nlist=pt_tree->GetNodeList();
	//std::cout << "n_list size" << nlist.size() << std::endl;
	//std::vector<pvfmm::MortonId> mins=pt_tree->GetMins();
//	for(int i=0;i<mins.size();i++){
//		std::cout << mins[i] << std::endl;
//	}
//	for(int i=0;i<nlist.size();i++){
//		pvfmm::Vector<double> *sv = &(nlist[i]->src_value);
//		pvfmm::Vector<double> *sc = &(nlist[i]->src_coord);
//		if(sv->Capacity() >0){
//			std::cout << i << " : " << (sc[0])[0] << std::endl;
//			std::cout << (nlist[i]->Coord())[0] << std::endl;
//		}
//	}
//	int n_trg = 1;

	//std::vector<double> cheb_node_coord3=pvfmm::cheb_nodes<double>(cheb_deg, 3);
	//int n_chebnodes3=cheb_node_coord3.size();
	//int n_octnodes = nlist.size();
	//int n_trg = n_chebnodes3*n_octnodes;
//	for(int i=0;i<trg_value.size();i++){
		//std::cout << trg_value[i] << std::endl;
//	}
	

	//delete r_node;
	
//	FMM_Mat_t* fmm_mat = new FMM_Mat_t;
//	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
//	pt_sources->SetupFMM(fmm_mat);
//	pt_sources->RunFMM();
//	pt_sources->Copy_FMMOutput();
//	std::vector<double> src_values = pt_sources->ReadVals(src_coord);
//	InvMedTree<FMM_Mat_t>::SetSrcValues(src_coord,src_values,pt_tree);
//	pt_sources->Write2File("results/pt_after_fmm",0);


	//pvfmm::Vector<double> srcs = 

	// Need to put the values back into the tree. In this case we will put them into a different tree as part of the test.
	//
/*
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
*/
  // -------------------------------------------------------------------
  // Set up the linear system
  // -------------------------------------------------------------------
	// Set up the operator for CG;
	/*
	PetscInt m = pt_sources->m;
	PetscInt M = pt_sources->M;
	PetscInt n = pt_sources->n;
	PetscInt N = pt_sources->N;
	Mat A;
	InvMedData invmed_data
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
	*/
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
//    vec2tree(eta_vec, eta);
//    eta->Write2File("results/sol",0);
  }

  // Free work space.  All PETSc objects should be destroyed when they
  // are no longer needed.
  //ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  //ierr = VecDestroy(&sol);CHKERRQ(ierr);
  //ierr = VecDestroy(&rhs);CHKERRQ(ierr);
//  ierr = MatDestroy(&A);CHKERRQ(ierr);

	return 0;
}
