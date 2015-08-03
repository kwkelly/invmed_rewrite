#include "invmed_tree.hpp"
#include <iostream>
//#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <pvfmm.hpp>
#include <set>
#include "typedefs.hpp"
#include <mortonid.hpp>
#include <ctime>
#include <string>
#include <random>
#include "El.hpp"
#include "rsvd.hpp"
#include "point_distribs.hpp"
#include "helm_kernels.hpp"
#include "convert_elemental.hpp"
#include "operators.hpp"
#include "ops.hpp"
//#include "convert_petsc.hpp"


#define VTK_ORDER 4
//char SAVE_DIR[PETSC_MAX_PATH_LEN];
std::string SAVE_DIR_STR;

// pt source locations
//std::vector<double> pt_src_locs;
// random coefficients
//std::vector<double> coeffs;
//void phi_0_fn(const double* coord, int n, double* out);
//void phi_0_fn(const double* coord, int n, double* out)
//{
//	linear_comb_of_pt_src(coord, n, out, coeffs, pt_src_locs);
//}

auto test_function = prod_fn;

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

int test_less(const double &expected, const double &actual, const std::string &name, MPI_Comm &comm){
	int rank;
	MPI_Comm_rank(comm, &rank);
	if(rank == 0){
		if(actual < expected) std::cout << "\033[2;32m" << name << " passed! \033[0m- relative error=" << actual  << " expected=" << expected << std::endl;
		else std::cout << "\033[2;31m FAILURE! - " << name << " failed! \033[0m- relative error=" << actual << " expected=" << expected << std::endl;
	}
	return 0;
}


int Gfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	std::vector<double> detector_coord = unif_point_distrib(8,.25,.75,comm);

	InvMedTree<FMM_Mat_t> temp = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> test = InvMedTree<FMM_Mat_t>(int_test_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> mask = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm); // why not the mask fn?
	InvMedTree<FMM_Mat_t> sol  = InvMedTree<FMM_Mat_t>(int_test_sol_fn,5.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	test.Write2File((SAVE_DIR_STR+"test_fn").c_str(),6);
	mask.Write2File((SAVE_DIR_STR+"mask").c_str(),0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	temp.SetupFMM(fmm_mat);

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;
	//int n_detectors = detector_coord.size()/3;
	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);


	G_data g_data;
	g_data.temp = &temp;
	g_data.mask= &mask;
	g_data.src_coord = detector_coord;
	g_data.comm = comm;

	G_op G = G_op(detector_coord, mask_fn, kernel, bndry, comm);

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Zeros(x,M/2,1);
	tree2elemental(&test,x);
	elemental2tree(x,&temp);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gx(g);
	El::Zeros(Gx,n_detectors,1);
	temp.Write2File((SAVE_DIR_STR+"test_fn2").c_str(),6);
	std::cout << "Norm out" << El::OneNorm(x) << std::endl;
	//G_func(x,Gx,&g_data);
	G(x,Gx);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,n_detectors,1);

	sol.Write2File((SAVE_DIR_STR+"gf_sol").c_str(),6);
	//elemental2tree(Gx,temp);

	std::vector<double> detector_samples = sol.ReadVals(detector_coord); //Not sure exactly what this will do...
	vec2elemental(detector_samples,y);

	El::Complex<double> alpha;
	El::SetRealPart(alpha,-1.0);
	El::SetImagPart(alpha,0.0);
	El::Display(y,"y");
	El::Display(Gx,"Gx");

	El::Axpy(alpha,y,Gx);
	double norm_diff = El::TwoNorm(Gx)/El::TwoNorm(y);

	std::string name = __func__;
	test_less(1e-6,norm_diff,name,comm);

	delete fmm_mat;

	return 0;
}

////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){
	static char help[] = "\n\
												-ref_tol    <Real>   Tree refinement tolerance\n\
												-min_depth  <Int>    Minimum tree depth\n\
												-max_depth  <Int>    Maximum tree depth\n\
												-fmm_q      <Int>    Chebyshev polynomial degree\n\
												-fmm_m      <Int>    Multipole order (+ve even integer)\n\
												";
//	PetscInt  INPUT_DOF=2;
//	PetscReal  SCAL_EXP=1.0;
//	PetscBool  PERIODIC=PETSC_FALSE;
//	PetscBool TREE_ONLY=PETSC_FALSE;
//	PetscInt  MAXDEPTH  =MAX_DEPTH;// Maximum tree depth
//	PetscInt  MINDEPTH   =1;       // Minimum tree depth
//	PetscReal   REF_TOL  =1e-3;    // Tolerance
//	//PetscReal GMRES_TOL  =1e-10;   // Fine mesh GMRES tolerance
//	PetscReal 		TOL  =1e-10;    	// Fine mesh GMRES/CG tolerance
//	PetscInt  CHEB_DEG  =14;       // Fine mesh Cheb. order
//	PetscInt MUL_ORDER  =10;       // Fine mesh mult  order
//	PetscInt MAX_ITER  =200;
//	PetscReal f_max=1;
//	PetscReal eta_=1;
//	PetscInt OBS = 1;
//	PetscReal ALPHA = .001;

	El::Initialize( argc, argv );

  MPI_Comm comm=MPI_COMM_WORLD;
  int    rank,size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);


	//pvfmm::Profile::Enable(true);


	///////////////////////////////////////////////
	// SETUP
	//////////////////////////////////////////////

	// Define some stuff!
	//typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > MPI_Node_t;
	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

  //const pvfmm::Kernel<double>* kernel=&helm_kernel;
  //const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;

  //pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	//std::cout << "MINDEPTH: " << MINDEPTH << std::endl;

	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;
	InvMedTree<FMM_Mat_t>::cheb_deg = El::Input("-fmm_q","Chebyshev degree",6);
	InvMedTree<FMM_Mat_t>::mult_order = El::Input("-fmm_m","Multipole order",10);
	InvMedTree<FMM_Mat_t>::tol = El::Input("-ref_tol","Refinement Tolerance",1e-6);
	InvMedTree<FMM_Mat_t>::mindepth = El::Input("-min_depth","Minimum tree depth",3);
	InvMedTree<FMM_Mat_t>::maxdepth = El::Input("-max_depth","Maximum tree depth",3);
	InvMedTree<FMM_Mat_t>::adap = El::Input("-adap","Adaptivity for tree construction",true);
	SAVE_DIR_STR = El::Input("-dir","Directory for saving the functions and the matrices to",".");
	std::cout << SAVE_DIR_STR << std::endl;

	///////////////////////////////////////////////
	// TESTS
	//////////////////////////////////////////////
	Gfunc_test(comm);         MPI_Barrier(comm);

	El::Finalize();

	return 0;
}
