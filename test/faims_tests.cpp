#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <pvfmm.hpp>
#include <set>
#include "typedefs.hpp"
#include "invmed_utils.hpp"
#include <mortonid.hpp>
#include <ctime>
#include <string>
#include <random>
#include "El.hpp"

// pt source locations
std::vector<double> pt_src_locs;
// random coefficients
std::vector<double> coeffs;
void phi_0_fn(const double* coord, int n, double* out);
void phi_0_fn(const double* coord, int n, double* out)
{
	linear_comb_of_pt_src(coord, n, out, coeffs, pt_src_locs);
}

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

#undef __FUNCT__
#define __FUNCT__ "el_test"
int el_test(MPI_Comm &comm){

	int size;
	int rank;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = sc_osc_fn;
	temp->f_max = 1;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = zero_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp2 = new InvMedTree<FMM_Mat_t>(comm);
	temp2->bndry = bndry;
	temp2->kernel = kernel;
	temp2->fn = zero_fn;
	temp2->f_max = 0;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int M = temp->M;

	El::Grid g(comm, size);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> A(g);
	El::Zeros(A,M/2,1); // dividing by data_dof

	tree2elemental(temp,A);
	//El::Display(A,"A");
	elemental2tree(A,temp1);

	temp->Write2File("../results/eltestA",0);
	temp1->Write2File("../results/eltestB",0);
	temp1->Add(temp,-1);
	temp1->Write2File("../results/eltestC",0);

	double rel_norm  = temp1->Norm2()/temp->Norm2();
	std::cout << temp1->Norm2() << std::endl;

	std::string name = __func__;
	test_less(1e-6,rel_norm,name,comm);

	delete temp;
	return 0;

}

#undef __FUNCT__
#define __FUNCT__ "el_test2"
int el_test2(MPI_Comm &comm){

	int size;
	int rank;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	int gl_fact = size*(size+1)/2;

	int M = 24*gl_fact;
	int m = 24*(rank+1);

	El::Grid g(comm, size);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> A(g);
	El::Gaussian(A,M/2,1); // dividing by data_dof

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> A2(g);
	El::Zeros(A2,M/2,1); // dividing by data_dof

	std::vector<double> vec(m);

	elemental2vec(A,vec);

	//El::Display(A,"A");
	vec2elemental(vec,A2);
	El::Axpy(-1.0,A,A2);

	double rel_norm  = El::TwoNorm(A2)/El::TwoNorm(A);

	std::string name = __func__;
	test_less(1e-6,rel_norm,name,comm);

	/// Now for the Star Star distributed vectors

	int N = 50;
	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> B(g);
	El::Gaussian(B,N/2,1); // dividing by data_dof

	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> B2(g);
	El::Zeros(B2,N/2,1); // dividing by data_dof

	std::vector<double> vec2(N);
	elstar2vec(B,vec2);
	vec2elstar(vec2,B2);

	El::Axpy(-1.0,B,B2);

	rel_norm  = El::TwoNorm(B2)/El::TwoNorm(B);

	test_less(1e-6,rel_norm,name,comm);


	return 0;

}


int Zero_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = one_fn;
	one->f_max = 1;

	InvMedTree<FMM_Mat_t> *zero = new InvMedTree<FMM_Mat_t>(comm);
	zero->bndry = bndry;
	zero->kernel = kernel;
	zero->fn = zero_fn;
	zero->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	one->Zero();
	one->Add(zero,-1);

	double abs_err = one->Norm2();

	std::string name = __func__;
	test_less(1e-6,abs_err,name,comm);

	delete temp;
	delete zero;
	delete one;

	return 0;

}




int Ufunc2_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	std::vector<double> pt_srcs = unif_point_distrib(8,.25,.75,comm);

	int n_local_pt_srcs = pt_srcs.size()/3;
	int n_pt_srcs;
	MPI_Allreduce(&n_local_pt_srcs,&n_pt_srcs,1,MPI_INT,MPI_SUM,comm);

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp_c = new InvMedTree<FMM_Mat_t>(comm);
	temp_c->bndry = bndry;
	temp_c->kernel = kernel_conj;
	temp_c->fn = zero_fn;
	temp_c->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = zero_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = eight_pt_sol_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	//fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	//temp->SetupFMM(fmm_mat);
	int mult_order = InvMedTree<FMM_Mat_t>::mult_order;
	std::vector<double> src_vals = temp->ReadVals(pt_srcs);
	std::vector<double> trg_coord = temp->ChebPoints();
	int trg_coord_size = trg_coord.size();
	
	pvfmm::PtFMM_Tree* pt_tree=pvfmm::PtFMM_CreateTree(pt_srcs, src_vals, trg_coord, comm );
	// Load matrices.
	pvfmm::PtFMM* matrices = new pvfmm::PtFMM;
	matrices->Initialize(mult_order, comm, kernel);

	// FMM Setup
	pt_tree->SetupFMM(matrices);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	U_data u_data;
	u_data.temp = temp;
	u_data.temp_c = temp_c;
	u_data.mask = mask;
	//u_data.src_coord = detector_coord;
	u_data.n_local_pt_srcs=n_local_pt_srcs;
	u_data.bndry = bndry;
	u_data.kernel=kernel;
	//u_data.fn = phi_0_fn;
	//u_data.coeffs=&coeffs;
	u_data.comm=comm;
	u_data.pt_tree=pt_tree;
	u_data.trg_coord_size=trg_coord_size/3;

	El::Grid g(comm, size);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(n_pt_srcs,1,g);
	El::Fill(x,El::Complex<double>(1.0)); 
	El::Display(x);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Ux(g);
	El::Zeros(Ux,M/2,1);
	U_func2(x,Ux,&u_data);

	elemental2tree(Ux,temp);
	sol->Multiply(mask,1);
	temp->Add(sol,-1);

	double rel_err = temp->Norm2()/sol->Norm2();

	std::string name = __func__;
	test_less(1e-6,rel_err,name,comm);

	delete matrices;
	delete temp;
	delete temp_c;
	delete temp1;
	delete mask;
	delete pt_tree;
	delete sol;

	return 0;

}


int Ufunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	pt_src_locs = equicube(2,.25,.75);
	if(!rank){
		for(int i=0;i<pt_src_locs.size();i++){
			std::cout <<  pt_src_locs[i] << std::endl;
		}
	}
	MPI_Barrier(comm);
	MPI_Barrier(comm);

	std::vector<double> detector_coord;
	if(!rank){
		detector_coord = pt_src_locs;
	}

	int n_pt_srcs = pt_src_locs.size()/3;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp_c = new InvMedTree<FMM_Mat_t>(comm);
	temp_c->bndry = bndry;
	temp_c->kernel = kernel_conj;
	temp_c->fn = zero_fn;
	temp_c->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = zero_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = eight_pt_sol_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	//fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	//temp->SetupFMM(fmm_mat);

	FMM_Mat_t *fmm_mat_c=new FMM_Mat_t;
	fmm_mat_c->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);
	temp_c->SetupFMM(fmm_mat_c);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	U_data u_data;
	u_data.temp = temp;
	u_data.temp_c = temp_c;
	u_data.mask = mask;
	u_data.src_coord = detector_coord;
	u_data.bndry = bndry;
	u_data.kernel=kernel;
	u_data.fn = phi_0_fn;
	u_data.coeffs=&coeffs;
	u_data.comm=comm;

	std::vector<double> input;
	{
		input.clear();
		input.resize(n_detectors*data_dof);
		#pragma omp parallel for
		for(int i=0;i<n_detectors*data_dof;i++){
			input[i] = (i%2 == 0) ? 1 : 0; // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
		}
	}

	{
		coeffs.clear();
		coeffs.resize(n_detectors*data_dof);
		#pragma omp parallel for
		for(int i=0;i<n_detectors*data_dof;i++){
			coeffs[i] = 1; // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
		}
	}

	El::Grid g(comm, size);
	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> x(g);
	El::Zeros(x,n_detectors,1); 
	vec2elstar(input,x);


	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x1(g);
	El::Zeros(x1,n_detectors,1); 
	x1 = x;

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Ux(g);
	El::Zeros(Ux,M/2,1);
	U_func(x1,Ux,&u_data);

	elemental2tree(Ux,temp);
	sol->Multiply(mask,1);
	temp->Add(sol,-1);

	double rel_err = temp->Norm2()/sol->Norm2();

	std::string name = __func__;
	test_less(1e-6,rel_err,name,comm);

	delete temp;
	delete temp_c;
	delete temp1;
	delete mask;
	delete sol;

	return 0;

}

int Utfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	pt_src_locs = equicube(2,.25,.75);

	std::vector<double> detector_coord;
	if(!rank){
		detector_coord = pt_src_locs;
		
	}

	int n_pt_srcs = pt_src_locs.size()/3;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp_c = new InvMedTree<FMM_Mat_t>(comm);
	temp_c->bndry = bndry;
	temp_c->kernel = kernel_conj;
	temp_c->fn = zero_fn;
	temp_c->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = int_test_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = one_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = int_test_sol_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	// Using the normal kernel, not the conjugate for this test
	FMM_Mat_t *fmm_mat_c=new FMM_Mat_t;
	fmm_mat_c->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	temp_c->SetupFMM(fmm_mat_c);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	U_data u_data;
	u_data.temp = temp;
	u_data.temp_c = temp_c;
	u_data.mask = mask;
	u_data.src_coord = detector_coord;
	u_data.bndry = bndry;
	u_data.kernel=kernel;
	u_data.fn = phi_0_fn;
	u_data.coeffs=&coeffs;
	u_data.comm=comm;


	{
		coeffs.clear();
		coeffs.resize(n_detectors*data_dof);
		#pragma omp parallel for
		for(int i=0;i<n_detectors*data_dof;i++){
			coeffs[i] = 1; // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
		}
	}

	El::Grid g(comm, size);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g); 
	El::Gaussian(x,n_detectors,1); //sum over the detector size on all procs

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Zeros(y,M/2,1);
	tree2elemental(temp1,y);


	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Uty(g);
	El::Zeros(Uty,n_detectors,1);

	Ut_func(y,Uty,&u_data);
	std::vector<double> sol_vec = sol->ReadVals(detector_coord);
	vec2elemental(sol_vec,x);
	El::Display(x);
	El::Display(Uty);

	El::Complex<double> alpha;
	El::SetRealPart(alpha,-1.0);
	El::SetImagPart(alpha,0.0);
	El::Axpy(alpha,x,Uty);
	double rel_err = El::TwoNorm(Uty)/El::TwoNorm(x);

	std::string name = __func__;
	test_less(1e-6,rel_err,name,comm);

	return 0;

}



int Ufunc2Utfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	std::vector<double> pt_srcs = unif_point_distrib(8,.25,.75,comm);

	int n_local_pt_srcs = pt_srcs.size()/3;
	int n_pt_srcs;
	MPI_Allreduce(&n_local_pt_srcs,&n_pt_srcs,1,MPI_INT,MPI_SUM,comm);


	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp_c = new InvMedTree<FMM_Mat_t>(comm);
	temp_c->bndry = bndry;
	temp_c->kernel = kernel_conj;
	temp_c->fn = zero_fn;
	temp_c->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = zero_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	mask->Write2File("../results/mask",0);

	//FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	//fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	//temp->SetupFMM(fmm_mat);

	FMM_Mat_t *fmm_mat_c=new FMM_Mat_t;
	fmm_mat_c->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);
	temp_c->SetupFMM(fmm_mat_c);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	int n_detectors;
	int n_local_detectors = pt_srcs.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);



	int mult_order = InvMedTree<FMM_Mat_t>::mult_order;
	std::vector<double> src_vals = temp->ReadVals(pt_srcs);
	std::vector<double> trg_coord = temp->ChebPoints();
	int trg_coord_size = trg_coord.size();
	
	pvfmm::PtFMM_Tree* pt_tree=pvfmm::PtFMM_CreateTree(pt_srcs, src_vals, trg_coord, comm );
	// Load matrices.
	pvfmm::PtFMM* matrices = new pvfmm::PtFMM;
	matrices->Initialize(mult_order, comm, kernel);

	// FMM Setup
	pt_tree->SetupFMM(matrices);





	U_data u_data;
	u_data.temp = temp;
	u_data.temp_c = temp_c;
	u_data.mask = mask;
	u_data.src_coord = pt_srcs;
	u_data.n_local_pt_srcs=n_local_pt_srcs;
	u_data.bndry = bndry;
	u_data.kernel=kernel;
	//u_data.fn = phi_0_fn;
	//u_data.coeffs=&coeffs;
	u_data.comm=comm;
	u_data.pt_tree=pt_tree;
	u_data.trg_coord_size=trg_coord_size/3;


	El::Grid g(comm, size);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g); 
	El::Gaussian(x,n_detectors,1); //sum over the detector size on all procs

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Gaussian(y,M/2,1);
	elemental2tree(y,temp);
	std::vector<double> filter = {1};
	temp->FilterChebTree(filter);
	tree2elemental(temp,y);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Ux(g);
	El::Zeros(Ux,M/2,1);
	U_func2(x,Ux,&u_data);
	//El::Display(Ux,"Ux");
	//El::Display(y,"y");

	elemental2tree(y,temp);
	elemental2tree(Ux,temp1);
	temp1->Write2File("../results/fromtest",8);
	temp->ConjMultiply(temp1,1);
	std::vector<double> Uxy = temp->Integrate();

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Uty(g);
	El::Zeros(Uty,n_detectors,1);

	Ut_func(y,Uty,&u_data);

	El::Complex<double> xUty = El::Dot(x,Uty);
	El::Display(x);
	El::Display(Uty);


	double d1 = std::min(fabs(Uxy[0]),fabs(El::RealPart(xUty)));
	double d2 = std::min(fabs(Uxy[1]),fabs(El::ImagPart(xUty)));
	std::string name = __func__;
	test_less(1e-6,(fabs(Uxy[0] - El::RealPart(xUty))/d1),name,comm);
	test_less(1e-6,(fabs(Uxy[1] - El::ImagPart(xUty))/d2),name,comm);

	return 0;

}






int UfuncUtfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	pt_src_locs = equicube(2,.25,.75);

	std::vector<double> detector_coord;
	if(!rank){
		detector_coord = pt_src_locs;
		
	}

	int n_pt_srcs = pt_src_locs.size()/3;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp_c = new InvMedTree<FMM_Mat_t>(comm);
	temp_c->bndry = bndry;
	temp_c->kernel = kernel_conj;
	temp_c->fn = zero_fn;
	temp_c->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = zero_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	mask->Write2File("../results/mask",0);

	//FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	//fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	//temp->SetupFMM(fmm_mat);

	FMM_Mat_t *fmm_mat_c=new FMM_Mat_t;
	fmm_mat_c->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);
	temp_c->SetupFMM(fmm_mat_c);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	U_data u_data;
	u_data.temp = temp;
	u_data.temp_c = temp_c;
	u_data.mask = mask;
	u_data.src_coord = detector_coord;
	u_data.bndry = bndry;
	u_data.kernel=kernel;
	u_data.fn = phi_0_fn;
	u_data.coeffs=&coeffs;
	u_data.comm=comm;


	{
		coeffs.clear();
		coeffs.resize(n_detectors*data_dof);
		#pragma omp parallel for
		for(int i=0;i<n_detectors*data_dof;i++){
			coeffs[i] = 1; // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
		}
	}

	El::Grid g(comm, size);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g); 
	El::Gaussian(x,n_detectors,1); //sum over the detector size on all procs

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Gaussian(y,M/2,1);
	elemental2tree(y,temp);
	std::vector<double> filter = {1};
	temp->FilterChebTree(filter);
	tree2elemental(temp,y);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Ux(g);
	El::Zeros(Ux,M/2,1);
	U_func(x,Ux,&u_data);
	//El::Display(Ux,"Ux");
	//El::Display(y,"y");

	elemental2tree(y,temp);
	elemental2tree(Ux,temp1);
	temp->ConjMultiply(temp1,1);
	std::vector<double> Uxy = temp->Integrate();

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Uty(g);
	El::Zeros(Uty,n_detectors,1);

	Ut_func(y,Uty,&u_data);

	El::Complex<double> xUty = El::Dot(x,Uty);
	El::Display(x);
	El::Display(Uty);


	double d1 = std::min(fabs(Uxy[0]),fabs(El::RealPart(xUty)));
	double d2 = std::min(fabs(Uxy[1]),fabs(El::ImagPart(xUty)));
	std::string name = __func__;
	test_less(1e-6,(fabs(Uxy[0] - El::RealPart(xUty))/d1),name,comm);
	test_less(1e-6,(fabs(Uxy[1] - El::ImagPart(xUty))/d2),name,comm);

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
	PetscErrorCode ierr;

	std::vector<double> detector_coord = unif_point_distrib(8,.25,.75,comm);

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *test_fn = new InvMedTree<FMM_Mat_t>(comm);
	test_fn->bndry = bndry;
	test_fn->kernel = kernel;
	test_fn->fn = int_test_fn;
	test_fn->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = int_test_sol_fn;
	sol->f_max = 1;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = one_fn;
	mask->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	mask->Write2File("../results/mask",0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);

	temp->SetupFMM(fmm_mat);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;
	//int n_detectors = detector_coord.size()/3;
	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);


	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = detector_coord;

	El::Grid g(comm,size);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g);
	El::Zeros(x,M/2,1);
	tree2elemental(test_fn,x);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Gx(g);
	El::Zeros(Gx,n_detectors,1);
	G_func(x,Gx,&g_data);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Zeros(y,n_detectors,1);

	std::vector<double> detector_samples = sol->ReadVals(detector_coord); //Not sure exactly what this will do...
	vec2elemental(detector_samples,y);

	El::Complex<double> alpha;
	El::SetRealPart(alpha,-1.0);
	El::SetImagPart(alpha,0.0);

	El::Axpy(alpha,y,Gx);
	double norm_diff = El::TwoNorm(Gx)/El::TwoNorm(y);

	std::string name = __func__;
	test_less(1e-6,norm_diff,name,comm);

	return 0;

}


int Gtfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	std::vector<double> detector_coord = unif_point_distrib(8,.25,.75,comm);

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = one_fn;
	temp->f_max = 1;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = one_fn;
	temp1->f_max = 1;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = eight_pt_sol_fn;
	sol->f_max = 1;
	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	mask->Write2File("../results/mask",0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);

	temp->SetupFMM(fmm_mat);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;
	//int n_detectors = detector_coord.size()/3;
	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	//std::cout << n_local_detectors << std::endl;
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	std::vector<double> detector_samples = temp->ReadVals(detector_coord);
	pvfmm::PtFMM_Tree* Gt_tree = temp->CreatePtFMMTree(detector_coord, detector_samples, kernel);

	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = detector_coord;
	g_data.pt_tree = Gt_tree;
	g_data.comm = comm;

	El::Grid g(comm,size);


	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Zeros(y,n_detectors,1);
	vec2elemental(detector_samples,y);
	//El::Gaussian(y,n_detectors,1);

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Gty(g);
	El::Zeros(Gty,M/2,1);

	Gt_func(y,Gty,&g_data);

	elemental2tree(Gty,temp);
	temp->Add(sol,-1);
	temp->Multiply(mask,1);
	sol->Multiply(mask,1);
	double rel_norm = temp->Norm2()/sol->Norm2();

	std::string name = __func__;
	test_less(1e-6,rel_norm,name,comm);

	return 0;

}


int GfuncGtfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	//std::vector<double> detector_coord = {.5,.5,.5}; //equiplane(1,0,1.0);
	std::vector<double> detector_coord = rand_unif_point_distrib(49,.25,.75,comm);

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = zero_fn;
	temp1->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *smooth = new InvMedTree<FMM_Mat_t>(comm);
	smooth->bndry = bndry;
	smooth->kernel = kernel;
	smooth->fn = prod_fn;
	smooth->f_max = 1;
	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	mask->Write2File("../results/mask",0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);

	temp->SetupFMM(fmm_mat);

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	std::vector<double> detector_samples = temp->ReadVals(detector_coord); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* Gt_tree = temp->CreatePtFMMTree(detector_coord, detector_samples, kernel_conj);

	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = detector_coord;
	g_data.pt_tree = Gt_tree;

	El::Grid g(comm,size);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g);
	//El::Gaussian(x,M/2,1);
	El::Zeros(x,M/2,1);
	//elemental2tree(x,temp);
	//std::vector<double> fvec = {1};
	//temp->FilterChebTree(fvec);
	tree2elemental(smooth,x);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Gaussian(y,n_detectors,1);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Gx(g);
	El::Zeros(Gx,n_detectors,1);
	G_func(x,Gx,&g_data);

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Gty(g);
	El::Zeros(Gty,M/2,1);

	Gt_func(y,Gty,&g_data);

	elemental2tree(x,temp);
	elemental2tree(Gty,temp1);
	temp->ConjMultiply(temp1,1);

	std::vector<double> xGty = temp->Integrate();

	//El::Complex<double> Gxy = El::Dot(x,Gty);
	El::Complex<double> Gxy = El::Dot(y,Gx);

	double d1 = std::min(fabs(xGty[0]),fabs(El::RealPart(Gxy)));
	double d2 = std::min(fabs(xGty[1]),fabs(El::ImagPart(Gxy)));

	std::string name = __func__;
	test_less(1e-6,(fabs(xGty[0] - El::RealPart(Gxy)))/d1,name,comm);
	test_less(1e-6,(fabs(xGty[1] - El::ImagPart(Gxy)))/d2,name,comm);

	return 0;

}

int orthogonality_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	El::Grid g(comm,size);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;


	InvMedTree<FMM_Mat_t> *s = new InvMedTree<FMM_Mat_t>(comm);
	s->bndry = bndry;
	s->kernel = kernel;
	s->fn = sin2pix_fn;
	s->f_max = 1;

	InvMedTree<FMM_Mat_t> *c = new InvMedTree<FMM_Mat_t>(comm);
	c->bndry = bndry;
	c->kernel = kernel;
	c->fn = cos2pix_fn;
	c->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = s->m;
	int M = s->M;
	int n = s->n;
	int N = s->N;

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> sv(g);
	El::Zeros(sv,M/2,1);
	tree2elemental(s,sv);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> cv(g);
	El::Zeros(cv,M/2,1);
	tree2elemental(c,cv);

	s->ConjMultiply(c,1);

	std::vector<double> vec = s->Integrate();

	std::string name = __func__;
	test_less(1e-6,fabs(vec[0]),name,comm);
	test_less(1e-6,fabs(vec[1]),name,comm);


	El::Complex<double> cts = El::Dot(cv,sv);

	std::cout << "cts: " << cts << std::endl;

	return 0;

}


int BfuncBtfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;


	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = one_fn;
	temp->f_max = 1;

	InvMedTree<FMM_Mat_t> *temp1 = new InvMedTree<FMM_Mat_t>(comm);
	temp1->bndry = bndry;
	temp1->kernel = kernel;
	temp1->fn = one_fn;
	temp1->f_max = 1;

	InvMedTree<FMM_Mat_t> *smooth = new InvMedTree<FMM_Mat_t>(comm);
	smooth->bndry = bndry;
	smooth->kernel = kernel;
	smooth->fn = prod_fn;
	smooth->f_max = 1;

	InvMedTree<FMM_Mat_t> *temp2 = new InvMedTree<FMM_Mat_t>(comm);
	temp2->bndry = bndry;
	temp2->kernel = kernel;
	temp2->fn = zero_fn;
	temp2->f_max = 0;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;

	int N_disc = M/2;
	int R_d = 10;
	int R_s = 10;

	El::Grid g(comm,size);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> S_G(g);
	El::Gaussian(S_G,R_d,1);
	//El::Fill(S_G,El::Complex<double>(1));

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> V_G(g);
	El::Gaussian(V_G,N_disc,R_d);
	//El::Fill(Vt_G,El::Complex<double>(1));
	//El::Identity(Vt_G,R_d,N_disc);
	std::vector<double> fvec = {1};
	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> W_i = El::View(V_G, 0, i, N_disc, 1);
		elemental2tree(W_i,temp);
		temp->FilterChebTree(fvec);
		tree2elemental(temp,W_i);
	}
	//El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Vt_G(g);
	//El::Adjoint(V_G,Vt_G);


	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> US_U(g);
	El::Gaussian(US_U,N_disc,R_s);
//	El::Zeros(US_U,N_disc,R_s);
//	El::Fill(US_U,El::Complex<double>(1));

	// filter the one of the inputs..
	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> W_i = El::View(US_U, 0, i, N_disc, 1);
		elemental2tree(W_i,temp);
		temp->FilterChebTree(fvec);
		tree2elemental(temp,W_i);
	}

	// create and then filter input vector
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g);
	El::Gaussian(x,N_disc,1);
	//elemental2tree(x,temp);
	//temp->FilterChebTree(fvec);
	tree2elemental(smooth,x);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y(g);
	El::Gaussian(y,R_s*R_d,1);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Bx(g);
	El::Zeros(Bx,R_s*R_d,1);

	using namespace std::placeholders;
	auto B_sf  = std::bind(B_func,_1,_2,&S_G,&V_G,&US_U,temp,temp1);
	auto Bt_sf = std::bind(Bt_func,_1,_2,&S_G,&V_G,&US_U,temp,temp1,temp2);

	B_sf(x,Bx);

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Bty(g);
	El::Zeros(Bty,N_disc,1);

	Bt_sf(y,Bty);

	elemental2tree(x,temp);
	elemental2tree(Bty,temp1);
	temp->ConjMultiply(temp1,1);

	std::vector<double> xBty = temp->Integrate();

	El::Complex<double> Bxy = El::Dot(y,Bx);
	//El::Display(y,"y");
	//El::Display(Bx,"Bx");

	double d1 = std::min(fabs(xBty[0]),fabs(El::RealPart(Bxy)));
	double d2 = std::min(fabs(xBty[1]),fabs(El::ImagPart(Bxy)));

	//std::cout << xBty[0] << "  " << xBty[1] << std::endl;
	//std::cout << Bxy << std::endl;

	std::string name = __func__;
	test_less(1e-6,(fabs(xBty[0] - El::RealPart(Bxy)))/d1,name,comm);
	test_less(1e-6,(fabs(xBty[1] - El::ImagPart(Bxy)))/d2,name,comm);

	return 0;

}





int main(int argc, char* argv[]){
	static char help[] = "\n\
												-eta        <Real>   Inf norm of \\eta\n\
												-ref_tol    <Real>   Tree refinement tolerance\n\
												-min_depth  <Int>    Minimum tree depth\n\
												-max_depth  <Int>    Maximum tree depth\n\
												-fmm_q      <Int>    Chebyshev polynomial degree\n\
												-fmm_m      <Int>    Multipole order (+ve even integer)\n\
												-tol 			  <Real>   GMRES/CG residual tolerance\n\
												-iter 			<Int>    GMRES/CG maximum iterations\n\
												-obs				<Int>		 0 for partial, 1 for full\n\
												-alpha      <Real>	 Regularization parameter\n\
												";
	PetscInt  VTK_ORDER=0;
	PetscInt  INPUT_DOF=2;
	PetscReal  SCAL_EXP=1.0;
	PetscBool  PERIODIC=PETSC_FALSE;
	PetscBool TREE_ONLY=PETSC_FALSE;
	PetscInt  MAXDEPTH  =MAX_DEPTH;// Maximum tree depth
	PetscInt  MINDEPTH   =1;       // Minimum tree depth
	PetscReal   REF_TOL  =1e-3;    // Tolerance
	//PetscReal GMRES_TOL  =1e-10;   // Fine mesh GMRES tolerance
	PetscReal 		TOL  =1e-10;    	// Fine mesh GMRES/CG tolerance
	PetscInt  CHEB_DEG  =14;       // Fine mesh Cheb. order
	PetscInt MUL_ORDER  =10;       // Fine mesh mult  order
	PetscInt MAX_ITER  =200;
	PetscReal f_max=1;
	PetscReal eta_=1;
	PetscInt OBS = 1;
	PetscReal ALPHA = .001;

  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,0,help);
	El::Initialize( argc, argv );

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
  PetscOptionsGetReal(NULL, "-tol" ,		  			&TOL  ,NULL);
  PetscOptionsGetInt (NULL,   "-fmm_q"   ,& CHEB_DEG  ,NULL);
  PetscOptionsGetInt (NULL,   "-fmm_m"   ,&MUL_ORDER  ,NULL);
  PetscOptionsGetInt (NULL, "-iter",&       MAX_ITER  ,NULL);
  PetscOptionsGetReal(NULL,       "-eta" ,&    eta_   ,NULL);
  PetscOptionsGetInt (NULL, "-obs",&             OBS  ,NULL);
  PetscOptionsGetReal (NULL, "-alpha",&         ALPHA  ,NULL);

	//pvfmm::Profile::Enable(true);

	// Define some stuff!


	///////////////////////////////////////////////
	// SETUP
	//////////////////////////////////////////////

	// Define some stuff!
	typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > MPI_Node_t;
	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

  const pvfmm::Kernel<double>* kernel=&helm_kernel;
  const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;

  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	//std::cout << "MINDEPTH: " << MINDEPTH << std::endl;

	InvMedTree<FMM_Mat_t>::cheb_deg = CHEB_DEG;
	InvMedTree<FMM_Mat_t>::mult_order = MUL_ORDER;
	InvMedTree<FMM_Mat_t>::tol = REF_TOL;
	InvMedTree<FMM_Mat_t>::mindepth = MINDEPTH;
	InvMedTree<FMM_Mat_t>::maxdepth = MAXDEPTH;
	InvMedTree<FMM_Mat_t>::adap = true;
	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;

	// Define new trees


	///////////////////////////////////////////////
	// TESTS
	//////////////////////////////////////////////
	orthogonality_test(comm);
	//el_test(comm);
	//el_test2(comm);
	//Zero_test(comm);
	//Ufunc2Utfunc_test(comm);
	//Ufunc2_test(comm);
	//Utfunc_test(comm);
	//Ufunc_test(comm);
	//UfuncUtfunc_test(comm);
	//UfuncUtfunc_test(comm);
	//BfuncBtfunc_test(comm);
	//Gtfunc_test(comm);
	//Gfunc_test(comm);
	//GfuncGtfunc_test(comm);
	El::Finalize();
	PetscFinalize();

	return 0;
}
