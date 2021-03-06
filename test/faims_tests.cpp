#include "invmed_tree.hpp"
#include <iostream>
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
#include "ops.hpp"

#define VTK_ORDER 4
std::string SAVE_DIR_STR;

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


int el_test(MPI_Comm &comm){

	int size;
	int rank;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> temp  = InvMedTree<FMM_Mat_t>(prod_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp1 = InvMedTree<FMM_Mat_t>(sc_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp2 = InvMedTree<FMM_Mat_t>(prod_fn,1.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int M = temp.M;
	if(!rank) std::cout << "M: " << M << std::endl;
	//El::Grid g(comm, size);
	El::Grid g(comm);
	//std::vector<double> vec(sz);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> A(g);
	El::Zeros(A,M/2,1); // dividing by data_dof
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> B(g);
	El::Zeros(B,M/2,1); // dividing by data_dof
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> C(g);
	El::Zeros(C,M/2,1); // dividing by data_dof
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> D(g);
	El::Zeros(D,M/2,1); // dividing by data_dof
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> E(g);
	El::Zeros(E,M/2,1); // dividing by data_dof
	tree2elemental(&temp,A);
	//El::Display(A,"A");
	elemental2tree(A,&temp1);
	tree2elemental(&temp1,B);
	elemental2tree(B,&temp2);

	temp.Write2File((SAVE_DIR_STR+"eltestA").c_str(),0);
	temp1.Write2File((SAVE_DIR_STR+"eltestB").c_str(),0);
	temp1.Add(&temp,-1);
	temp1.Write2File((SAVE_DIR_STR+"eltestC").c_str(),0);

	double rel_norm  = temp1.Norm2()/temp.Norm2();
	double t1 = temp1.Norm2();
	double t = temp.Norm2();
	if(!rank) std::cout << t1 << std::endl;
	if(!rank) std::cout << t << std::endl;

	tree2elemental(&temp,D);
	tree2elemental(&temp2,E);
	El::Axpy(-1.0,D,E);
	double other_norm = El::TwoNorm(E) / El::TwoNorm(D);

	El::Write(B,SAVE_DIR_STR+"B",El::ASCII_MATLAB);

	std::string name = __func__;
	test_less(1e-6,rel_norm,name,comm);
	test_less(1e-6,other_norm,name,comm);

	return 0;
}
int el_test2(MPI_Comm &comm){

	int size;
	int rank;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	int gl_fact = size*(size+1)/2;

	int M = 24*gl_fact;
	int m = 24*(rank+1);

	El::Grid g(comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> A(g);
	El::Gaussian(A,M/2,1); // dividing by data_dof

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> A2(g);
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

	int data_dof = 2;

	InvMedTree<FMM_Mat_t> one  = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> zero = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	MPI_Barrier(comm);

	//one->Write2File("one",0);
	one.Zero();
	one.Write2File((SAVE_DIR_STR+"yeszero").c_str(),0);
	zero.Write2File((SAVE_DIR_STR+"startzero").c_str(),0);
	one.Add(&zero,-1);
	one.Write2File((SAVE_DIR_STR+"zero").c_str(),0);

	double abs_err = one.Norm2();

	std::string name = __func__;
	test_less(1e-6,abs_err,name,comm);

	return 0;
}


int init_test(MPI_Comm &comm){

	int myrank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> one(one_fn,1.0,kernel,bndry,comm);

	// initialize the trees
	std::cout << "before" << std::endl;
	InvMedTree<FMM_Mat_t>::SetupInvMed();
	std::cout << "after" << std::endl;

	one.Write2File((SAVE_DIR_STR+"one_im").c_str(),0);

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

	std::vector<double> pt_srcs = unif_point_distrib(8,.25,.75,comm);

	int n_local_pt_srcs = pt_srcs.size()/3;
	int n_pt_srcs;
	MPI_Allreduce(&n_local_pt_srcs,&n_pt_srcs,1,MPI_INT,MPI_SUM,comm);

	InvMedTree<FMM_Mat_t> temp   = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> mask   = InvMedTree<FMM_Mat_t>(cmask_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> sol    = InvMedTree<FMM_Mat_t>(eight_pt_sol_fn,1.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;


	Particle_FMM_op U = Particle_FMM_op(pt_srcs, cmask_fn, kernel, comm);

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(n_pt_srcs,1,g);
	El::Fill(x,El::Complex<double>(1.0)); 

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Ux(g);
	El::Zeros(Ux,M/2,1);
	U(x,Ux);

	elemental2tree(Ux,&temp);

	sol.Multiply(&mask,1);
	temp.Add(&sol,-1);

	double rel_err = temp.Norm2()/sol.Norm2();

	std::string name = __func__;
	test_less(1e-6,rel_err,name,comm);

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

	std::vector<double> src_coord = unif_point_distrib(8, .25, .75, comm);

	InvMedTree<FMM_Mat_t> test  = InvMedTree<FMM_Mat_t>(int_test_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> sol    = InvMedTree<FMM_Mat_t>(int_test_sol_fn,1.0,kernel,bndry,comm);
	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = test.m;
	int M = test.M;
	int n = test.n;
	int N = test.N;

	int n_srcs;
	int n_local_srcs = src_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_srcs,&n_srcs,1,MPI_INT,MPI_SUM,comm);

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g); 
	El::Gaussian(x,n_srcs,1); //sum over the detector size on all procs

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,M/2,1);
	tree2elemental(&test,y);

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Uty(g);
	El::Zeros(Uty,n_srcs,1);

	Volume_FMM_op Ut = Volume_FMM_op(src_coord, one_fn, kernel_conj, bndry, comm);
	Ut(y,Uty);
	std::vector<double> sol_vec = sol.ReadVals(src_coord);

	vec2elemental(sol_vec,x);

	El::Complex<double> alpha = El::Complex<double>(-1.0);
	El::Axpy(alpha,x,Uty);
	double rel_err = El::TwoNorm(Uty)/El::TwoNorm(x);

	std::string name = __func__;
	test_less(1e-6,rel_err,name,comm);

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

	std::vector<double> src_coord = unif_point_distrib(8,.25,.75,comm);
	int n_local_srcs = src_coord.size()/3;
	int n_srcs;
	MPI_Allreduce(&n_local_srcs,&n_srcs,1,MPI_INT,MPI_SUM,comm);

	InvMedTree<FMM_Mat_t> temp   = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp_c = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel_conj,bndry,comm);
	InvMedTree<FMM_Mat_t> temp1  = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> mask   = InvMedTree<FMM_Mat_t>(cmask_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> smooth = InvMedTree<FMM_Mat_t>(sin_fn,1.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;

	Particle_FMM_op U = Particle_FMM_op(src_coord, cmask_fn, kernel, comm);

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g); 
	El::Gaussian(x,n_srcs,1); //sum over the detector size on all procs

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Ux(g);
	El::Zeros(Ux,M/2,1);
	U(x,Ux);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,M/2,1);
	tree2elemental(&smooth,y);

	elemental2tree(y,&temp);
	elemental2tree(Ux,&temp1);
	temp.ConjMultiply(&temp1,1);
	std::vector<double> Uxy = temp.Integrate();

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Uty(g);
	El::Zeros(Uty,n_srcs,1);

	Volume_FMM_op Ut = Volume_FMM_op(src_coord, cmask_fn, kernel_conj, bndry, comm);
	Ut(y,Uty);

	El::Complex<double> xUty = El::Dot(x,Uty);

	std::string name = __func__;
	test_less(1e-6,(fabs(Uxy[0] - El::RealPart(xUty))),name,comm);
	test_less(1e-6,(fabs(Uxy[1] - El::ImagPart(xUty))),name,comm);

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
	InvMedTree<FMM_Mat_t> sol  = InvMedTree<FMM_Mat_t>(int_test_sol_fn,5.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	Volume_FMM_op G = Volume_FMM_op(detector_coord, mask_fn, kernel, bndry, comm);

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Zeros(x,M/2,1);
	tree2elemental(&test,x);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gx(g);
	El::Zeros(Gx,n_detectors,1);
	G(x,Gx);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,n_detectors,1);

	std::vector<double> detector_samples = sol.ReadVals(detector_coord);
	vec2elemental(detector_samples,y);

	El::Complex<double> alpha = El::Complex<double>(-1.0);

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

	std::vector<double> detector_coord = unif_point_distrib(8,.25,.75,comm);

	InvMedTree<FMM_Mat_t> temp  = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> mask  = InvMedTree<FMM_Mat_t>(cmask_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> sol   = InvMedTree<FMM_Mat_t>(eight_pt_sol_fn,1.0,kernel,bndry,comm);
	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	El::Grid g(comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,n_detectors,1);
	El::Fill(y,El::Complex<double>(1.0));

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gty(g);
	El::Zeros(Gty,M/2,1);
	Particle_FMM_op Gt = Particle_FMM_op(detector_coord, cmask_fn, kernel, comm);

	Gt(y,Gty);

	elemental2tree(Gty,&temp);
	temp.Add(&sol,-1);
	temp.Multiply(&mask,1);
	sol.Multiply(&mask,1);
	double rel_norm = temp.Norm2()/sol.Norm2();

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

	std::vector<double> detector_coord = unif_point_distrib(8,.25,.75,comm);

	InvMedTree<FMM_Mat_t> temp   = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp1  = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> smooth = InvMedTree<FMM_Mat_t>(sin_fn,1.0,kernel,bndry,comm);
	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;

	int n_detectors;
	int n_local_detectors = detector_coord.size()/3; // sum of number of detector_coord on each proc
	MPI_Allreduce(&n_local_detectors,&n_detectors,1,MPI_INT,MPI_SUM,comm);

	El::Grid g(comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Zeros(x,M/2,1);
	tree2elemental(&smooth,x);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Gaussian(y,n_detectors,1,El::Complex<double>(1.0),1.0);

	Volume_FMM_op G = Volume_FMM_op(detector_coord, cmask_fn, kernel, bndry, comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gx(g);
	El::Zeros(Gx,n_detectors,1);
	G(x,Gx);

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gty(g);
	El::Zeros(Gty,M/2,1);

	Particle_FMM_op Gt = Particle_FMM_op(detector_coord, cmask_fn, kernel_conj, comm);
	Gt(y,Gty);

	elemental2tree(x,&temp);
	elemental2tree(Gty,&temp1);
	temp.ConjMultiply(&temp1,1);

	std::vector<double> xGty = temp.Integrate();

	El::Complex<double> Gxy = El::Dot(y,Gx);

	std::string name = __func__;
	test_less(1e-6,(fabs(xGty[0] - El::RealPart(Gxy))),name,comm);
	test_less(1e-6,(fabs(xGty[1] - El::ImagPart(Gxy))),name,comm);

	return 0;
}

int orthogonality_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	El::Grid g(comm);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	InvMedTree<FMM_Mat_t> s = InvMedTree<FMM_Mat_t>(sin2pix_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> c = InvMedTree<FMM_Mat_t>(cos2pix_fn,1.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = s.m;
	int M = s.M;
	int n = s.n;
	int N = s.N;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> sv(g);
	El::Zeros(sv,M/2,1);
	tree2elemental(&s,sv);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> cv(g);
	El::Zeros(cv,M/2,1);
	tree2elemental(&c,cv);

	s.ConjMultiply(&c,1);

	std::vector<double> vec = s.Integrate();

	std::string name = __func__;
	test_less(1e-6,fabs(vec[0]),name,comm);
	test_less(1e-6,fabs(vec[1]),name,comm);

	El::Complex<double> cts = El::Dot(cv,sv);

	std::cout << "cts: " << cts << std::endl;

	return 0;
}

/*
int BfuncBtfunc_test(MPI_Comm &comm){

	int rank, size;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	InvMedTree<FMM_Mat_t> temp   = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp1  = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> smooth = InvMedTree<FMM_Mat_t>(prod_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp2  = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;

	int N_disc = M/2;
	int R_d = 10;
	int R_s = 10;

	El::Grid g(comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> S_G(g);
	El::Gaussian(S_G,R_d,1);
	//El::Fill(S_G,El::Complex<double>(1));

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G(g);
	El::Gaussian(V_G,N_disc,R_d);
	//El::Fill(Vt_G,El::Complex<double>(1));
	//El::Identity(Vt_G,R_d,N_disc);
	std::vector<double> fvec = {1};
	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> W_i = El::View(V_G, 0, i, N_disc, 1);
		elemental2tree(W_i,&temp);
		temp.FilterChebTree(fvec);
		tree2elemental(&temp,W_i);
	}
	//El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_G(g);
	//El::Adjoint(V_G,Vt_G);


	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> US_U(g);
	El::Gaussian(US_U,N_disc,R_s);
	//	El::Zeros(US_U,N_disc,R_s);
	//	El::Fill(US_U,El::Complex<double>(1));

	// filter the one of the inputs..
	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> W_i = El::View(US_U, 0, i, N_disc, 1);
		elemental2tree(W_i,&temp);
		temp.FilterChebTree(fvec);
		tree2elemental(&temp,W_i);
	}

	// create and then filter input vector
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Gaussian(x,N_disc,1);
	//elemental2tree(x,temp);
	//temp->FilterChebTree(fvec);
	tree2elemental(&smooth,x);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Gaussian(y,R_s*R_d,1);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Bx(g);
	El::Zeros(Bx,R_s*R_d,1);

	using namespace std::placeholders;
	auto B_sf  = std::bind(B_func,_1,_2,&S_G,&V_G,&US_U,&temp,&temp1);
	auto Bt_sf = std::bind(Bt_func,_1,_2,&S_G,&V_G,&US_U,&temp,&temp1,&temp2);

	B_sf(x,Bx);

	// Now do it the other way
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Bty(g);
	El::Zeros(Bty,N_disc,1);

	Bt_sf(y,Bty);

	elemental2tree(x,&temp);
	elemental2tree(Bty,&temp1);
	temp.ConjMultiply(&temp1,1);

	std::vector<double> xBty = temp.Integrate();

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
*/

int grsvd_test(MPI_Comm &comm){
	int N_d_sugg = 100;
	int N_s_sugg = 100;
	int R_d = 25;
	int R_s = 25;
	int R_b = 25;
	int k = 25;
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel_low;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj_low;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	double rsvd_tol = 0.0001;

	int data_dof = 2;

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////

	// pt srcs need to be on all processors
	/* switch to using fmm instead of direct evaluation
		 pt_src_locs = equiplane(N_s_sugg,0,0.1);
	//pt_src_locs = {.5,.5,.5};
	if(!rank) std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int N_s = pt_src_locs.size()/3;
	*/

	std::vector<double> pt_srcs = unif_plane(N_s_sugg,0,0.1,comm);

	int lN_s = pt_srcs.size()/3;
	int N_s;
	MPI_Allreduce(&lN_s,&N_s,1,MPI_INT,MPI_SUM,comm);
	if(!rank) std::cout << "Number of sources generated=" << N_s << std::endl;


	std::vector<double> d_locs = unif_plane(N_d_sugg, 0, 0.9, comm);

	int lN_d = d_locs.size()/3;
	int N_d;
	MPI_Allreduce(&lN_d, &N_d, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of detectors generated=" << N_d << std::endl;

	// also needs to be global...

	//////////////////////////////////////////////////////////////
	// File name prefix
	//////////////////////////////////////////////////////////////
	std::string params = "-" + std::to_string((long long)(InvMedTree<FMM_Mat_t>::maxdepth)) + "-" + std::to_string((long long)(N_d)) + "-" + std::to_string((long long)(N_s)) + "-" + std::to_string((long long)(R_d)) + "-" + std::to_string((long long)(R_s)) + "-" + std::to_string((long long)(R_b)) + "-" + std::to_string((long long)(k));
	std::cout << "Params " << params << std::endl;


	//////////////////////////////////////////////////////////////
	// Set up FMM stuff
	//////////////////////////////////////////////////////////////

	InvMedTree<FMM_Mat_t> temp   = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm); //phi_0_fn??
	InvMedTree<FMM_Mat_t> temp2  = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp_c = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel_conj,bndry,comm);
	InvMedTree<FMM_Mat_t> mask   = InvMedTree<FMM_Mat_t>(mask_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> eta    = InvMedTree<FMM_Mat_t>(eta_smooth_fn,0.01,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	temp.FMMSetup();

	// Tree sizes
	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;
	int N_disc = N/2;

	// Set the scalars for multiplying in Gemm
	auto alpha = El::Complex<double>(1.0);
	auto beta = El::Complex<double>(0.0);

	// Set grid
	El::Grid g(comm);

	/////////////////////////////////////////////////////////////////
	// Eta to Elemental Vec
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Perturbation to Elemental Vector" << std::endl;
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> EtaVec(g);
	El::Zeros(EtaVec,N_disc,1);
	tree2elemental(&eta,EtaVec);

	/////////////////////////////////////////////////////////////////
	// Randomize the Incident Field
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Incident Field Randomization" << std::endl;

	Particle_FMM_op U = Particle_FMM_op(pt_srcs, mask_fn, kernel, comm);
	Volume_FMM_op Ut = Volume_FMM_op(pt_srcs, mask_fn, kernel_conj, bndry, comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_U(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR>	S_U(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_U(g);

	rsvd::RSVDCtrl ctrl_U;
	ctrl_U.m=N_disc;
	ctrl_U.n=N_s;
	ctrl_U.r=R_s;
	ctrl_U.l=10;
	ctrl_U.q=0;
	ctrl_U.tol=rsvd_tol;
	ctrl_U.max_sz=N_s;
	ctrl_U.adap=rsvd::ADAP;
	ctrl_U.orientation=rsvd::NORMAL;
	rsvd::rsvd(U_U,S_U,V_U,U,Ut,ctrl_U);
	R_s = ctrl_U.r;
	if(!rank) std::cout << "R_s = " << R_s << std::endl;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> US_U(g);
	El::Zeros(US_U,N_disc,R_s);
	El::DiagonalScale(El::RIGHT,El::NORMAL,S_U,U_U);
	US_U = U_U;


	{// test that U is ok...
		// test a random input using both the analytical and the approximated 
		// operator
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> USVt_U(g);
		El::Zeros(USVt_U,N_disc,N_s);
		El::Gemm(El::NORMAL,El::ADJOINT,alpha,US_U,V_U,beta,USVt_U);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
		El::Gaussian(x,N_s,1);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y_svd(g);
		El::Zeros(y_svd,N_disc,1);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y_ex(g);
		El::Zeros(y_ex,N_disc,1);

		El::Gemm(El::NORMAL,El::NORMAL,alpha,USVt_U,x,beta,y_svd);
		U(x,y_ex);
		elemental2tree(y_ex,&temp);
		//temp->Write2File((SAVE_DIR+STR+"y_ex").c_str(),VTK_ORDER);
		elemental2tree(y_svd,&temp_c);
		//temp_c->Write2File("/work/02370/kwkelly/maverick/files/results/y_svd",VTK_ORDER);
		temp_c.Add(&temp,-1);
		temp_c.Write2File((SAVE_DIR_STR+"U_diff").c_str(),VTK_ORDER);
		//Axpy(-1.0,y_ex,y_svd);
		double ndiff = temp_c.Norm2()/temp.Norm2();

		if(!rank) std::cout << "Incident Field SVD accuracy: " << ndiff << std::endl;
	}

	/////////////////////////////////////////////////////////////////
	// Compute the scattered field
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Scattered Field Computation" << std::endl;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Phi(g);
	El::Zeros(Phi,N_d,R_s);
	for(int i=0;i<R_s;i++){
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> US_U_i = El::LockedView(US_U, 0, i, N_disc, 1);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Phi_i = El::View(Phi, 0, i, N_d, 1);
		elemental2tree(US_U_i,&temp);

		temp.Multiply(&mask,1);
		temp.Multiply(&eta,1);
		temp.ClearFMMData();
		temp.RunFMM();
		temp.Copy_FMMOutput();
		if(i == 0){
			temp.Write2File((SAVE_DIR_STR+"scattered_field"+params).c_str(),VTK_ORDER);
		}
		std::vector<double> detector_values = temp.ReadVals(d_locs);

		vec2elemental(detector_values,Phi_i);
	}

	/////////////////////////////////////////////////////////////////
	// Low rank factorization of G
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Low Rank Factorization of G" << std::endl;

	Volume_FMM_op G = Volume_FMM_op(d_locs, mask_fn, kernel, bndry, comm);
	Particle_FMM_op Gt = Particle_FMM_op(d_locs, mask_fn, kernel_conj, comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR>	S_G(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G(g);

	rsvd::RSVDCtrl ctrl_G;
	ctrl_G.m=N_d;
	ctrl_G.n=N_disc;
	ctrl_G.r=R_d;
	ctrl_G.l=10;
	ctrl_G.q=0;
	ctrl_G.tol=rsvd_tol;
	ctrl_G.max_sz=N_d;
	ctrl_G.adap=rsvd::ADAP;
	ctrl_G.orientation=rsvd::ADJOINT;
	rsvd::rsvd(U_G,S_G,V_G,G,Gt,ctrl_G);
	R_d = ctrl_G.r;
	if(!rank) std::cout << "R_d = " << R_d << std::endl;

	El::Write(U_G,SAVE_DIR_STR+"U_G"+params,El::ASCII_MATLAB);
	El::Write(S_G,SAVE_DIR_STR+"S_G"+params,El::ASCII_MATLAB);
	El::Write(V_G,SAVE_DIR_STR+"V_G"+params,El::ASCII_MATLAB);



	{ // Test that G is good
		// Since G takes a function as an input we can not just randomly generate the Chebyshev coefficients gaussing random. 
		// See comment from the factorization of G_eps for more detail

		/*
		 * See if the ||Gw - USV*w||/||Gw|| is small
		 */

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> r(g);
		El::Zeros(r,N_disc,1);

		InvMedTree<FMM_Mat_t> w = InvMedTree<FMM_Mat_t>(sin_fn,1.0,kernel,bndry,comm);
		w.CreateTree(false);

		MPI_Barrier(comm);

		tree2elemental(&w,r);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> e(g);
		El::Zeros(e,R_d,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,e);
		El::DiagonalScale(El::LEFT,El::NORMAL,S_G,e);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> g1(g);
		El::Zeros(g1,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,U_G,e,beta,g1); // GY now actually U

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> i(g);
		El::Zeros(i,N_d,1);

		G(r,i);
		El::Display(i,"i");
		El::Display(g1,"g1");
		//elemental2tree(i,&temp);
		//temp.Add(&w,-1.0);

		El::Axpy(-1.0,i,g1);
		double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
		if(!rank) std::cout << "||Gw - USV*w|/||Gw|||=" << ndiff << std::endl;

		/*
		 * Test that ||w - VV*w||/||w|| is small so that w can be well represented in the basis of V
		 */
		tree2elemental(&w,r);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gw(g);
		El::Zeros(Vt_Gw,R_d,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vt_Gw);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VVt_Gw(g);
		El::Zeros(VVt_Gw,N_disc,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,V_G,Vt_Gw,beta,VVt_Gw);

		elemental2tree(VVt_Gw,&temp);
		temp.Write2File((SAVE_DIR_STR+"projection").c_str(),VTK_ORDER);
		temp.Add(&w,-1);
		temp.Write2File((SAVE_DIR_STR+"proj_diff").c_str(),VTK_ORDER);

		El::Axpy(-1.0,r,VVt_Gw);
		double coeff_relnorm = El::TwoNorm(VVt_Gw)/El::TwoNorm(r);
		if(!rank) std::cout << "coefficients ||w - VV*w||/||w||=" << coeff_relnorm << std::endl;

		double ls_error = temp.Norm2()/w.Norm2();
		if(!rank) std::cout << "||w - VV*w|| / ||w||=" << ls_error << std::endl;

		elemental2tree(r,&temp);
		temp.Write2File((SAVE_DIR_STR+"w_later").c_str(),VTK_ORDER); // just to see that w is as it was before.

		/*
		 * Take a look at the first and last few vectors that for the basis for V and ensure that the columns are orthogonal
		 */

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_1 = El::LockedView(V_G,0,0,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_2 = El::LockedView(V_G,0,1,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_3 = El::LockedView(V_G,0,2,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_4 = El::LockedView(V_G,0,3,N_disc,1);

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l1 = El::LockedView(V_G,0,R_d-1,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l2 = El::LockedView(V_G,0,R_d-2,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l3 = El::LockedView(V_G,0,R_d-3,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l4 = El::LockedView(V_G,0,R_d-4,N_disc,1);
		elemental2tree(V_G_1,&temp);
		temp.Write2File((SAVE_DIR_STR+"v1").c_str(),VTK_ORDER);
		elemental2tree(V_G_2,&temp);
		temp.Write2File((SAVE_DIR_STR+"v2").c_str(),VTK_ORDER);
		elemental2tree(V_G_3,&temp);
		temp.Write2File((SAVE_DIR_STR+"v3").c_str(),VTK_ORDER);
		elemental2tree(V_G_4,&temp);
		temp.Write2File((SAVE_DIR_STR+"v4").c_str(),VTK_ORDER);
		elemental2tree(V_G_l1,&temp);
		temp.Write2File((SAVE_DIR_STR+"vl1").c_str(),VTK_ORDER);
		elemental2tree(V_G_l2,&temp);
		temp.Write2File((SAVE_DIR_STR+"vl2").c_str(),VTK_ORDER);
		elemental2tree(V_G_l3,&temp);
		temp.Write2File((SAVE_DIR_STR+"vl3").c_str(),VTK_ORDER);
		elemental2tree(V_G_l4,&temp);
		temp.Write2File((SAVE_DIR_STR+"vl4").c_str(),VTK_ORDER);


		// test that the vectors are orhtogonal
		elemental2tree(V_G_1,&temp);
		elemental2tree(V_G_2,&temp2);
		temp.ConjMultiply(&temp2,1);
		std::vector<double> integral = temp.Integrate();
		if(!rank) std::cout << "<V_G_1, V_G_2> = " << integral[0] << "+i" << integral[1] << std::endl;

		elemental2tree(V_G_1,&temp);
		elemental2tree(V_G_3,&temp2);
		temp.ConjMultiply(&temp2,1);
		integral = temp.Integrate();
		if(!rank) std::cout << "<V_G_1, V_G_3> = " << integral[0] << "+i" << integral[1] << std::endl;

		elemental2tree(V_G_1,&temp);
		elemental2tree(V_G_l1,&temp2);
		temp.ConjMultiply(&temp2,1);
		integral = temp.Integrate();
		if(!rank) std::cout << "<V_G_1, V_G_end> = " << integral[0] << "+i" << integral[1] << std::endl;

		/*
		 * Test the orthogonality of the matrices via elemental
		 */

		El::DistMatrix<El::Complex<double>,VR,STAR> I(g);
		El::DistMatrix<El::Complex<double>,VR,STAR> UtU(g);
		El::Zeros(UtU,R_d,R_d);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,U_G,beta,UtU);
		El::Identity(I,R_d,R_d);
		El::Axpy(-1.0,I,UtU);
		double ortho_diff = FrobeniusNorm(UtU)/FrobeniusNorm(I);
		if(!rank) std::cout << "Ortho diff: " << ortho_diff << std::endl;

		El::DistMatrix<El::Complex<double>,VR,STAR> VtV(g);
		El::Zeros(VtV,R_d,R_d);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,V_G,beta,VtV);
		El::Axpy(-1.0,I,VtV);
		ortho_diff = El::FrobeniusNorm(VtV)/El::FrobeniusNorm(I);
		if(!rank) std::cout << "Ortho diff: " << ortho_diff << std::endl;

		/*
		 * Take a look at the spectrum
		 */

		El::Display(S_G,"Sigma_G");

		/*
		 * s_1*||V*w||
		 */

		double sig_1 = El::RealPart(S_G.Get(0,0));
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vtw(g);
		El::Zeros(Vtw,R_d,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vtw);
		double Vtw_norm = El::TwoNorm(Vtw);

		if(!rank) std::cout << "s_1 * ||V*w||=" << sig_1*Vtw_norm << std::endl;

		/*
		 * ||G_v1 - s_1 u_1||
		 */
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gv1(g);
		El::Zeros(Gv1,N_d,1);
		G(V_G_1,Gv1);

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G_1 = El::LockedView(U_G,0,0,N_d,1);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
		El::Zeros(temp_vec,N_d,1);
		El::Axpy(sig_1,U_G_1,temp_vec);
		El::Axpy(-1.0,Gv1,temp_vec);
		double norm_diff1 = El::TwoNorm(temp_vec);

		if(!rank) std::cout << "||Gv_1 - s_1 u_1||=" << norm_diff1 << std::endl;

		/*
		 * G*u_1 - s_1 u_1||
		 */
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gtu1(g);
		El::Zeros(Gtu1,N_disc,1);
		Gt(U_G_1,Gtu1);

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_0 = El::LockedView(V_G,0,0,N_disc,1);
		elemental2tree(Gtu1,&temp);
		temp.Write2File((SAVE_DIR_STR+"Gtu1").c_str(),VTK_ORDER);
		El::Zeros(temp_vec,N_disc,1);
		El::Axpy(sig_1,V_G_0,temp_vec);
		El::Axpy(-1.0,Gtu1,temp_vec);
		double norm_diff2 = El::TwoNorm(temp_vec);

		if(!rank) std::cout << "||Gtu_1 - s_1 v_1||=" << norm_diff2 << std::endl;

		/*
		 * ensure that ||G eta|| <=s1*||V'eta||
		 */

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gw2(g);
		El::Zeros(Gw2,N_d,1);
		G(r,Gw2);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gw2(g);
		El::Zeros(Vt_Gw2,R_d,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vt_Gw2);

		double g_w_norm2 = El::TwoNorm(Gw2);
		double Vt_Gw_norm = El::TwoNorm(Vt_Gw2);
		if(!rank) std::cout << "||Gw|| <= s_1*||V*w||=" << (g_w_norm2 <= sig_1*Vt_Gw_norm) << std::endl;
	}


	/*
	 * No we are on to B and then finally solving everything
	 */

	return 0;
}

int save_mat_test(MPI_Comm &comm){
	int N_d_sugg = 100;
	int N_s_sugg = 100;
	int R_d = 25;
	int R_s = 25;
	int R_b = 25;
	int k = 25;
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel_low;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj_low;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int data_dof = 2;

	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int mult_order = InvMedTree<FMM_Mat_t>::mult_order;

	int depth = InvMedTree<FMM_Mat_t>::maxdepth;

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////

	std::vector<double> pt_srcs = unif_plane(N_s_sugg,0,0.1,comm);

	int lN_s = pt_srcs.size()/3;
	int N_s;
	MPI_Allreduce(&lN_s,&N_s,1,MPI_INT,MPI_SUM,comm);
	if(!rank) std::cout << "Number of sources generated=" << N_s << std::endl;


	std::vector<double> d_locs = unif_plane(N_d_sugg, 0, 0.9, comm);

	int lN_d = d_locs.size()/3;
	int N_d;
	MPI_Allreduce(&lN_d, &N_d, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of detectors generated=" << N_d << std::endl;

	// also needs to be global...

	//////////////////////////////////////////////////////////////
	// File name prefix
	//////////////////////////////////////////////////////////////
	std::string params = "-" + std::to_string((long long)(InvMedTree<FMM_Mat_t>::maxdepth)) + "-" + std::to_string((long long)(N_d)) + "-" + std::to_string((long long)(N_s)) + "-" + std::to_string((long long)(R_d)) + "-" + std::to_string((long long)(R_s)) + "-" + std::to_string((long long)(R_b)) + "-" + std::to_string((long long)(k));
	std::cout << "Params " << params << std::endl;


	//////////////////////////////////////////////////////////////
	// Set up FMM stuff
	//////////////////////////////////////////////////////////////

	InvMedTree<FMM_Mat_t> temp   = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm); //phi_0_fn??
	InvMedTree<FMM_Mat_t> temp2  = InvMedTree<FMM_Mat_t>(one_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> temp_c = InvMedTree<FMM_Mat_t>(zero_fn,0.0,kernel_conj,bndry,comm);
	InvMedTree<FMM_Mat_t> mask   = InvMedTree<FMM_Mat_t>(mask_fn,1.0,kernel,bndry,comm);
	InvMedTree<FMM_Mat_t> eta    = InvMedTree<FMM_Mat_t>(eta_smooth_fn,0.01,kernel,bndry,comm);

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	int m = temp.m;
	int M = temp.M;
	int n = temp.n;
	int N = temp.N;
	int N_disc = N/2;

	// Set the scalars for multiplying in Gemm
	auto alpha = El::Complex<double>(1.0);
	auto beta = El::Complex<double>(0.0);

	El::Grid g(comm);

	/////////////////////////////////////////////////////////////////
	// Create the matrix of G*
	/////////////////////////////////////////////////////////////////

	Volume_FMM_op G = Volume_FMM_op(d_locs, mask_fn, kernel, bndry, comm);
	Particle_FMM_op Gt = Particle_FMM_op(d_locs, mask_fn, kernel, comm);

	// need to loop through each of the detectors and get the column vector associated to it.
	El::DistMatrix<El::Complex<double>,El::VC, El::STAR> eye(g);
	eye.Resize(N_d,1);

	El::DistMatrix<El::Complex<double>,El::VC, El::STAR> Gs_matrix(g);
	Gs_matrix.Resize(N_disc,N_d);

	for(int i=0;i<N_d;i++){
		El::Zeros(eye,N_d,1);
		eye.Set(i,0,El::Complex<double>(1.0)); // all zeros with a one in the correct spot.
		auto Gs_i = View(Gs_matrix, 0, i, N_disc, 1); // get the ith column of the G_matrix
		Gt(eye,Gs_i);
	}


	El::DistMatrix<El::Complex<double>,El::VC, El::STAR> G_matrix(g);
	G_matrix.Resize(N_d,N_disc);

	for(int i=0;i<N_disc;i++){
		El::Zeros(eye,N_disc,1);
		eye.Set(i,0,El::Complex<double>(1.0)); // all zeros with a one in the correct spot.
		auto G_i = View(G_matrix, 0, i, N_d, 1); // get the ith column of the G_matrix
		G(eye,G_i);
	}

	El::Write(G_matrix,SAVE_DIR_STR+"G_matrix"+"_"+std::to_string((long long)(depth))+"_"+std::to_string((long long)(cheb_deg))+"_"+std::to_string((long long)(mult_order)),El::ASCII_MATLAB);
	El::Write(Gs_matrix,SAVE_DIR_STR+"Gs_matrix"+"_"+std::to_string((long long)(depth))+"_"+std::to_string((long long)(cheb_deg))+"_"+std::to_string((long long)(mult_order)),El::ASCII_MATLAB);

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
												-dir        <String> Direcory for saming matrices to\n\
												";

	El::Initialize( argc, argv );

	MPI_Comm comm=MPI_COMM_WORLD;
	int    rank,size;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&size);


	//pvfmm::Profile::Enable(true);


	///////////////////////////////////////////////
	// SETUP
	//////////////////////////////////////////////

	//typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > MPI_Node_t;
	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

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
	init_test(comm);          MPI_Barrier(comm);
	orthogonality_test(comm); MPI_Barrier(comm);
	el_test(comm);            MPI_Barrier(comm);
	el_test2(comm);           MPI_Barrier(comm);
	Zero_test(comm);          MPI_Barrier(comm);
	Gfunc_test(comm);         MPI_Barrier(comm);
	Gtfunc_test(comm);        MPI_Barrier(comm);
	GfuncGtfunc_test(comm);   MPI_Barrier(comm);
	Ufunc_test(comm);        MPI_Barrier(comm);
	Utfunc_test(comm);        MPI_Barrier(comm);
	UfuncUtfunc_test(comm);  MPI_Barrier(comm);
	//BfuncBtfunc_test(comm);   MPI_Barrier(comm);
	grsvd_test(comm);         MPI_Barrier(comm);
	save_mat_test(comm);      MPI_Barrier(comm);

	El::Finalize();

	return 0;
}
