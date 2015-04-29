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
#include <functional>

// pt source locations
std::vector<double> pt_src_locs;
// random coefficients
std::vector<double> coeffs;
void phi_0_fn(const double* coord, int n, double* out);
void phi_0_fn(const double* coord, int n, double* out)
{
	linear_comb_of_pt_src(coord, n, out, coeffs, pt_src_locs);
}



typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;


void MDM(El::DistMatrix<El::Complex<double>> &A,
		       El::DistMatrix<El::Complex<double>> &b,
					 InvMedTree<FMM_Mat_t> *temp,
					 InvMedTree<FMM_Mat_t> *temp_c
					 ){

	const El::Grid& g = A.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);
	int m = A.Height();
	int n = A.Width();

	for(int i=0;i<m;i++){ // Hadamard product
		El::DistMatrix<El::Complex<double>> A_i = El::View(A, i, 0, 1, n);
		El::DistMatrix<El::Complex<double>> At_i(g);
		El::Transpose(A_i,At_i);
		elemental2tree(At_i,temp);
		elemental2tree(b,temp_c);
		temp->Multiply(temp_c,1);
		tree2elemental(temp,At_i);
		El::Transpose(At_i,A_i);
	}


	//for(int i=0;i<m;i++){ // Hadamard product
	//	El::DistMatrix<El::Complex<double>> A_i = El::View(A, i, 0, 1, n);
	//	El::Hadamard(A_i,b,C);
	//	A_i = C;
	//
	//	}

	return;
}

void recsvd(El::DistMatrix<El::Complex<double>> &U,
		       El::DistMatrix<El::Complex<double>> &S,
	 				 El::DistMatrix<El::Complex<double>> &Vt,
	 				 El::DistMatrix<El::Complex<double>> &A
					 ){
	// We will overwrite the svd factors
	// Transform USV^TA to [USV^Ta_1;USV^Ta_2;...USV^Ta_n]

	const El::Grid& g = U.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);


	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);




	return;

}



template<class F>
void rsvd2(El::DistMatrix<El::Complex<double>> &U,
		       El::DistMatrix<El::Complex<double>> &S,
	 				 El::DistMatrix<El::Complex<double>> &Vt,
					 F A,
					 F At,
					 int m,
					 int n,
					 int r){

	const El::Grid& g = U.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);


	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);

	El::DistMatrix<El::Complex<double>> rw(g);
	El::DistMatrix<El::Complex<double>> Q(g); // the projection of omega through A*
	El::Zeros(Q,m,r);


	for(int i=0;i<r;i++){
		El::Gaussian(rw, n, 1);
		El::DistMatrix<El::Complex<double>> Y_i = El::View(Q, 0, i, m, 1);
		A(rw,Y_i);
	}

	//El::Display(Q,"Q");

	El::DistMatrix<El::Complex<double>> R(g); // the projection of omaga through G*
	El::qr::Explicit(Q, R, El::QRCtrl<double>());

	//El::DistMatrix<El::Complex<double>> e(g);
	//El::Zeros(e,r,r);
	//El::Gemm(El::ADJOINT,El::NORMAL,alpha,Q,Q,beta,e);

	//El::Display(e,"e");

	//std::cout << "Q: " << Q.Height() << " " << Q.Width() << std::endl;

	// Now Q is such that G \approx QQ*G.Thus we can compute GY
	// Compute it's SVD and then multiply by Q, thus giving the approx 
	// SVD of G!!!

	El::DistMatrix<El::Complex<double>> GtQ(g);
	El::Zeros(GtQ,n,r);
	//El::Matrix<El::Complex<double>> GY_i;
	for(int i=0;i<r;i++){
		if(!rank) std::cout << "ChebFMM " << i << std::endl; 
		El::DistMatrix<El::Complex<double>> Q_i = El::View(Q, 0, i, m, 1);
		El::DistMatrix<El::Complex<double>> GtQ_i = El::View(GtQ, 0, i, n, 1);
		At(Q_i,GtQ_i);
		//El::Display(Q_i,"Q_i");
		//El::Display(GQ_i,"GQ_i");
	}


	El::DistMatrix<double> s(g);
	El::Zeros(s,r,1);

	El::DistMatrix<El::Complex<double>> V(g);
	El::Zeros(V,n,r);

	El::DistMatrix<El::Complex<double>> GQ(g);
	El::Adjoint(GtQ,GQ);
	El::SVD(GQ, s, V, El::SVDCtrl<double>());



	//El::DistMatrix<El::Complex<double>> Vt(g);
	El::Adjoint(V,Vt);


	std::vector<double> d(r);
	El::DistMatrix<double,El::STAR,El::STAR> s_star = s;
	d.assign(s_star.Buffer(),s_star.Buffer()+r);


	//El::DistMatrix<El::Complex<double>> Sigma(g);
	El::Zeros(S,r,r);
	El::Diagonal(S, d);

	// G \approx QQ*G = QU\Sigma\V*
	// So We take V* and multiply by Q*
	//El::DistMatrix<El::Complex<double>> U(g);
	El::Zeros(U,m,r);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,Q,GQ,beta,U);


	//El::Display(U,"GQ");
	//El::Display(Sigma,"Sigma");
	//El::Display(Vt,"Vt");


	//U_G = U;
	//S_G = Sigma;
	//Vt_G = Vt;

	return;

}


template<class F>
void
rsvd(El::DistMatrix<El::Complex<double>> &U,
		       El::DistMatrix<El::Complex<double>> &S,
	 				 El::DistMatrix<El::Complex<double>> &Vt,
					 F A,
					 F At,
					 int m,
					 int n,
					 int r)
{

	const El::Grid& g = U.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);

	El::DistMatrix<El::Complex<double>> rw(g);
	El::DistMatrix<El::Complex<double>> Q(g); // the projection of omega through A*
	El::Zeros(Q,n,r);


	for(int i=0;i<r;i++){
		El::Gaussian(rw, m, 1);
		El::DistMatrix<El::Complex<double>> Q_i = El::View(Q, 0, i, n, 1);
		At(rw,Q_i);
	}

	//El::Display(Q,"Q");

	El::DistMatrix<El::Complex<double>> R(g); // the projection of omaga through G*
	El::qr::Explicit(Q, R, El::QRCtrl<double>());
	/*
	{
		El::DistMatrix<El::Complex<double>> e(g);
		El::Zeros(e,r,r);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,Q,Q,beta,e);

		El::Display(e,"e");
	}
	*/


	// Now Y is such that G* \approx QQ*G*.Thus we can compute GQ
	// Compute it's SVD and then multiply by Q*, thus giving the approx 
	// SVD of G!!!

	//El::DistMatrix<El::Complex<double>> GQ(g);
	El::Zeros(U,m,r);
	//El::Matrix<El::Complex<double>> GY_i;
	for(int i=0;i<r;i++){
		El::DistMatrix<El::Complex<double>> Q_i = El::View(Q, 0, i, n, 1);
		El::DistMatrix<El::Complex<double>> U_i = El::View(U, 0, i, m, 1);
		A(Q_i,U_i);
	}


	El::DistMatrix<double> s(g);
	El::Zeros(s,r,1);

	El::DistMatrix<El::Complex<double>> V(g);
	El::Zeros(V,r,r);

	El::SVD(U, s, V, El::SVDCtrl<double>());


	std::vector<double> d(r);
	El::DistMatrix<double,El::STAR,El::STAR> s_star = s;
	d.assign(s_star.Buffer(),s_star.Buffer()+r);


	//El::DistMatrix<El::Complex<double>> Sigma(g);
	El::Zeros(S,r,r);
	El::Diagonal(S, d);


	// G \approx GQQ* = U\Sigma\V*Q*
	// So We take V* and multiply by Q*
	//El::DistMatrix<El::Complex<double>> Vt_tilde(g);
	El::Zeros(Vt,r,n);
	El::Gemm(El::ADJOINT,El::ADJOINT,alpha,V,Q,beta,Vt);

	//U_G = GQ;
	//S_G = Sigma;
	//Vt_G = Vt_tilde;

	return;
}


void svd_test(MPI_Comm &comm, int m, int n, int r){
	// Set up some trees
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	//PetscErrorCode ierr;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// Set the scalars for multiplying in Gemm
	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);

	// Set grid
	El::Grid g(comm,size);


	El::DistMatrix<El::Complex<double>> B1(g);
	El::DistMatrix<El::Complex<double>> B2(g);
	El::DistMatrix<El::Complex<double>> A(g);
	El::DistMatrix<El::Complex<double>> At(g);
	El::Gaussian(B1,m,r);
	El::Gaussian(B2,r,n);
	El::Zeros(A,m,n);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,B1,B2,beta,A);
	El::Adjoint(A,At);

	El::Helmholtz(A, m*n, alpha);
	A.Resize(m,n);
	El::Adjoint(A,At);

	/////////////////////////////////////////////////////////////////
	// Low rank factorization of A
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Low Rank Factorization of G" << std::endl;
		
	using namespace std::placeholders;
	auto A_sf  = std::bind(rsvd_test_func,_1,_2,&A);
	auto At_sf = std::bind(rsvd_test_t_func,_1,_2,&At);

	El::DistMatrix<El::Complex<double>> U(g);
	El::DistMatrix<El::Complex<double>>	S(g);
	El::DistMatrix<El::Complex<double>> Vt(g);

	rsvd(U,S,Vt,A_sf,At_sf,m,n,r);
	//rsvd2(U,S,Vt,A_sf,At_sf,m,n,r);


	{ // Test the A is good

		El::DistMatrix<El::Complex<double>> US(g);
		El::DistMatrix<El::Complex<double>>	USVt(g);

		El::DistMatrix<El::Complex<double>> VSt(g);
		El::DistMatrix<El::Complex<double>>	VStUt(g);

		El::Zeros(US,m,r);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,U,S,beta,US);

		El::Zeros(VSt,n,r);
		El::Gemm(El::ADJOINT,El::ADJOINT,alpha,Vt,S,beta,VSt);

		El::Zeros(USVt,m,n);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,US,Vt,beta,USVt);

		El::Zeros(VStUt,n,m);
		El::Gemm(El::NORMAL,El::ADJOINT,alpha,VSt,U,beta,VStUt);

		//El::Display(USVt,"USVt");
		//El::Display(A,"A");

		El::Axpy(-1.0,A,USVt);
		El::Axpy(-1.0,At,VStUt);

		//El::Display(USVt,"USVt");

		double ndiff = El::TwoNorm(USVt)/El::TwoNorm(A);
		if(!rank) std::cout << "Norm diff: " << ndiff << std::endl;
		ndiff = El::TwoNorm(VStUt)/El::TwoNorm(At);
		if(!rank) std::cout << "Norm diff: " << ndiff << std::endl;

	}
	return;
}

int faims(MPI_Comm &comm, int R_d, int k, int create_number){
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	//PetscErrorCode ierr;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int data_dof = 2;

	//PetscRandom r;
	//PetscRandomCreate(comm,&r);
	//PetscRandomSetSeed(r,time(NULL));
	//PetscRandomSetType(r,PETSCRAND48);

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////

	// pt srcs need to be on all processors
	pt_src_locs = equiplane(create_number,0,0.1);
	//pt_src_locs = {.5,.5,.5};
	if(!rank) std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int N_s = pt_src_locs.size()/3;

	std::vector<double> d_locs;
	if(!rank) d_locs = equiplane(create_number,0,0.9);
	int lN_d = d_locs.size()/3;
	int N_d;
	MPI_Allreduce(&lN_d, &N_d, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number gnereated=" << N_d << std::endl;

	// also needs to be global...
	coeffs.clear();
	coeffs.resize(N_s*data_dof);
	{
		#pragma omp parallel for
		for(int i=0;i<N_s*data_dof;i++){
			coeffs[i] = (i%2 == 0) ? 1 : 0 ;
		}
	}

	//////////////////////////////////////////////////////////////
	// Set up FMM stuff
	//////////////////////////////////////////////////////////////

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = phi_0_fn;
	temp->f_max = 4;


	InvMedTree<FMM_Mat_t> *temp_c = new InvMedTree<FMM_Mat_t>(comm);
	temp_c->bndry = bndry;
	temp_c->kernel = kernel_conj;
	temp_c->fn = zero_fn;
	temp_c->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = mask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->fn = eta_fn;
	eta->f_max = .01;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	temp->SetupFMM(fmm_mat);

	FMM_Mat_t *fmm_mat_c=new FMM_Mat_t;
	fmm_mat_c->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);
	temp_c->SetupFMM(fmm_mat_c);

	std::vector<double> ds = temp_c->ReadVals(d_locs);
	pvfmm::PtFMM_Tree* Gt_tree = temp_c->CreatePtFMMTree(d_locs, ds, kernel_conj);

	// Tree sizes
	int m = temp->m;
	int M = temp->M;
	int n = temp->n;
	int N = temp->N;
	int N_disc = N/2;

	// Set the scalars for multiplying in Gemm
	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);

	if(!rank) std::cout << "alpha " << alpha << std::endl; 
	if(!rank) std::cout << "beta " << beta<< std::endl; 

	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = d_locs;
	g_data.pt_tree = Gt_tree;
	g_data.comm = comm;

	U_data u_data;
	u_data.temp = temp;
	u_data.temp_c = temp_c;
	u_data.mask = mask;
	u_data.src_coord = pt_src_locs;
	u_data.bndry = bndry;
	u_data.kernel=kernel;
	u_data.fn = phi_0_fn;
	u_data.coeffs=&coeffs;
	u_data.comm=comm;

	eta->Write2File("../results/eta",0);

	// Set grid
	El::Grid g(comm,size);

	/////////////////////////////////////////////////////////////////
	// Eta to Elemental Vec
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Perturbation to Elemental Vector" << std::endl;
	El::DistMatrix<El::Complex<double>> EtaVec(g);
	El::Zeros(EtaVec,N_disc,1);
	tree2elemental(eta,EtaVec);

	/////////////////////////////////////////////////////////////////
	// Randomize the Incident Field
	/////////////////////////////////////////////////////////////////
	int R_s = 1;
	if(!rank) std::cout << "Incident Field Randomization" << std::endl;

	using namespace std::placeholders;
	auto U_sf  = std::bind(U_func,_1,_2,&u_data);
	auto Ut_sf = std::bind(Ut_func,_1,_2,&u_data);

	El::DistMatrix<El::Complex<double>> U_U(g);
	El::DistMatrix<El::Complex<double>>	S_U(g);
	El::DistMatrix<El::Complex<double>> Vt_U(g);

	rsvd2(U_U,S_U,Vt_U,U_sf,Ut_sf,N_disc,N_s,R_s);


	El::DistMatrix<El::Complex<double>> US_U(g);
	El::Zeros(US_U,N_disc,R_s);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,U_U,S_U,beta,US_U);
	
	//auto U_rand = U_U;

	temp->Write2File("../results/U_rand",0);

	/////////////////////////////////////////////////////////////////
	// Compute the scattered field
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Scattered Field Computation" << std::endl;

	El::DistMatrix<El::Complex<double>> Phi(g);
	El::Zeros(Phi,N_d,1);
	elemental2tree(US_U,temp);
	temp->Multiply(mask,1);
	temp->Multiply(eta,1);
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();
	temp->Write2File("../results/scattered_field",0);
	std::vector<double> detector_values = temp->ReadVals(d_locs);
	vec2elemental(detector_values,Phi);

	/////////////////////////////////////////////////////////////////
	// Low rank factorization of G
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Low Rank Factorization of G" << std::endl;

	auto G_sf  = std::bind(G_func,_1,_2,&g_data);
	auto Gt_sf = std::bind(Gt_func,_1,_2,&g_data);

	El::DistMatrix<El::Complex<double>> U_G(g);
	El::DistMatrix<El::Complex<double>>	S_G(g);
	El::DistMatrix<El::Complex<double>> Vt_G(g);

	rsvd(U_G,S_G,Vt_G,G_sf,Gt_sf,N_d,N_disc,R_d);

	//El::DistMatrix<El::Complex<double>> GY = U_G;
	//El::DistMatrix<El::Complex<double>>	Sigma = S_G;
	//El::DistMatrix<El::Complex<double>>	Vt_tilde = Vt_G;


	//std::cout << "GY: " << GY.Height() << " " << GY.Width() << std::endl;
	//std::cout << "Sigma: " << Sigma.Height() << " " <<  Sigma.Width() << std::endl;
	//std::cout << "Vt_tilde: " << Vt_tilde.Height() << " " << Vt_tilde.Width() << std::endl;

	{ // Test that G is good
		El::DistMatrix<El::Complex<double>> r(g);
		El::Gaussian(r,N_disc,1);
		elemental2tree(r,temp);
		std::vector<double> filt = {1,1};
		temp->FilterChebTree(filt);
		tree2elemental(temp,r);

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = cs_fn;
		t->f_max = 1;
		t->CreateTree(false);

		MPI_Barrier(comm);

		tree2elemental(t,r);

		El::DistMatrix<El::Complex<double>> e(g);
		El::Zeros(e,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Vt_G,r,beta,e);

		El::DistMatrix<El::Complex<double>> f(g);
		El::Zeros(f,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,S_G,e,beta,f);
		//El::Display(f);
		//El::Display(GY);

		El::DistMatrix<El::Complex<double>> g1(g);
		El::Zeros(g1,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,U_G,f,beta,g1); // GY now actually U

		El::DistMatrix<El::Complex<double>> i(g);
		El::Zeros(i,N_d,1);

		G_func(r,i,&g_data);
		
		El::Axpy(-1.0,i,g1);
		double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
		if(!rank) std::cout << "Norm diff: " << ndiff << std::endl;

		delete t;

	}

	/////////////////////////////////////////////////////////////////
	// Transform phi
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Transform the Scattered Field" << std::endl;

	El::DistMatrix<El::Complex<double>> Phi_hat(g);
	El::Zeros(Phi_hat,R_d,1);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,Phi,beta,Phi_hat);


	/////////////////////////////////////////////////////////////////
	// Compute the SVD of S_G Vt_G DW
	/////////////////////////////////////////////////////////////////
	//
	auto B_sf  = std::bind(B_func,_1,_2,&S_G,&Vt_G,&US_U,temp,temp_c);
	auto Bt_sf = std::bind(Bt_func,_1,_2,&S_G,&Vt_G,&US_U,temp,temp_c);

	El::DistMatrix<El::Complex<double>> U_B(g);
	El::DistMatrix<El::Complex<double>>	S_B(g);
	El::DistMatrix<El::Complex<double>> Vt_B(g);

	//rsvd2(U_B,S_B,Vt_B,B_sf,Bt_sf,R_s*R_d,N_disc,R_s*R_d);

	std::cout << "U B H: " << U_B.Height() << std::endl;
	std::cout << "U B W: " << U_B.Width() << std::endl;
	std::cout << "S B H: " << S_B.Height() << std::endl;
	std::cout << "S B W: " << S_B.Width() << std::endl;
	std::cout << "Vt B H: " << Vt_B.Height() << std::endl;
	std::cout << "Vt B W: " << Vt_B.Width() << std::endl;


	El::DistMatrix<El::Complex<double>> showme(g);
	El::Zeros(showme,R_s*R_d,1);

	B_sf(US_U,showme);

	El::Display(showme,"Show me");

	

	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_G*(\eta.U_rand)
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Multiply matrix by the incident field" << std::endl;
	
	// U_rand is diagonal N_disc x N_disc. Instead of forming it,
	// we can multiply each row of Vt_tilde pointwise by U_rand
	//El::DistMatrix<El::Complex<double>> Ut_rand(g);
	//El::Zeros(Ut_rand,1,N_disc);
	//El::Transpose(U_rand,Ut_rand);

	El::DistMatrix<El::Complex<double>> VtU(g);
	El::Zeros(VtU,R_d,N_disc);

	MDM(Vt_G,US_U,temp,temp_c);

	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_tildeU_rand \eta
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Rest of the Multiplcations"<<  std::endl;

	El::DistMatrix<El::Complex<double>> G_eps(g);
	El::Zeros(G_eps,R_d*R_s,N_disc);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,S_G,Vt_G,beta,G_eps);
	//El::DistMatrix<El::Complex<double>> G_eps_part(g);
	//El::Zeros(G_eps_part,R_d*R_s,R_d*R_s);
	//El::Gemm(El::NORMAL,El::NORMAL,alpha,U_B,S_B,beta,G_eps_part);
	//El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps_part,Vt_B,beta,G_eps);

	{// test if G_eps is ok
		El::DistMatrix<El::Complex<double>> r(g);
		El::Gaussian(r,N_disc,1);

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = cs_fn;
		t->f_max = 1;
		t->CreateTree(false);

		//elemental2tree(r,t);
		//std::vector<double> filt = {1};
		//t->FilterChebTree(filt);
		tree2elemental(t,r);

		El::DistMatrix<El::Complex<double>> a(g);
		El::Zeros(a,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps,r,beta,a);

		El::DistMatrix<El::Complex<double>> b(g);
		El::Zeros(b,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,U_G,a,beta,b);


		elemental2tree(US_U,temp);

		t->Multiply(temp,1);
		tree2elemental(t,r);


		El::DistMatrix<El::Complex<double>> c(g);
		El::Zeros(c,N_d,1);
		G_func(r,c,&g_data);
		El::Display(c,"c");
		El::Display(b,"b");


		El::Axpy(-1.0,c,b);
		double rnorm = El::TwoNorm(b)/El::TwoNorm(c);
		if(!rank) std::cout << "G_eps rel norm: " << rnorm << std::endl;

		delete t;

	}

	/////////////////////////////////////////////////////////////////
	// Solve
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Solve"<<  std::endl;
	El::DistMatrix<El::Complex<double>> Eta_recon(g);
	El::Zeros(Eta_recon,N_disc,1);

	El::Complex<double> gamma;
	El::SetRealPart(gamma, 1.5);
	El::SetImagPart(gamma, 0.0);

	// Truncated SVD
	El::DistMatrix<double> s_sol(g);
	El::DistMatrix<El::Complex<double>> V_sol(g);
	El::SVD(G_eps, s_sol, V_sol, El::SVDCtrl<double>());

	{// test SVD
		std::vector<double> d_sol(R_d);
		El::DistMatrix<double,El::STAR,El::STAR> s_sol_star = s_sol;
		d_sol.assign(s_sol_star.Buffer(),s_sol_star.Buffer()+R_d);

		El::DistMatrix<El::Complex<double>> Sigma_sol(g);
		El::Zeros(Sigma_sol,R_d,R_d);
		El::Diagonal(Sigma_sol, d_sol);
		El::Display(Sigma_sol,"d_sol");

		El::DistMatrix<El::Complex<double>> r(g);
		El::Gaussian(r,N_disc,1);

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = cs_fn;
		t->f_max = 1;
		t->CreateTree(false);

		tree2elemental(t,r);


		El::DistMatrix<El::Complex<double>> e(g);
		El::Zeros(e,R_d,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_sol,r,beta,e);

		El::DistMatrix<El::Complex<double>> f(g);
		El::Zeros(f,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Sigma_sol,e,beta,f);

		El::DistMatrix<El::Complex<double>> g1(g);
		El::Zeros(g1,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps,f,beta,g1); // GY now actually U


		El::DistMatrix<El::Complex<double>> i(g);
		El::Zeros(i,N_d,1);
		El::DistMatrix<El::Complex<double>> h(g);
		El::Zeros(h,R_d,1);

		elemental2tree(US_U,temp_c);
		temp_c->Multiply(t,1);
		tree2elemental(temp_c,r);
		G_func(r,i,&g_data);

		// 
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,i,beta,h); // GY now actually U
		
		El::Axpy(-1.0,h,g1);
		double frnorm = El::TwoNorm(g1)/El::TwoNorm(h);
		if(!rank) std::cout << "Final SVD rel norm: " << frnorm << std::endl;

		delete t;
	}


	if(!rank){
		std::cout << "G H " << G_eps.Height() << std::endl;
		std::cout << "G W " << G_eps.Width() << std::endl;
		std::cout << "s H " << s_sol.Height() << std::endl;
		std::cout << "s W " << s_sol.Width() << std::endl;
		std::cout << "V H " << V_sol.Height() << std::endl;
		std::cout << "V W " << V_sol.Width() << std::endl;
	}

	// truncate
	El::DistMatrix<double> s_sol_k = El::View(s_sol,0,0,k,1);
	El::DistMatrix<El::Complex<double>> V_sol_k = El::View(V_sol,0,0,N_disc,k);
	El::DistMatrix<El::Complex<double>> G_eps_k = El::View(G_eps,0,0,R_d,k);

	El::DistMatrix<El::Complex<double>> C(g);
	El::Zeros(C,k,1);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,G_eps_k,Phi_hat,beta,C);

	std::vector<El::Complex<double>> s_sol_k_inv(k); // needs to be complex for Gemm
	for(int i=0;i<k;i++){
		El::Complex<double> v1;
		double v = s_sol_k.Get(i,0);
		v = 1.0/v;
		El::SetRealPart(v1,v);
		El::SetImagPart(v1,0.0);
		s_sol_k_inv[i] = v1;
	}
	El::DistMatrix<El::Complex<double>> Sig_k(g);
	El::Diagonal(Sig_k,s_sol_k_inv);
	El::Display(Sig_k,"Sig_k");

	El::DistMatrix<El::Complex<double>> D(g);
	El::Zeros(D,k,1);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,Sig_k,C,beta,D);

	El::Gemm(El::NORMAL,El::NORMAL,alpha,V_sol_k,D,beta,Eta_recon);
	if(!rank){
		std::cout << "V H " << Eta_recon.Height() << std::endl;
		std::cout << "V W " << Eta_recon.Width() << std::endl;
	}

	

	/////////////////////////////////////////////////////////////////
	// Test
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Test!" << std::endl ;
	elemental2tree(Eta_recon,temp);
	temp->Multiply(mask,1);
	temp->Write2File("../results/sol",0);
	temp->Add(eta,-1);
	double relerr = temp->Norm2()/eta->Norm2();
	if(!rank) std::cout << "Relative Error: " << relerr << std::endl;



	return 0;
}

////////////////////////////////////////////////////
///////////////////////////////////////////////////
/////////////////////////////////////////////////
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
	PetscInt N_pts = 8;
	PetscInt R_d = N_pts/2;
	PetscInt k = R_d /2;

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
  PetscOptionsGetInt (NULL, "-N_pts",&             N_pts  ,NULL); // This number gets squared
  PetscOptionsGetInt (NULL, "-R_d",&             R_d  ,NULL); // Reduced rank of detectors
  PetscOptionsGetInt (NULL, "-k",&             k  ,NULL); // rank for truncated svd

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

	InvMedTree<FMM_Mat_t>::cheb_deg = CHEB_DEG;
	InvMedTree<FMM_Mat_t>::mult_order = MUL_ORDER;
	InvMedTree<FMM_Mat_t>::tol = REF_TOL;
	InvMedTree<FMM_Mat_t>::mindepth = MINDEPTH;
	InvMedTree<FMM_Mat_t>::maxdepth = MAXDEPTH;
	InvMedTree<FMM_Mat_t>::adap = true;
	InvMedTree<FMM_Mat_t>::dim = 3;
	InvMedTree<FMM_Mat_t>::data_dof = 2;

	if(!rank){
		std::cout << "N_pts: " << N_pts << std::endl;
		std::cout << "R_d: " << R_d << std::endl;
		std::cout << "k: " << k << std::endl;
	}

	//faims(comm, R_d, k, N_pts);
	svd_test(comm,10,20,9);
	El::Finalize();
	PetscFinalize();

	return 0;
}
