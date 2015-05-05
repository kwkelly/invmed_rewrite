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
#include "rsvd.hpp"

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


void MDM(El::DistMatrix<El::Complex<double>,El::VR,El::STAR> &A,
		       El::DistMatrix<El::Complex<double>,El::VR,El::STAR> &b,
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
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> A_i = El::View(A, i, 0, 1, n);
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> At_i(g);
		El::Transpose(A_i,At_i);
		elemental2tree(At_i,temp);
		elemental2tree(b,temp_c);
		temp->Multiply(temp_c,1);
		tree2elemental(temp,At_i);
		El::Transpose(At_i,A_i);
	}


	//for(int i=0;i<m;i++){ // Hadamard product
	//	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> A_i = El::View(A, i, 0, 1, n);
	//	El::Hadamard(A_i,b,C);
	//	A_i = C;
	//
	//	}

	return;
}

int faims(MPI_Comm &comm, int N_d_sugg, int N_s_sugg, int R_d, int R_s, int R_b, int k){
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	//PetscErrorCode ierr;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int data_dof = 2;

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////

	// pt srcs need to be on all processors
	pt_src_locs = equiplane(N_s_sugg,0,0.1);
	//pt_src_locs = {.5,.5,.5};
	if(!rank) std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int N_s = pt_src_locs.size()/3;

	//std::vector<double> d_locs;
	//if(!rank) d_locs = equiplane(create_number,0,0.9);
	std::vector<double> d_locs = unif_plane(N_d_sugg, 0, 0.9, comm);

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
	auto alpha = make_one<El::Complex<double>>();
	auto beta = make_zero<El::Complex<double>>();

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
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> EtaVec(g);
	El::Zeros(EtaVec,N_disc,1);
	tree2elemental(eta,EtaVec);

	/////////////////////////////////////////////////////////////////
	// Randomize the Incident Field
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Incident Field Randomization" << std::endl;

	using namespace std::placeholders;
	auto U_sf  = std::bind(U_func,_1,_2,&u_data);
	auto Ut_sf = std::bind(Ut_func,_1,_2,&u_data);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> U_U(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR>	S_U(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Vt_U(g);

	rsvd2(U_U,S_U,Vt_U,U_sf,Ut_sf,N_disc,N_s,R_s);


	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> US_U(g);
	El::Zeros(US_U,N_disc,R_s);
	El::DiagonalScale(El::RIGHT,El::NORMAL,S_U,U_U);
	US_U = U_U;


	// test that U is ok...
	{
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> USVt_U(g);
		El::Zeros(USVt_U,N_disc,N_s);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,US_U,Vt_U,beta,USVt_U);
		
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> x(g);
		El::Gaussian(x,N_s,1);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y_svd(g);
		El::Zeros(y_svd,N_disc,1);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> y_ex(g);
		El::Zeros(y_ex,N_disc,1);

		El::Gemm(El::NORMAL,El::NORMAL,alpha,USVt_U,x,beta,y_svd);
		U_sf(x,y_ex);
		elemental2tree(y_ex,temp);
		temp->Write2File("../results/y_ex",4);
		elemental2tree(y_svd,temp_c);
		temp_c->Write2File("../results/y_svd",4);
		temp_c->Add(temp,-1);
		temp_c->Write2File("../results/U_diff",4);
		Axpy(-1.0,y_ex,y_svd);
		double ndiff = El::TwoNorm(y_svd)/El::TwoNorm(y_ex);

		if(!rank) std::cout << "Incident Field SVD accuracy: " << ndiff << std::endl;

	}
	
	/////////////////////////////////////////////////////////////////
	// Compute the scattered field
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Scattered Field Computation" << std::endl;

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Phi(g);
	El::Zeros(Phi,N_d,R_s);
	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> US_U_i = El::View(US_U, i, 0, 1, N_disc);
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Phi_i = El::View(Phi, i, 0, 1, N_d);
		elemental2tree(US_U_i,temp);

		temp->Multiply(mask,1);
		temp->Multiply(eta,1);
		temp->ClearFMMData();
		temp->RunFMM();
		temp->Copy_FMMOutput();
		temp->Write2File("../results/scattered_field",0);
		std::vector<double> detector_values = temp->ReadVals(d_locs);

		vec2elemental(detector_values,Phi_i);
	}

	/////////////////////////////////////////////////////////////////
	// Low rank factorization of G
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Low Rank Factorization of G" << std::endl;

	auto G_sf  = std::bind(G_func,_1,_2,&g_data);
	auto Gt_sf = std::bind(Gt_func,_1,_2,&g_data);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> U_G(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR>	S_G(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Vt_G(g);

	rsvd(U_G,S_G,Vt_G,G_sf,Gt_sf,N_d,N_disc,R_d);

	{ // Test that G is good
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> r(g);
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

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> e(g);
		El::Zeros(e,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Vt_G,r,beta,e);

		//El::DistMatrix<El::Complex<double>,El::VR,El::STAR> f(g);
		//El::Zeros(f,R_d,1);
		//El::Gemm(El::NORMAL,El::NORMAL,alpha,S_G,e,beta,f);
		El::DiagonalScale(El::LEFT,El::NORMAL,S_G,e);
		//El::Display(f);
		//El::Display(GY);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> g1(g);
		El::Zeros(g1,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,U_G,e,beta,g1); // GY now actually U

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> i(g);
		El::Zeros(i,N_d,1);

		G_func(r,i,&g_data);
		
		El::Axpy(-1.0,i,g1);
		double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
		if(!rank) std::cout << "Relative Error in Approximation of G: " << ndiff << std::endl;

		delete t;

	}

	/////////////////////////////////////////////////////////////////
	// Transform phi
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Transform the Scattered Field" << std::endl;

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Phi_hat(g);
	El::Zeros(Phi_hat,R_d,1);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,Phi,beta,Phi_hat);


	/////////////////////////////////////////////////////////////////
	// Compute the SVD of S_G Vt_G DW
	/////////////////////////////////////////////////////////////////
	//
	auto B_sf  = std::bind(B_func,_1,_2,&S_G,&Vt_G,&US_U,temp,temp_c);
	auto Bt_sf = std::bind(Bt_func,_1,_2,&S_G,&Vt_G,&US_U,temp,temp_c);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> U_B(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR>	S_B(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Vt_B(g);

	rsvd(U_B,S_B,Vt_B,B_sf,Bt_sf,R_s*R_d,N_disc,R_b);


	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_G*(\eta.U_rand)
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Multiply matrix by the incident field" << std::endl;
	
	// U_rand is diagonal N_disc x N_disc. Instead of forming it,
	// we can multiply each row of Vt_tilde pointwise by U_rand
	//El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Ut_rand(g);
	//El::Zeros(Ut_rand,1,N_disc);
	//El::Transpose(U_rand,Ut_rand);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> VtU(g);
	El::Zeros(VtU,R_d,N_disc);

	//MDM(Vt_G,US_U,temp,temp_c);

	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_tildeU_rand \eta
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Rest of the Multiplcations"<<  std::endl;

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> G_eps(g);
	El::Zeros(G_eps,R_d*R_s,N_disc);
	//El::Gemm(El::NORMAL,El::NORMAL,alpha,S_G,Vt_G,beta,G_eps);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> G_eps_part(g);
	//El::Zeros(G_eps_part,R_b,R_d*R_s);
	//El::Gemm(El::NORMAL,El::NORMAL,alpha,U_B,S_B,beta,G_eps_part);
	El::DiagonalScale(El::RIGHT, El::NORMAL,S_B,U_B);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,U_B,Vt_B,beta,G_eps);

	{// test if G_eps is ok
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> r(g);
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

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> a(g);
		El::Zeros(a,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps,r,beta,a);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> b(g);
		El::Zeros(b,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,U_G,a,beta,b);


		elemental2tree(US_U,temp);

		t->Multiply(temp,1);
		tree2elemental(t,r);


		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> c(g);
		El::Zeros(c,N_d,1);
		G_func(r,c,&g_data);
		//El::Display(c,"c");
		//El::Display(b,"b");


		El::Axpy(-1.0,c,b);
		double rnorm = El::TwoNorm(b)/El::TwoNorm(c);
		if(!rank) std::cout << "G_eps rel norm: " << rnorm << std::endl;

		delete t;

	}

	/////////////////////////////////////////////////////////////////
	// Solve
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Solve"<<  std::endl;
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Eta_recon(g);
	El::Zeros(Eta_recon,N_disc,1);

	El::Complex<double> gamma;
	El::SetRealPart(gamma, 1.5);
	El::SetImagPart(gamma, 0.0);

	// Truncated SVD
	El::DistMatrix<double> s_sol(g);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> V_sol(g);
	El::SVD(G_eps, s_sol, V_sol, El::SVDCtrl<double>());

	{// test SVD
		std::vector<double> d_sol(R_d*R_s);
		El::DistMatrix<double,El::STAR,El::STAR> s_sol_star = s_sol;
		d_sol.assign(s_sol_star.Buffer(),s_sol_star.Buffer()+R_d*R_s);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Sigma_sol(g);
		El::Zeros(Sigma_sol,R_d*R_s,R_d*R_s);
		El::Diagonal(Sigma_sol, d_sol);
		El::Display(Sigma_sol,"d_sol");

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> r(g);
		El::Gaussian(r,N_disc,1);

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = cs_fn;
		t->f_max = 1;
		t->CreateTree(false);

		tree2elemental(t,r);


		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> e(g);
		El::Zeros(e,R_d*R_s,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_sol,r,beta,e);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> f(g);
		El::Zeros(f,R_d*R_s,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Sigma_sol,e,beta,f);

		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> g1(g);
		El::Zeros(g1,R_d*R_s,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps,f,beta,g1); // GY now actually U


		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> i(g);
		El::Zeros(i,N_d,1);
		El::DistMatrix<El::Complex<double>,El::VR,El::STAR> h(g);
		El::Zeros(h,R_d*R_s,1);

		elemental2tree(US_U,temp_c);
		temp_c->Multiply(t,1);
		tree2elemental(temp_c,r);
		G_func(r,i,&g_data);

		// 
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,i,beta,h); // GY now actually U
	
		El::Display(h,"h");
		El::Display(g1,"g1");

		El::Axpy(-1.0,h,g1);
		double frnorm = El::TwoNorm(g1)/El::TwoNorm(h);
		if(!rank) std::cout << "Final SVD rel norm: " << frnorm << std::endl;

		delete t;
	}


	//if(!rank){
	//	std::cout << "G H " << G_eps.Height() << std::endl;
	//	std::cout << "G W " << G_eps.Width() << std::endl;
	//	std::cout << "s H " << s_sol.Height() << std::endl;
	//	std::cout << "s W " << s_sol.Width() << std::endl;
	//	std::cout << "V H " << V_sol.Height() << std::endl;
	//	std::cout << "V W " << V_sol.Width() << std::endl;
	//}

	// truncate
	El::DistMatrix<double> s_sol_k = El::View(s_sol,0,0,k,1);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> V_sol_k = El::View(V_sol,0,0,N_disc,k);
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> G_eps_k = El::View(G_eps,0,0,R_d,k);

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> C(g);
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
	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> Sig_k(g);
	El::Diagonal(Sig_k,s_sol_k_inv);
	El::Display(Sig_k,"Sig_k");

	El::DistMatrix<El::Complex<double>,El::VR,El::STAR> D(g);
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

	delete temp;
	delete mask;
	delete eta;
	delete temp_c;
	delete fmm_mat;
	delete fmm_mat_c;

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
	PetscInt N_s = 8;
	PetscInt N_d = 8;
	PetscInt R_d = N_s/2;
	PetscInt R_s = R_d;
	PetscInt R_b = R_d;
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
  PetscOptionsGetInt (NULL, "-N_d",&             N_d  ,NULL); 
  PetscOptionsGetInt (NULL, "-N_s",&             N_s  ,NULL); 
  PetscOptionsGetInt (NULL, "-R_d",&             R_d  ,NULL); // Reduced rank of detectors
  PetscOptionsGetInt (NULL, "-k",&             k  ,NULL); // rank for truncated svd
  PetscOptionsGetInt (NULL, "-R_s",&             R_s  ,NULL); // Reduced rank of detectors
  PetscOptionsGetInt (NULL, "-R_b",&             R_b  ,NULL); // Reduced rank of detectors

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
		std::cout << "N_s: " << N_s << std::endl;
		std::cout << "N_d: " << N_d << std::endl;
		std::cout << "R_d: " << R_d << std::endl;
		std::cout << "R_s: " << R_s << std::endl;
		std::cout << "R_b: " << R_b << std::endl;
		std::cout << "k: " << k << std::endl;
	}

	faims(comm, N_d, N_s, R_d, R_s, R_b, k);
	El::Finalize();
	PetscFinalize();

	return 0;
}
