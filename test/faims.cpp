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



typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;



int faims(MPI_Comm &comm, int R_d, int k, int create_number){
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	//PetscErrorCode ierr;

	int data_dof = 2;

	//PetscRandom r;
	//PetscRandomCreate(comm,&r);
	//PetscRandomSetSeed(r,time(NULL));
	//PetscRandomSetType(r,PETSCRAND48);

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////

	pt_src_locs = equiplane(create_number,0,0.1);
	//pt_src_locs = {.5,.5,.5};
	std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int N_s = pt_src_locs.size()/3;

	std::vector<double> d_locs = equiplane(create_number,0,0.9);
	std::cout << "Number gnereated=" << d_locs.size() << std::endl;
	int N_d = d_locs.size()/3;
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

	std::cout << "alpha " << alpha << std::endl; 
	std::cout << "beta " << beta<< std::endl; 

	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = d_locs;
	g_data.pt_tree = Gt_tree;

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

	/////////////////////////////////////////////////////////////////
	// Eta to Elemental Vec
	/////////////////////////////////////////////////////////////////
	std::cout << "Perturbation to Elemental Vector" << std::endl;
	El::Matrix<El::Complex<double>> EtaVec;
	El::Zeros(EtaVec,N_disc,1);
	tree2elemental(eta,EtaVec);

	/////////////////////////////////////////////////////////////////
	// Randomize the Incident Field
	/////////////////////////////////////////////////////////////////
	int R_s = 1;
	std::cout << "Incident Field Randomization" << std::endl;
	El::Matrix<double> rand_coeffs;
	El::Gaussian(rand_coeffs, N_s*data_dof, 1);
	coeffs.assign(rand_coeffs.Buffer(),rand_coeffs.Buffer()+N_s*data_dof);

	// Create random combination via direct evaluation
	InvMedTree<FMM_Mat_t>* rand_inc = new InvMedTree<FMM_Mat_t>(comm);
	rand_inc->bndry = bndry;
	rand_inc->kernel = kernel;
	rand_inc->fn = phi_0_fn;
	rand_inc->f_max = 4;
	rand_inc->CreateTree(false);

	// Need to mask it to remove the singularities
	rand_inc->Multiply(mask,1);

	// convert the tree into a vector. This vector represents the function
	// that we passed into the tree constructor (which contains the current 
	// random coefficients).
	El::Matrix<El::Complex<double>> U_rand;
	El::Zeros(U_rand,N_disc,1);
	tree2elemental(rand_inc,U_rand);

	// and normalize it
	El::Scale(El::TwoNorm(U_rand),U_rand);
	El::Matrix<El::Complex<double>> U_rand_sc;
	El::Zeros(U_rand_sc,N_s,1);
	Ut_func(U_rand,U_rand_sc,u_data);

	El::Matrix<El::Complex<double>> U_rand_sc_t;
	El::Zeros(U_rand_sc_t,1,N_s);
	El::Adjoint(U_rand_sc,U_rand_sc_t);

	El::Matrix<double> s_u;
	El::Zeros(s_u,1,1);

	El::Matrix<El::Complex<double>> V_u;
	El::Zeros(V_u,1,N_s);

	El::SVD(U_rand_sc_t, s_u, V_u, El::SVDCtrl<double>());
	El::Display(U_rand_sc_t);
	El::Display(s_u);

	std::vector<double> s_u_vec(R_s);
	s_u_vec.assign(s_u.Buffer(),s_u.Buffer()+R_s);

	El::Matrix<El::Complex<double>> Sigma_u;
	El::Diagonal(Sigma_u, s_u_vec);

	El::Matrix<El::Complex<double>> UrU;
	El::Zeros(UrU,N_disc,R_s);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,U_rand,U_rand_sc_t,beta,UrU); // the SVD above transofrmed this U_rand_sc_t into U_u, the SVD factor

	El::Matrix<El::Complex<double>> UrUS;
	El::Zeros(UrUS,N_disc,R_s);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,UrU,Sigma_u,beta,UrUS); // the SVD above transofrmed this U_rand_sc_t into U_u, the SVD factor
	El::Copy(UrUS,U_rand);

	elemental2tree(U_rand,temp);
	temp->Write2File("../results/U_rand",0);

	/////////////////////////////////////////////////////////////////
	// Compute the scattered field
	/////////////////////////////////////////////////////////////////
	std::cout << "Scattered Field Computation" << std::endl;

	El::Matrix<El::Complex<double>> Phi;
	El::Zeros(Phi,N_d,1);
	elemental2tree(U_rand,temp);
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
	std::cout << "Low Rank Factorization of G" << std::endl;
		
	/*
	 * First we need a random Gaussian vector
	 */

	El::Matrix<El::Complex<double>> RandomWeights;
	El::Matrix<El::Complex<double>> Y; // the projection of omaga through G*
	//El::Matrix<El::Complex<double>> Y_i;
	El::Zeros(Y,N_disc,R_d);


	for(int i=0;i<R_d;i++){
		std::cout << "Pt_FMM " << i << std::endl; 
		El::Gaussian(RandomWeights, N_d, 1);
		El::Matrix<El::Complex<double>> Y_i = El::View(Y, 0, i, N_disc, 1);
		Gt_func(RandomWeights,Y_i,g_data);
	}

	El::Matrix<El::Complex<double>> R; // the projection of omaga through G*
	El::qr::Explicit(Y, R, El::QRCtrl<double>());

	// Now Y is such that G* \approx YY*G*.Thus we can compute GY
	// Compute it's SVD and then multiply by Q, thus giving the approx 
	// SVD of G!!!
	
	El::Matrix<El::Complex<double>> GY;
	El::Zeros(GY,N_d,R_d);
	//El::Matrix<El::Complex<double>> GY_i;
	for(int i=0;i<R_d;i++){
		std::cout << "ChebFMM " << i << std::endl; 
		El::Matrix<El::Complex<double>> Y_i = El::View(Y, 0, i, N_disc, 1);
		El::Matrix<El::Complex<double>> GY_i = El::View(GY, 0, i, N_d, 1);
		G_func(Y_i,GY_i,g_data);
	}

	El::Matrix<double> s;
	El::Zeros(s,R_d,1);

	El::Matrix<El::Complex<double>> V;
	El::Zeros(V,R_d,R_d);

	El::SVD(GY, s, V, El::SVDCtrl<double>());

	std::vector<double> d(R_d);
	d.assign(s.Buffer(),s.Buffer()+R_d);

	El::Matrix<El::Complex<double>> Sigma;
	El::Zeros(Sigma,R_d,R_d);
	El::Diagonal(Sigma, d);

	// G \approx GYY* = U\Sigma\V*Y*
	// So We take V* and multiply by Y*
	El::Matrix<El::Complex<double>> Vt_tilde;
	El::Zeros(Vt_tilde,R_d,N_disc);
	El::Gemm(El::ADJOINT,El::ADJOINT,alpha,V,Y,beta,Vt_tilde);

	{ // Test the G is good
		El::Matrix<El::Complex<double>> r;
		El::Gaussian(r,N_disc,1);
		elemental2tree(r,temp);
		std::vector<double> filt = {1,1};
		temp->FilterChebTree(filt);
		tree2elemental(temp,r);


		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = prod_fn;
		t->f_max = 1;
		t->CreateTree(false);

		tree2elemental(t,r);


		El::Matrix<El::Complex<double>> e;
		El::Zeros(e,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Vt_tilde,r,beta,e);

		El::Matrix<El::Complex<double>> f;
		El::Zeros(f,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Sigma,e,beta,f);

		El::Matrix<El::Complex<double>> g;
		El::Zeros(g,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,GY,f,beta,g); // GY now actually U


		El::Matrix<El::Complex<double>> i;
		El::Zeros(i,N_d,1);

		G_func(r,i,g_data);
		
		El::Axpy(-1.0,i,g);
		std::cout << "Norm diff: " << El::TwoNorm(g)/El::TwoNorm(i) << std::endl;

		delete t;

	}

	/////////////////////////////////////////////////////////////////
	// Transform phi
	/////////////////////////////////////////////////////////////////
	std::cout << "Transform the Scattered Field" << std::endl;


	El::Matrix<El::Complex<double>> Phi_hat;
	El::Zeros(Phi_hat,R_d,1);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,GY,Phi,beta,Phi_hat);


	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_tilde*(\eta.U_rand)
	/////////////////////////////////////////////////////////////////
	std::cout << "Multiply matrix by the incident field" << std::endl;
	
	// U_rand is diagonal N_disc x N_disc. Instead of forming it,
	// we can multiply each row of Vt_tilde pointwise by U_rand
	El::Matrix<El::Complex<double>> Ut_rand;
	El::Zeros(Ut_rand,1,N_disc);
	El::Transpose(U_rand,Ut_rand);

	El::Matrix<El::Complex<double>> Vt_tildeU;
	El::Zeros(Vt_tildeU,R_d,N_disc);

	
	for(int i=0;i<R_d;i++){ // Hadamard product
		El::Matrix<El::Complex<double>> V_i = El::View(Vt_tilde, i, 0, 1, N_disc);
		El::Matrix<El::Complex<double>> VU_i = El::View(Vt_tildeU, i, 0, 1, N_disc);
		El::Matrix<El::Complex<double>> Vt_i;
		El::Matrix<El::Complex<double>> VUt_i;
		El::Transpose(V_i,Vt_i);
		El::Transpose(VU_i,VUt_i);
		elemental2tree(Vt_i,temp);
		elemental2tree(U_rand,temp_c);
		temp->Multiply(temp_c,1);
		tree2elemental(temp,VUt_i);
		El::Transpose(VUt_i,VU_i);

	}

	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_tildeU_rand \eta
	/////////////////////////////////////////////////////////////////
	std::cout << "Rest of the Multiplcations"<<  std::endl;

	El::Matrix<El::Complex<double>> G_eps;
	El::Zeros(G_eps,R_d,N_disc);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,Sigma,Vt_tildeU,beta,G_eps);

	{// test if G_eps is ok
		El::Matrix<El::Complex<double>> r;
		El::Zeros(r,N_disc,1);

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = prod_fn;
		t->f_max = 1;
		t->CreateTree(false);

		tree2elemental(t,r);

		El::Matrix<El::Complex<double>> a;
		El::Zeros(a,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps,r,beta,a);

		El::Matrix<El::Complex<double>> b;
		El::Zeros(b,N_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,GY,a,beta,b);


		elemental2tree(U_rand,temp);

		t->Multiply(temp,1);
		tree2elemental(t,r);


		El::Matrix<El::Complex<double>> c;
		El::Zeros(c,N_d,1);
		G_func(r,c,g_data);
		El::Display(c,"c");
		El::Display(b,"b");


		El::Axpy(-1.0,c,b);
		std::cout << "G_eps rel norm: " << El::TwoNorm(b)/El::TwoNorm(c) << std::endl;

		delete t;

	}

	/////////////////////////////////////////////////////////////////
	// Solve
	/////////////////////////////////////////////////////////////////
	std::cout << "Solve"<<  std::endl;
	El::Matrix<El::Complex<double>> Eta_recon;
	El::Zeros(Eta_recon,N_disc,1);

	El::Complex<double> gamma;
	El::SetRealPart(gamma, 1.5);
	El::SetImagPart(gamma, 0.0);

	// Truncated SVD
	El::Matrix<double> s_sol;
	El::Matrix<El::Complex<double>> V_sol;
	El::SVD(G_eps, s_sol, V_sol, El::SVDCtrl<double>());

	{// test SVD
		std::vector<double> d_sol(R_d);
		d_sol.assign(s_sol.Buffer(),s_sol.Buffer()+R_d);

		El::Matrix<El::Complex<double>> Sigma_sol;
		El::Zeros(Sigma_sol,R_d,R_d);
		El::Diagonal(Sigma_sol, d_sol);
		El::Display(Sigma_sol,"d_sol");

		El::Matrix<El::Complex<double>> r;
		El::Gaussian(r,N_disc,1);

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = prod_fn;
		t->f_max = 1;
		t->CreateTree(false);

		tree2elemental(t,r);


		El::Matrix<El::Complex<double>> e;
		El::Zeros(e,R_d,1);
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_sol,r,beta,e);

		El::Matrix<El::Complex<double>> f;
		El::Zeros(f,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,Sigma_sol,e,beta,f);

		El::Matrix<El::Complex<double>> g;
		El::Zeros(g,R_d,1);
		El::Gemm(El::NORMAL,El::NORMAL,alpha,G_eps,f,beta,g); // GY now actually U


		El::Matrix<El::Complex<double>> i;
		El::Zeros(i,N_d,1);
		El::Matrix<El::Complex<double>> h;
		El::Zeros(h,R_d,1);

		elemental2tree(U_rand,temp_c);
		temp_c->Multiply(t,1);
		tree2elemental(temp_c,r);
		G_func(r,i,g_data);

		// 
		El::Gemm(El::ADJOINT,El::NORMAL,alpha,GY,i,beta,h); // GY now actually U
		
		El::Axpy(-1.0,h,g);
		std::cout << "Final SVD rel norm: " << El::TwoNorm(g)/El::TwoNorm(h) << std::endl;

		delete t;
	}



	std::cout << "G H " << G_eps.Height() << std::endl;
	std::cout << "G W " << G_eps.Width() << std::endl;
	std::cout << "s H " << s_sol.Height() << std::endl;
	std::cout << "s W " << s_sol.Width() << std::endl;
	std::cout << "V H " << V_sol.Height() << std::endl;
	std::cout << "V W " << V_sol.Width() << std::endl;

	// truncate
	El::Matrix<double> s_sol_k = El::View(s_sol,0,0,k,1);
	El::Matrix<El::Complex<double>> V_sol_k = El::View(V_sol,0,0,N_disc,k);
	El::Matrix<El::Complex<double>> G_eps_k = El::View(G_eps,0,0,R_d,k);

	El::Matrix<El::Complex<double>> C;
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
	El::Matrix<El::Complex<double>> Sig_k;
	El::Diagonal(Sig_k,s_sol_k_inv);
	El::Display(Sig_k,"Sig_k");

	El::Matrix<El::Complex<double>> D;
	El::Zeros(D,k,1);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,Sig_k,C,beta,D);

	El::Gemm(El::NORMAL,El::NORMAL,alpha,V_sol_k,D,beta,Eta_recon);
	std::cout << "V H " << Eta_recon.Height() << std::endl;
	std::cout << "V W " << Eta_recon.Width() << std::endl;

	

	/////////////////////////////////////////////////////////////////
	// Test
	/////////////////////////////////////////////////////////////////
	std::cout << "Test!" << std::endl ;
	elemental2tree(Eta_recon,temp);
	temp->Multiply(mask,1);
	temp->Write2File("../results/sol",0);
	temp->Add(eta,-1);
	std::cout << "Relative Error: " << temp->Norm2()/eta->Norm2() << std::endl;



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

	std::cout << N_pts << std::endl;
	std::cout << R_d << std::endl;
	std::cout << k << std::endl;

	faims(comm, R_d, k, N_pts);
	El::Finalize();
	PetscFinalize();

	return 0;
}
