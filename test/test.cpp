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

// -------------------------------------------------------------------
// Declarations
// -------------------------------------------------------------------
int norm_test(MPI_Comm &comm);
int multiply_test(MPI_Comm &comm);
int multiply_test2(MPI_Comm &comm);
int multiply_test3(MPI_Comm &comm);
int add_test(MPI_Comm &comm);
int conj_multiply_test(MPI_Comm &comm);
int copy_test(MPI_Comm &comm);
int ptfmm_trg2tree_test(MPI_Comm &comm);
int mult_op_test(MPI_Comm &comm);
int mult_op_sym_test(MPI_Comm &comm);
int mgs_test(MPI_Comm &comm);
int compress_incident_test(MPI_Comm &comm);

// -------------------------------------------------------------------
// Definitions
// -------------------------------------------------------------------

int norm_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = one_fn;
	one->f_max = 1;

	// initialize the tree
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	double norm = one->Norm2();
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	double n_cubes = (one->GetNGLNodes()).size();
	double n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	double diff = fabs(norm - sqrt(n_cubes*n_nodes3));
	std::cout << norm << std::endl;
	std::cout <<  (one->GetNGLNodes()).size() << std::endl;
	std::cout <<  (one->GetNGLNodes()).size() << std::endl;


	if(diff < 1e-10){
		std::cout << "\033[2;32mNorm test passed! \033[0m- absolute error=" << diff  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Norm test failed! \033[0m- absolute norm=" << diff  << std::endl;
	}
	delete one;

	return 0;
}

int multiply_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *gfun = new InvMedTree<FMM_Mat_t>(comm);
	gfun->bndry = bndry;
	gfun->kernel = kernel;
	gfun->fn = ctr_pt_sol_fn;
	gfun->f_max = 1;

	InvMedTree<FMM_Mat_t> *gfunc = new InvMedTree<FMM_Mat_t>(comm);
	gfunc->bndry = bndry;
	gfunc->kernel = kernel;
	gfunc->fn = ctr_pt_sol_conj_fn;
	gfunc->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = ctr_pt_sol_conj_prod_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//multiply the two, then get their difference
	gfun->Multiply(gfunc,1);
	gfun->Add(sol,-1);

	double rel_norm = gfun->Norm2()/sol->Norm2();


	if(rel_norm < 1e-10){
		std::cout << "\033[2;32mMultiply test passed! \033[0m- relative error=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Multiply test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	delete gfunc;
	delete gfun;
	delete sol;

	return 0;
}

int multiply_test2(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

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

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = one_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	sc->Write2File("results/sc",0);
	scc->Write2File("results/scc",0);
	sol->Write2File("results/sol",0);

	//multiply the two, then get their difference
	sc->Multiply(scc,1);
	sc->Write2File("results/product",0);
	sc->Add(sol,-1);
	sc->Write2File("results/should_be_zero",0);

	double rel_norm = sc->Norm2()/sol->Norm2();


	if(rel_norm < 1e-10){
		std::cout << "\033[2;32mMultiply test 2 passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31mFAILURE! - Multiply test 2 failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	delete sc;
	delete scc;
	delete sol;

	return 0;
}

int multiply_test3(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *p1 = new InvMedTree<FMM_Mat_t>(comm);
	p1->bndry = bndry;
	p1->kernel = kernel;
	p1->fn = poly_fn;
	p1->f_max = 1;

	InvMedTree<FMM_Mat_t> *p2 = new InvMedTree<FMM_Mat_t>(comm);
	p2->bndry = bndry;
	p2->kernel = kernel;
	p2->fn = poly_fn;
	p2->f_max = 1;

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = one_fn;
	one->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = poly_prod_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//multiply the two, then get their difference
	p1->Multiply(p2,1);
	//p1->Multiply(one,1);
	p1->Add(sol,-1);
	p1->Write2File("results/shouldbezero",0);

	double rel_norm = p1->Norm2()/sol->Norm2();


	if(rel_norm < 1e-10){
		std::cout << "\033[2;32mMultiply test 3 passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31mFAILURE! - Multiply test 3 failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	delete p1;
	delete p2;
	delete sol;

	return 0;
}

int conj_multiply_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *sc = new InvMedTree<FMM_Mat_t>(comm);
	sc->bndry = bndry;
	sc->kernel = kernel;
	sc->fn = sc_fn;
	sc->f_max = 1;

	InvMedTree<FMM_Mat_t> *sc2 = new InvMedTree<FMM_Mat_t>(comm);
	sc2->bndry = bndry;
	sc2->kernel = kernel;
	sc2->fn = sc_fn;
	sc2->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = one_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//multiply the two, then get their difference
	sc->ConjMultiply(sc2,1);
	sc->Add(sol,-1);

	double rel_norm = sc->Norm2();

	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Conjugate multiply test passed! \033[0m - relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Conjugate multiply test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}


	delete sc;
	delete sc2;
	delete sol;

	return 0;
}


int conj_multiply_test2(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

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

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = sc2_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//multiply the two, then get their difference
	sc->ConjMultiply(scc,1);
	sc->Add(sol,-1);

	double rel_norm = sc->Norm2();

	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Conjugate multiply test 2 passed! \033[0m - relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Conjugate multiply test 2 failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}


	delete sc;
	delete scc;
	delete sol;

	return 0;
}

int add_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

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

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = twosin_fn;
	sol->f_max = 2;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	//multiply the two, then get their difference
	sc->Add(scc,1);
	sc->Add(sol,-1);

	double rel_norm = sc->Norm2()/sol->Norm2();


	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Addition test passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Addition test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	delete sc;
	delete scc;
	delete sol;

	return 0;
}

int copy_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *sc = new InvMedTree<FMM_Mat_t>(comm);
	sc->bndry = bndry;
	sc->kernel = kernel;
	sc->fn = sc_fn;
	sc->f_max = 1;


	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = sc_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	InvMedTree<FMM_Mat_t> *sc2 = new InvMedTree<FMM_Mat_t>(comm);
	sc2->Copy(sc);

	// get difference
	sc->Add(sc2,-1);

	double rel_norm = sc->Norm2()/sol->Norm2();


	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Copy test passed!\033[0m - relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Copy test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}


	delete sc;
	delete sc2;
	delete sol;

	return 0;
}

int ptfmm_trg2tree_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = one_fn;
	one->f_max = 1;


	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = ctr_pt_sol_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	std::vector<double> src_coord;
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);

	std::vector<double> src_vals = one->ReadVals(src_coord);
	// test the values it reads.
	int val_test = 0;
	if(fabs(src_vals[0] - 1) > 1e-10 || fabs(src_vals[1] - 0) > 1e-10){
		val_test = 1;
	}

	// create the particle fmm tree we need
	pvfmm::PtFMM_Tree* pt_tree = one->CreatePtFMMTree(src_coord, src_vals, kernel);
	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value);

	// Insert the values back in
	one->Trg2Tree(trg_value);
	one->Add(sol,-1);

	double rel_norm = one->Norm2();


	if(val_test){
		std::cout << "\033[2;31mFAILURE! Wrong value extracted. \033[0m Got: " << src_vals[0] << ", " << src_vals[1] <<". Expected: 1, 0" << std::endl;
	}

	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m PtFMM/Trg2Tree test passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - PtFMM/Trg2Tree test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}


	delete one;
	delete sol;
	delete pt_tree;

	return 0;
}

int int_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

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

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


  FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	test_fn->SetupFMM(fmm_mat);
	test_fn->RunFMM();
	test_fn->Copy_FMMOutput();
	test_fn->Write2File("results/after_int",0);

	// Use the center as the point that we can read forom
	// set up the particle fmm tree
	// set up the operator
	PetscInt m = test_fn->m;
	PetscInt M = test_fn->M;
	PetscInt n = test_fn->n;
	PetscInt N = test_fn->N;

	// check the solution...
	test_fn->Add(sol,-1);
	test_fn->Write2File("results/should_be_zero",0);
	sol->Write2File("results/sol",0);

	double rel_norm = test_fn->Norm2();

	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Integration test passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Integration test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	delete test_fn;
	delete sol;
	delete fmm_mat;

	return 0;
}


int int_test2(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = one_fn;
	one->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	// check the solution...
	//double on = one->Norm2();
	std::vector<double> integral = one->Integrate();

	if(abs(integral[0] - 1 < 1e-5) and abs(integral[1] < 1e-5)){
		std::cout << "\033[2;32m Integration test passed! \033[0m-  integra=" << integral[0] << " " << integral[1]   << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Integration test failed! \033[0m- integra=" << integral[0] << " " << integral[1]  << std::endl;
	}

	delete one;

	return 0;
}


int int_test3(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	InvMedTree<FMM_Mat_t> *prod = new InvMedTree<FMM_Mat_t>(comm);
	prod->bndry = bndry;
	prod->kernel = kernel;
	prod->fn = prod_fn;
	prod->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


	// check the solution...
	//double on = one->Norm2();
	std::vector<double> integral = prod->Integrate();

	if(abs(integral[0] - .125 < 1e-5) and abs(integral[1] -.125 < 1e-5)){
		std::cout << "\033[2;32m Integration test passed! \033[0m-  integra=" << integral[0] << " " << integral[1]   << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Integration test failed! \033[0m- integra=" << integral[0] << " " << integral[1]  << std::endl;
	}

	delete prod;

	return 0;
}

int mult_op_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	PetscInt ierr;

	InvMedTree<FMM_Mat_t> *ctr_pt = new InvMedTree<FMM_Mat_t>(comm);
	ctr_pt->bndry = bndry;
	ctr_pt->kernel = kernel;
	ctr_pt->fn = int_test_fn;
	ctr_pt->f_max = 1;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 1;

	InvMedTree<FMM_Mat_t> *phi_0 = new InvMedTree<FMM_Mat_t>(comm);
	phi_0->bndry = bndry;
	phi_0->kernel = kernel;
	phi_0->fn = one_fn;
	phi_0->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = ctr_pt_sol_neg_conj_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


  FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	temp->SetupFMM(fmm_mat);

	// Use the center as the point that we can read forom
	std::vector<double> src_coord;
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	// set up the particle fmm tree
	std::vector<double> src_vals = temp->ReadVals(src_coord); // these vals really don't matter. will be reset before use.
	pvfmm::PtFMM_Tree* pt_tree = temp->CreatePtFMMTree(src_coord, src_vals, kernel_conj);
	// set up the operator
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
	invmed_data.alpha = 0;
	MatCreateShell(comm,m,n,M,N,&invmed_data,&A);
	MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);

	// Set up the input and output vectors. input should be int_test_fn
	Vec input_vec, output_vec;
  ierr = VecCreateMPI(comm,n,PETSC_DETERMINE,&input_vec); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,n,PETSC_DETERMINE,&output_vec); CHKERRQ(ierr);

	tree2vec(ctr_pt,input_vec);
	MatMult(A,input_vec,output_vec);

	// check the solution...
	vec2tree(output_vec,ctr_pt);
	VecDestroy(&input_vec);
	VecDestroy(&output_vec);

	ctr_pt->Add(sol,-1);
	ctr_pt->Write2File("results/should_be_zero",0);
	sol->Write2File("results/sol",0);

	double rel_norm = ctr_pt->Norm2()/sol->Norm2();

	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Multiply operator test passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Multiply operator test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	MatDestroy(&A);

	delete ctr_pt;
	delete sol;
	delete temp;
	delete phi_0;
	delete fmm_mat;

	return 0;
}

/*
int mult_op_sym_test(MPI_Comm &comm){
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	PetscInt ierr;

	InvMedTree<FMM_Mat_t> *ctr_pt = new InvMedTree<FMM_Mat_t>(comm);
	ctr_pt->bndry = bndry;
	ctr_pt->kernel = kernel;
	ctr_pt->fn = int_test_fn;
	ctr_pt->f_max = 1;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 1;

	InvMedTree<FMM_Mat_t> *phi_0 = new InvMedTree<FMM_Mat_t>(comm);
	phi_0->bndry = bndry;
	phi_0->kernel = kernel;
	phi_0->fn = one_fn;
	phi_0->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = ctr_pt_sol_neg_conj_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();


  FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	temp->SetupFMM(fmm_mat);

	// Use the center as the point that we can read forom
	std::vector<double> src_coord;
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	src_coord.push_back(0.5);
	// set up the particle fmm tree
	std::vector<double> src_vals = temp->ReadVals(src_coord); // these vals really don't matter. will be reset before use.
	pvfmm::PtFMM_Tree* pt_tree = temp->CreatePtFMMTree(src_coord, src_vals, kernel_conj);
	// set up the operator
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
	invmed_data.alpha = 0;
	MatCreateShell(comm,m,n,M,N,&invmed_data,&A);
	MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);

	// Set up the input and output vectors. input should be int_test_fn
	Vec x_in, x_out, y_in, y_out;
  ierr = VecCreateMPI(comm,n,PETSC_DETERMINE,&x_in); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,n,PETSC_DETERMINE,&x_out); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,n,PETSC_DETERMINE,&y_in); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,n,PETSC_DETERMINE,&y_out); CHKERRQ(ierr);
	std::srand(std::time(NULL));
	for(int i=0;i<n;i++){
		VecSetValue(x_in,i,((double)std::rand()/(double)RAND_MAX),INSERT_VALUES);
		VecSetValue(y_in,i,((double)std::rand()/(double)RAND_MAX),INSERT_VALUES);
	}


	// Compute (M^TMx,y)
	MatMult(A,x_in,x_out);

	// hessian inner product
	PetscScalar hess_val;
	VecDot(x_out,y_in,hess_val);

	// Compute (Mx,My)
	vec2tree(x_in,temp);
	temp->RunFMM();
	temp->Copy_FMMOutput();
	tree2vec(temp,x_out);


	vec2tree(y_in,temp);
	temp->RunFMM();
	temp->Copy_FMMOutput();
	tree2vec(temp,y_out);




	// check the solution...
	vec2tree(output_vec,ctr_pt);
	VecDestroy(&input_vec);
	VecDestroy(&output_vec);

	ctr_pt->Add(sol,-1);
	ctr_pt->Write2File("results/should_be_zero",0);
	sol->Write2File("results/sol",0);

	double rel_norm = ctr_pt->Norm2()/sol->Norm2();

	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Multiply operator test passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Multiply operator test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}

	MatDestroy(&A);

	delete ctr_pt;
	delete sol;
	delete temp;
	delete phi_0;
	delete fmm_mat;

	return 0;


}
*/

int spectrum_test(MPI_Comm &comm){

  const pvfmm::Kernel<double>* kernel=&helm_kernel;
  const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

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

	// Get the total number of chebyshev nodes.
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	// total num of nodes (non ghost, leaf);
	int n_tnodes = (eta->GetNGLNodes()).size();
	int total_size = n_nodes3*n_tnodes;
	std::cout << "total_size: " << total_size << std::endl;

	std::vector<double> src_coord;
	src_coord.push_back(.3);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.3);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.3);
	src_coord.push_back(.7);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.7);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.5);
	src_coord.push_back(.7);
	//std::vector<double> src_coord = randsph(6,.12);
	//src_coord = randunif(total_size);
	//std::vector<double> src_coord = test_pts();
	std::cout << "size: " << src_coord.size() << std::endl;


  FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	eta->Write2File("results/eta",0);
	phi_0->Write2File("results/pt_sources",0);
	// compute phi_0 from the pt_sources
	phi_0->SetupFMM(fmm_mat);
	phi_0->RunFMM();
	phi_0->Copy_FMMOutput();
  phi_0->Write2File("results/phi_0",0);

	temp->SetupFMM(fmm_mat);

	std::vector<double> src_vals = temp->ReadVals(src_coord); // these vals really don't matter. will be reset before use.
	pvfmm::PtFMM_Tree* pt_tree = temp->CreatePtFMMTree(src_coord, src_vals, kernel_conj);
	// set up the operator
	PetscInt m = phi_0->m;
	PetscInt M = phi_0->M;
	PetscInt n = phi_0->n;
	PetscInt N = phi_0->N;

	for(int j=5;j>=0;j--){
		std::vector<double> src_values(12);
		std::fill(src_values.begin(),src_values.end(),0);
		src_values[j*2] = 1;

		std::vector<double> trg_value;
		pt_tree->ClearFMMData();
		pvfmm::PtFMM_Evaluate(pt_tree, trg_value, 0, &src_values);
		temp->Trg2Tree(trg_value);
		temp->ConjMultiply(phi_0,1);

		temp->Multiply(phi_0,1);

		// Run FMM ( Compute: G[ \eta * u ] )
		temp->ClearFMMData();
		temp->RunFMM();
		temp->Copy_FMMOutput();
		std::vector<double> out_vec = temp->ReadVals(src_coord);
		for(int i=0;i<6;i++){
			std::cout << out_vec[i*2] << " + " <<  out_vec[i*2+1] << "i"<< std::endl;
		}
	}



/*
	if(rel_norm < 1e-10){
		std::cout << "\033[2;32m Multiply operator test passed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
	else{
		std::cout << "\033[2;31m FAILURE! - Multiply operator test failed! \033[0m- relative norm=" << rel_norm  << std::endl;
	}
*/

	delete pt_sources;
	delete eta;
	delete temp;
	delete phi_0;
	delete fmm_mat;

	return 0;
}

int mgs_test(MPI_Comm &comm){
	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND);
	PetscErrorCode ierr;
	int n = 10;
	int m = 20;
	if(n > m){
		std::cout << "n must be less than m" << std::endl;
		return 0;
	}
	std::vector<Vec> vecs(n);

	Mat A;
	ierr = 	MatCreate(comm,&A); CHKERRQ(ierr);
	ierr = 	MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
	ierr = 	MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n); CHKERRQ(ierr); //create an m by n matrix. Let petsc decide local sizes
	ierr = 	MatSetUp(A);CHKERRQ(ierr);
	ierr = 	MatSetOption(A,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

	Mat Q;
	ierr = MatCreate(comm,&Q);CHKERRQ(ierr);
	ierr = MatSetType(Q,MATSEQDENSE);CHKERRQ(ierr);
	ierr = MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
	ierr = MatSetUp(Q);CHKERRQ(ierr);
	ierr = MatSetOption(Q,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

	Mat R;
	ierr = MatCreate(comm,&R);CHKERRQ(ierr);
	ierr = MatSetType(R,MATSEQDENSE);CHKERRQ(ierr);
	ierr = MatSetSizes(R,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
	ierr = MatSetUp(R);CHKERRQ(ierr);
	ierr = MatSetOption(R,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);


	std::vector<int> rows(m);
	for(int i=0;i<m;i++){
		rows[i] = i;
	}

	for(int i=0;i<n;i++){
		PetscScalar *a;
		Vec x;
		VecCreateMPI(comm,PETSC_DECIDE,m,&x);
		VecSetRandom(x,r);
		VecGetArray(x,&a);
		MatSetValues(A,m,&rows[0],1,&i,a,INSERT_VALUES);
		VecRestoreArray(x,&a);
		vecs[i]=x;
		//if(i == 1){
			//VecView(x,PETSC_VIEWER_STDOUT_SELF);
			//std::cout << "============================================" << std::endl;
		//}

	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> d(0.0,1.0);
	El::DistMatrix<double> D;
	m = 20;
	n = 10;
	D.Resize(20,10);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			std::cout << i << " : " << j << std::endl;
			double r2 = randn(0,1,r);
			std::cout << r2 << std::endl;
			D.Set(i,j,r2);
		}
	}

	El::Print(D,"D");
	El::DistMatrix<double> Q1;
	El::DistMatrix<double> R1;
	El::qr::Explicit(D,R1,El::QRCtrl<double>());
	El::Print(R1,"R1");
	El::Print(D,"D");

	El::DistMatrix<double> eye;
	El::Zeros(eye,n,n);
	eye.Resize(10,10);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,D,D,0.0,eye);
	El::Print(eye,"eye");


	Vec x;
	VecCreateMPI(comm,m,PETSC_DETERMINE,&x);
	VecSetRandom(x,r);
	std::cout << "============================================" << std::endl;
	VecView(x,PETSC_VIEWER_STDOUT_SELF);

	Vec2ElMatCol(x,D,4);

	std::cout << "============================================" << std::endl;
	El::Display(D,"D2");

	std::cout << "============================================" << std::endl;
	ElMatCol2Vec(x,D,7);
	VecView(x,PETSC_VIEWER_STDOUT_SELF);

	std::cout << "============================================" << std::endl;

	for(int i=0;i<n;i++){
		VecView(vecs[i],PETSC_VIEWER_STDOUT_SELF);
	}

	Vecs2ElMat(vecs,D);
	El::Display(D,"D");

	El::Scale(2,D);

	ElMat2Vecs(vecs,D);

	for(int i=0;i<n;i++){
		VecView(vecs[i],PETSC_VIEWER_STDOUT_SELF);
	}
	El::Print(D,"D");


	/*
	std::cout << "ortho_project" << std::endl;
	Vec x;
	VecCreateMPI(comm,m,PETSC_DETERMINE,&x);
	VecSetRandom(x,r);

	ortho_project(vecs,x);
	vecs.push_back(x);
	for(int i=0;i<vecs.size();i++){
			VecDot(vecs[i],vecs.back(),&val);
			std::cout << val << std::endl;
	}

	for(int i=0;i<n;i++){
		VecDestroy(&vecs[i]);
	}

	PetscReal norm_val;

	{
		Vec x;
		VecCreateMPI(comm,m,PETSC_DETERMINE,&x);
		VecSetRandom(x,r);
		VecNorm(x,NORM_2, &norm_val);
		VecScale(x, 1/norm_val);
		vecs.push_back(x);
	}


	for(int i=1;i<n;i++){
		// initialize random vector
		Vec x;
		VecCreateMPI(comm,m,PETSC_DETERMINE,&x);
		VecSetRandom(x,r);
		// normalize
		VecNorm(x,NORM_2, &norm_val);
		VecScale(x, 1/norm_val);
		// orhtogonalize
		ortho_project(vecs, x);
		// normalize
		VecNorm(x,NORM_2, &norm_val);
		std::cout << norm_val << std::endl;
		VecScale(x, 1/norm_val);
		// push back
		vecs.push_back(x);
	}
*/

	PetscRandomDestroy(&r);

	return 0;

}

int compress_incident_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	int create_number = 100;
	//pt_src_locs = equisph(create_number,1);
	create_number  = 2;
	pt_src_locs = equiplane(create_number,0,0.1); // should generate 100 points with the above change
	std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int n_pt_srcs = pt_src_locs.size()/3;
	int data_dof = 2;

	{
		coeffs.clear();
		for(int i=0;i<pt_src_locs.size()/3;i++){
			coeffs.push_back(1); // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
			//coeffs.push_back(randn(0,1,r));
			//coeffs.push_back(.5);
		}
	}

	// need to couple the incident field compressor with the non constant coefficient solver

	InvMedTree<FMM_Mat_t> *t1 = new InvMedTree<FMM_Mat_t>(comm);
	t1->bndry = bndry;
	t1->kernel = kernel;
	t1->fn = phi_0_fn; // this is the function that computes a linear combination of pt sources located on sphere
	t1->f_max = 4;


	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = mask_fn; // this is the function that computes a linear combination of pt sources located on sphere
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *bg = new InvMedTree<FMM_Mat_t>(comm);
	bg->bndry = bndry;
	bg->kernel = kernel;
	bg->fn = k2_fn;
	bg->f_max = 2;

	InvMedTree<FMM_Mat_t> *eta_k2 = new InvMedTree<FMM_Mat_t>(comm);
	eta_k2->bndry = bndry;
	eta_k2->kernel = kernel;
	eta_k2->fn = eta_plus_k2_fn;
	eta_k2->f_max = .2;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel_conj;
	temp->fn = zero_fn;
	temp->f_max = 0;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	bg->Write2File("results/bg",0);
	t1->Write2File("results/incident",0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);

	temp->SetupFMM(fmm_mat);

	std::vector<Vec> ortho_vec;

	PetscReal norm_val = 1;

	PetscInt m = t1->m;
	PetscInt M = t1->M;
	PetscInt n = t1->n;
	PetscInt N = t1->N;

	{
		int cd =InvMedTree<FMM_Mat_t>::cheb_deg;
		int nc = (cd+1)*(cd+2)*(cd+3)/6;
		int nls = (t1->GetNGLNodes()).size();
		int ys = 2*nc*nls;

		std::cout << "n " << n << std::endl;
		std::cout << "OTHER " << nc << " " <<  ys << " " << nls << " " << 2 << " " << nc << std::endl;

	}

	int num_trees = 0;
	int n_times = 0;

	double compress_tol = 0.1;
	Vec coeffs_vec;
	VecCreateMPI(comm,PETSC_DECIDE,pt_src_locs.size()/3,&coeffs_vec);

	IncidentData incident_data;
	incident_data.bndry = bndry;
	incident_data.kernel = kernel;
	incident_data.fn = phi_0_fn;
	incident_data.coeffs = &coeffs;
	incident_data.comm = comm;


	Mat inc_mat;
	MatCreateShell(comm,m,n_pt_srcs*data_dof,M,n_pt_srcs*data_dof,&incident_data,&inc_mat); // not sure about the sizes here...
	MatShellSetOperation(inc_mat,MATOP_MULT,(void(*)(void))incident_mult);


	// Transpose mult should we want it... not being used for now
	std::vector<double> src_samples = bg->ReadVals(pt_src_locs); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* u_trans = bg->CreatePtFMMTree(pt_src_locs, src_samples, kernel_conj);
	IncidentTransData inc_trans_data;
	inc_trans_data.comm = comm;
	inc_trans_data.temp_c = temp;
	inc_trans_data.src_coord = pt_src_locs;
	inc_trans_data.pt_tree = u_trans;

	// create the transpose of the incident field operator. This is a particle fmm
	Mat inc_mat_trans;
	MatCreateShell(comm,n_pt_srcs*data_dof,m,n_pt_srcs*data_dof,M,&inc_trans_data,&inc_mat_trans); // not sure about the sizes here...
	MatShellSetOperation(inc_mat_trans,MATOP_MULT,(void(*)(void))incident_transpose_mult);

	// Need to define a process grid
	int size;
	ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
	El::Grid grid(comm,size);

	El::DistMatrix<double> Q(grid);
	El::DistMatrix<double> Q_tilde(grid);
	El::DistMatrix<double> R_tilde(grid);
	RandQRData randqrdata;
	randqrdata.A = &inc_mat;
	randqrdata.Atrans = &inc_mat_trans;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = m;
	randqrdata.n = n_pt_srcs;
	randqrdata.M = M;
	randqrdata.N = n_pt_srcs;
	randqrdata.grid = &grid;

	ierr = RandQR(&randqrdata, 2, compress_tol,1);CHKERRQ(ierr);

	int l1 = Q.Height();
	int m1 = Q.Width();

	// let's see if the matrix Q is orthogonal
	El::DistMatrix<double> eye(grid);
	El::Zeros(eye,m1,m1);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,Q,Q,0.0,eye);
	El::Display(eye);

	// transform U
	El::DistMatrix<double> U_hat(grid);
	El::Zeros(U_hat,l1,m1);
	El::Gemm(El::NORMAL,El::TRANSPOSE,1.0,Q,R_tilde,0.0,U_hat);

	// transform the left side of the equations
	// we can just scatter the data now in U_hat
	std::vector<Vec> u_hat_vec;
	for(int i=0;i<m1;i++){
			Vec u_hat;
			VecCreateMPI(comm,n,PETSC_DETERMINE,&u_hat);
			u_hat_vec.push_back(u_hat);
	}
	ElMat2Vecs(u_hat_vec,U_hat);
	
	//ElMat2Vecs(ortho_vec,U_hat); //no longer orthogonal
	std::vector<Vec> phi_hat_vec;
	for(int i=0;i<m1;i++){
		vec2tree(u_hat_vec[i],temp);
		vec2tree(u_hat_vec[i],t1);
		scatter_born(t1,eta_k2,temp);
		{
			Vec phi_hat;
			VecCreateMPI(comm,n,PETSC_DETERMINE,&phi_hat);
			tree2vec(temp,phi_hat);
			u_hat_vec.push_back(phi_hat);
		}
	}

	PetscRandomDestroy(&r);
	for(int i=0;i<num_trees;i++){
		//delete treevec[i];
		VecDestroy(&ortho_vec[i]);
		//VecDestroy(&orig_vec[i]);
	}
	return 0;

}


int randqr_test1(MPI_Comm &comm){

	PetscErrorCode ierr;
	double compress_tol = 0.01;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	PetscInt m = 20;
	PetscInt M = 20;
	PetscInt n = 10;
	PetscInt N = 10;

	Mat A_mat;
	ierr = 	MatCreate(comm,&A_mat); CHKERRQ(ierr);
	ierr = 	MatSetType(A_mat,MATSEQDENSE);CHKERRQ(ierr);
	ierr = 	MatSetSizes(A_mat,PETSC_DECIDE,PETSC_DECIDE,m,n); CHKERRQ(ierr); //create an m by n matrix. Let petsc decide local sizes
	ierr = 	MatSetUp(A_mat);CHKERRQ(ierr);
	ierr = MatSetRandom(A_mat,r); CHKERRQ(ierr);

	Mat LR_mat;
	MatCreateShell(comm,M,M,M,M,&A_mat,&LR_mat);
	MatShellSetOperation(LR_mat,MATOP_MULT,(void(*)(void))LR_mult);

	Mat LRt_mat;
	MatCreateShell(comm,M,M,M,M,&A_mat,&LRt_mat);
	MatShellSetOperation(LRt_mat,MATOP_MULT,(void(*)(void))LR_mult);


	// create the transpose
	std::cout << "dbgr1" << std::endl;
	Mat At_mat;
	MatCreateTranspose(A_mat,&At_mat);
	std::cout << "dbgr2" << std::endl;

	// Need to define a process grid
	int size;
	ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
	El::Grid grid(comm,size);

	El::DistMatrix<double> Q(grid);
	El::DistMatrix<double> Q_tilde(grid);
	El::DistMatrix<double> R_tilde(grid);
	RandQRData randqrdata;
	randqrdata.A = &A_mat;
	randqrdata.Atrans = &At_mat;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = m;
	randqrdata.n = n;
	randqrdata.M = M;
	randqrdata.N = N;
	randqrdata.grid = &grid;

	ierr = RandQR(&randqrdata, 3, 8,1);CHKERRQ(ierr);

	int l1 = Q.Height();
	int m1 = Q.Width();

	// let's see if the matrix Q is orthogonal
	El::DistMatrix<double> eye(grid);
	El::Zeros(eye,m1,m1);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,Q,Q,0.0,eye);
	//El::Display(eye);

	// transform U
	El::DistMatrix<double> U_hat(grid);
	El::Zeros(U_hat,l1,m1);
	El::Gemm(El::NORMAL,El::TRANSPOSE,1.0,Q,R_tilde,0.0,U_hat);


	PetscRandomDestroy(&r);
	return 0;

}


int randqr_test2(MPI_Comm &comm){

	PetscErrorCode ierr;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	PetscInt m = 10;
	PetscInt M = 10;
	PetscInt n = 5;
	PetscInt N = 5;

	Mat A_mat;
	ierr = 	MatCreate(comm,&A_mat); CHKERRQ(ierr);
	ierr = 	MatSetType(A_mat,MATSEQDENSE);CHKERRQ(ierr);
	ierr = 	MatSetSizes(A_mat,PETSC_DECIDE,PETSC_DECIDE,m,n); CHKERRQ(ierr); //create an m by n matrix. Let petsc decide local sizes
	ierr = 	MatSetUp(A_mat);CHKERRQ(ierr);
	ierr = MatSetRandom(A_mat,r); CHKERRQ(ierr);

	Mat LR_mat;
	MatCreateShell(comm,M,M,M,M,&A_mat,&LR_mat);
	MatShellSetOperation(LR_mat,MATOP_MULT,(void(*)(void))LR_mult);

	Mat LRt_mat;
	MatCreateShell(comm,M,M,M,M,&A_mat,&LRt_mat);
	MatShellSetOperation(LRt_mat,MATOP_MULT,(void(*)(void))LR_mult);


	// create the transpose
	std::cout << "dbgr1" << std::endl;
	Mat At_mat;
	MatCreateTranspose(A_mat,&At_mat);
	std::cout << "dbgr2" << std::endl;

	// Need to define a process grid
	int size;
	ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
	El::Grid grid(comm,size);

	El::DistMatrix<double> Q(grid);
	El::DistMatrix<double> Q_tilde(grid);
	El::DistMatrix<double> R_tilde(grid);
	RandQRData randqrdata;
	randqrdata.A = &LR_mat;
	randqrdata.Atrans = &LRt_mat;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = m;
	randqrdata.n = m;
	randqrdata.M = M;
	randqrdata.N = M;
	randqrdata.grid = &grid;

	ierr = RandQR(&randqrdata, 0, 3,1);CHKERRQ(ierr);

	int l1 = Q.Height();
	int m1 = Q.Width();

	// let's see if the matrix Q is orthogonal
	El::DistMatrix<double> eye(grid);
	El::Zeros(eye,m1,m1);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,Q,Q,0.0,eye);
	//El::Display(eye);

	// transform U
	El::DistMatrix<double> U_hat(grid);
	El::Zeros(U_hat,l1,m1);
	El::Gemm(El::NORMAL,El::TRANSPOSE,1.0,Q,R_tilde,0.0,U_hat);


	PetscRandomDestroy(&r);
	return 0;

}


int factorize_G_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	std::vector<double> detector_coord = equiplane(2,0,.9);

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = mask_fn;
	mask->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);

	temp->SetupFMM(fmm_mat);

	PetscInt m = temp->m;
	PetscInt M = temp->M;
	PetscInt n = temp->n;
	PetscInt N = temp->N;
	int n_detectors = detector_coord.size()/3;

	double compress_tol = 0.1;

	std::vector<double> detector_samples = temp->ReadVals(detector_coord); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* Gt_tree = temp->CreatePtFMMTree(detector_coord, detector_samples, kernel_conj);

	InvMedData g_data;
	g_data.temp = temp;
	g_data.phi_0 = mask;
	g_data.src_coord = detector_coord;
	g_data.pt_tree = Gt_tree;

	Mat G_mat;
	MatCreateShell(comm,n_detectors*2,n,n_detectors*2,N,&g_data,&G_mat); // not sure about the sizes here...
	MatShellSetOperation(G_mat,MATOP_MULT,(void(*)(void))G_mult);

	// create the transpose of the incident field operator. This is a particle fmm
	Mat Gt_mat;
	MatCreateShell(comm,n,n_detectors*2,N,n_detectors*2,&g_data,&Gt_mat); // not sure about the sizes here...
	MatShellSetOperation(Gt_mat,MATOP_MULT,(void(*)(void))Gt_mult);

	// Need to define a process grid
	int size;
	ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
	El::Grid grid(comm,size);

	El::DistMatrix<double> Q(grid);
	El::DistMatrix<double> Q_tilde(grid);
	El::DistMatrix<double> R_tilde(grid);
	RandQRData randqrdata;
	randqrdata.A = &G_mat;
	randqrdata.Atrans = &Gt_mat;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = n_detectors*2;
	randqrdata.n = n;
	randqrdata.M = n_detectors*2;
	randqrdata.N = N;
	randqrdata.grid = &grid;

	ierr = RandQR(&randqrdata, 0, compress_tol,1);CHKERRQ(ierr);

	int l1 = Q.Height();
	int m1 = Q.Width();

	// let's see if the matrix Q is orthogonal
	El::DistMatrix<double> eye(grid);
	El::Zeros(eye,m1,m1);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,Q,Q,0.0,eye);
	El::Display(eye);

	// transform U
	El::DistMatrix<double> U_hat(grid);
	El::Zeros(U_hat,l1,m1);
	El::Gemm(El::NORMAL,El::TRANSPOSE,1.0,Q,R_tilde,0.0,U_hat);

	// transform the left side of the equations
	// we can just scatter the data now in U_hat
	/*
	std::vector<Vec> u_hat_vec;
	for(int i=0;i<m1;i++){
			Vec u_hat;
			VecCreateMPI(comm,n,PETSC_DETERMINE,&u_hat);
			u_hat_vec.push_back(u_hat);
	}
	ElMat2Vecs(u_hat_vec,U_hat);
	
	//ElMat2Vecs(ortho_vec,U_hat); //no longer orthogonal
	std::vector<Vec> phi_hat_vec;
	for(int i=0;i<m1;i++){
		vec2tree(u_hat_vec[i],temp);
		vec2tree(u_hat_vec[i],t1);
		scatter_born(t1,eta_k2,temp);
		{
			Vec phi_hat;
			VecCreateMPI(comm,n,PETSC_DETERMINE,&phi_hat);
			tree2vec(temp,phi_hat);
			u_hat_vec.push_back(phi_hat);
		}
	}
	*/

	PetscRandomDestroy(&r);
	//for(int i=0;i<num_trees;i++){
		//delete treevec[i];
	//	VecDestroy(&ortho_vec[i]);
		//VecDestroy(&orig_vec[i]);
	//}
	return 0;

}


int GfuncGtfunc_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	std::vector<double> detector_coord = {.5,.5,.5}; //equiplane(1,0,1.0);
	//std::vector<double> detector_coord = {.4,.4,.4,.5,.5,.5,.6,.6,.6};

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
	int n_detectors = detector_coord.size()/3;

	std::vector<double> detector_samples = temp->ReadVals(detector_coord); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* Gt_tree = temp->CreatePtFMMTree(detector_coord, detector_samples, kernel_conj);

	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = detector_coord;
	g_data.pt_tree = Gt_tree;

	El::Matrix<El::Complex<double>> x;
	El::Gaussian(x,M/2,1);
	elemental2tree(x,temp);
	std::vector<double> fvec = {1};
	temp->FilterChebTree(fvec);
	tree2elemental(temp,x);

	El::Matrix<El::Complex<double>> y;
	El::Gaussian(y,n_detectors,1);

	El::Matrix<El::Complex<double>> Gx;
	El::Zeros(Gx,n_detectors,1);
	G_func(x,Gx,g_data);
	//El::Display(Gx,"Gx");
	//El::Display(y,"y");

	std::cout << El::Dot(y,Gx) << std::endl;

	// Now do it the other way
	El::Matrix<El::Complex<double>> Gty;
	El::Zeros(Gty,M/2,1);

	//El::Display(x,"x");
	Gt_func(y,Gty,g_data);

	//El::Display(Gty,"Gty");
	elemental2tree(x,temp);
	elemental2tree(Gty,temp1);
	temp->ConjMultiply(temp1,1);

	std::vector<double> xGty = temp->Integrate();

	std::cout << "=================================" << std::endl;
	std::cout << xGty[0] << std::endl;
	std::cout << xGty[1] << std::endl;

	return 0;

}

int GGt_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	std::vector<double> detector_coord = {.5,.5,.5}; //equiplane(1,0,1.0);
	//std::vector<double> detector_coord = {.4,.4,.4,.5,.5,.5,.6,.6,.6};

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

	InvMedTree<FMM_Mat_t> *one = new InvMedTree<FMM_Mat_t>(comm);
	one->bndry = bndry;
	one->kernel = kernel;
	one->fn = cs_fn;
	one->f_max = 1;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;


	InvMedTree<FMM_Mat_t> *csol = new InvMedTree<FMM_Mat_t>(comm);
	csol->bndry = bndry;
	csol->kernel = kernel;
	csol->fn = ctr_pt_sol_conj_fn;
	csol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	mask->Write2File("../results/mask",0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);

	temp->SetupFMM(fmm_mat);

	PetscInt m = temp->m;
	PetscInt M = temp->M;
	PetscInt n = temp->n;
	PetscInt N = temp->N;
	int n_detectors = detector_coord.size()/3;

	std::vector<double> detector_samples = temp->ReadVals(detector_coord); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* Gt_tree = temp->CreatePtFMMTree(detector_coord, detector_samples, kernel_conj);

	InvMedData g_data;
	g_data.temp = temp;
	g_data.phi_0 = mask;
	g_data.src_coord = detector_coord;
	g_data.pt_tree = Gt_tree;

	Mat G_mat;
	MatCreateShell(comm,n_detectors*2,n,n_detectors*2,N,&g_data,&G_mat); // not sure about the sizes here...
	MatShellSetOperation(G_mat,MATOP_MULT,(void(*)(void))G_mult);

	// create the transpose of the incident field operator. This is a particle fmm
	Mat Gt_mat;
	MatCreateShell(comm,n,n_detectors*2,N,n_detectors*2,&g_data,&Gt_mat); // not sure about the sizes here...
	MatShellSetOperation(Gt_mat,MATOP_MULT,(void(*)(void))Gt_mult);

	Vec x;
	std::cout << "n: " << n <<std::endl;
	std::cout << "N: " << N <<std::endl;
	ierr = VecCreateMPI(comm,n,N,&x); CHKERRQ(ierr);
	ierr = VecSetRandom(x,r); CHKERRQ(ierr);
	vec2tree(x,temp);
	std::vector<double> fvec = {1};
	temp->FilterChebTree(fvec);
	tree2vec(temp,x);

	Vec y;
	Vec conj;
	ierr = VecCreateMPI(comm,n_detectors*2,n_detectors*2,&y); CHKERRQ(ierr);
	ierr = VecCreateMPI(comm,n_detectors*2,n_detectors*2,&conj); CHKERRQ(ierr);
	ierr = VecSetRandom(y,r); CHKERRQ(ierr);
	//VecSetValue(y,0,1,INSERT_VALUES);
	//VecSetValue(y,1,1,INSERT_VALUES);
	VecSetValue(conj,0,1,INSERT_VALUES);
	VecSetValue(conj,1,-1,INSERT_VALUES);


	Vec Gx;
	ierr = VecCreateMPI(comm,n_detectors*2,n_detectors*2,&Gx); CHKERRQ(ierr);
	VecView(Gx, PETSC_VIEWER_STDOUT_SELF);
	std::vector<double> t1 = sol->ReadVals(detector_coord);
	for(int i=0;i<t1.size();i++){
		std::cout << "sol: " << i << " " << t1[i] << std::endl;
	}

	ierr = MatMult(G_mat,x,Gx); CHKERRQ(ierr);
	VecView(Gx, PETSC_VIEWER_STDOUT_SELF);

	//PetscReal Gxy;
	PetscInt size;
	double real =0;
	double im = 0;
	VecGetSize(Gx,&size);
	const double* v1;
	ierr = VecGetArrayRead(Gx,&v1);CHKERRQ(ierr);
	const double* v2;
	ierr = VecGetArrayRead(y,&v2);CHKERRQ(ierr);
	for(int i=0;i<size/2;i++){
		double a =v1[i*2 + 0];
		double b =v1[i*2 + 1];
		double c =v2[i*2 + 0];
		double d =v2[i*2 + 1];

		real += a*c + b*d;
		im   += b*c - a*d;
	}

	std::cout << "ip =============================" <<std::endl;
	std::cout << real << std::endl;
	std::cout << im << std::endl;

	ierr = VecRestoreArrayRead(Gx,&v1);CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(y,&v2);CHKERRQ(ierr);
	

	// Now do it the other way
	Vec Gsy;
	ierr = VecCreateMPI(comm,n,N,&Gsy); CHKERRQ(ierr);

	//ierr = VecPointwiseMult(y,conj,y); CHKERRQ(ierr);
	ierr = MatMult(Gt_mat,y,Gsy); CHKERRQ(ierr);
	vec2tree(x,temp);
	vec2tree(Gsy,one);
	temp->ConjMultiply(one,1);

	std::vector<double> xGsy = temp->Integrate();

	std::cout << "=================================" << std::endl;
	std::cout << xGsy[0] << std::endl;
	std::cout << xGsy[1] << std::endl;

	PetscRandomDestroy(&r);
	return 0;

}

int UUt_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	int create_number = 100;
	//pt_src_locs = equisph(create_number,1);
	create_number  = 1;
	//pt_src_locs = equiplane(create_number,0,0.1); // should generate 100 points with the above change
	pt_src_locs = {.4,.4,.4,.6,.6,.6};
	pt_src_locs = {.5,.5,.5};
	std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int n_pt_srcs = pt_src_locs.size()/3;

	{
		coeffs.clear();
		for(int i=0;i<pt_src_locs.size()/3;i++){
			coeffs.push_back(1); // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
			//coeffs.push_back(randn(0,1,r));
			//coeffs.push_back(.5);
		}
	}

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel_conj;
	temp->fn = zero_fn;
	temp->f_max = 0;

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = cmask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *prod = new InvMedTree<FMM_Mat_t>(comm);
	prod->bndry = bndry;
	prod->kernel = kernel;
	prod->fn = cs_fn;
	prod->f_max = 1;

	InvMedTree<FMM_Mat_t> *sol = new InvMedTree<FMM_Mat_t>(comm);
	sol->bndry = bndry;
	sol->kernel = kernel;
	sol->fn = ctr_pt_sol_fn;
	sol->f_max = 1;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);

	temp->SetupFMM(fmm_mat);

	PetscInt m = temp->m;
	PetscInt M = temp->M;
	PetscInt n = temp->n;
	PetscInt N = temp->N;

	Vec coeffs_vec;
	VecCreateMPI(comm,PETSC_DECIDE,pt_src_locs.size()/3,&coeffs_vec);

	IncidentData incident_data;
	incident_data.bndry = bndry;
	incident_data.kernel = kernel;
	incident_data.fn = phi_0_fn;
	incident_data.coeffs = &coeffs;
	incident_data.mask = mask;
	incident_data.comm = comm;

	Mat U_mat;
	MatCreateShell(comm,m,n_pt_srcs*data_dof,M,n_pt_srcs*data_dof,&incident_data,&U_mat); // not sure about the sizes here...
	MatShellSetOperation(U_mat,MATOP_MULT,(void(*)(void))incident_mult);

	// Transpose mult should we want it... not being used for now
	//std::vector<double> src_samples = bg->ReadVals(pt_src_locs); //Not sure exactly what this will do...
	//pvfmm::PtFMM_Tree* u_trans = bg->CreatePtFMMTree(pt_src_locs, src_samples, kernel_conj);
	IncidentTransData inc_trans_data;
	inc_trans_data.comm = comm;
	inc_trans_data.mask = mask;
	inc_trans_data.temp_c = temp;
	inc_trans_data.src_coord = pt_src_locs;
	//inc_trans_data.pt_tree = u_trans;

	// create the transpose of the incident field operator. This is a particle fmm
	Mat Ut_mat;
	MatCreateShell(comm,n_pt_srcs*data_dof,m,n_pt_srcs*data_dof,M,&inc_trans_data,&Ut_mat); // not sure about the sizes here
	MatShellSetOperation(Ut_mat,MATOP_MULT,(void(*)(void))incident_transpose_mult);


	Vec x;
	ierr = VecCreateMPI(comm,n_pt_srcs*data_dof,n_pt_srcs*data_dof,&x); CHKERRQ(ierr);
	ierr = VecSetRandom(x,r); CHKERRQ(ierr);
	//ierr = VecSetValue(x,0,1,INSERT_VALUES);
	//ierr = VecSetValue(x,1,0,INSERT_VALUES);

	Vec y;
	ierr = VecCreateMPI(comm,m,M,&y); CHKERRQ(ierr);
	ierr = VecSetRandom(y,r); CHKERRQ(ierr);
	vec2tree(y,temp);
	std::vector<double> fvec = {1};
	temp->FilterChebTree(fvec);
	tree2vec(temp,y);

	Vec Ux;
	ierr = VecCreateMPI(comm,m,M,&Ux); CHKERRQ(ierr);

	ierr = MatMult(U_mat,x,Ux); CHKERRQ(ierr);
	//PetscReal Uxy;
	//ierr = VecDot(Ux,y,&Uxy); CHKERRQ(ierr);

	// No do it the other way
	Vec Usy;
	ierr = VecCreateMPI(comm,n_pt_srcs*data_dof,n_pt_srcs*data_dof,&Usy); CHKERRQ(ierr);
	ierr = MatMult(Ut_mat,y,Usy); CHKERRQ(ierr);
	vec2tree(Ux,temp);
	//temp->Multiply(mask,1);
	sol->Multiply(mask,1);
	temp->Add(sol,-1);
	temp->Write2File("../results/temp",0);
	std::cout << "nrm: " << temp->Norm2() << std::endl;
	
	//PetscReal xUsy;
	//ierr = VecDot(x,Usy,&xUsy); CHKERRQ(ierr);
	
	PetscInt size;
	double real =0;
	double im = 0;
	VecGetSize(x,&size);
	const double* v1;
	ierr = VecGetArrayRead(x,&v1);CHKERRQ(ierr);
	const double* v2;
	ierr = VecGetArrayRead(Usy,&v2);CHKERRQ(ierr);
	for(int i=0;i<size/2;i++){
		double a =v1[i*2 + 0];
		double b =v1[i*2 + 1];
		double c =v2[i*2 + 0];
		double d =v2[i*2 + 1];

		real += a*c + b*d;
		im   += b*c - a*d;
	}
	ierr = VecRestoreArrayRead(x,&v1);CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(Usy,&v2);CHKERRQ(ierr);


	vec2tree(Ux,temp);
	vec2tree(y,mask);
	temp->ConjMultiply(mask,1);


	std::cout << real << std::endl;
	std::cout << im << std::endl;
	std::vector<double> xGsy = temp->Integrate();

	std::cout << "=================================" << std::endl;
	std::cout << xGsy[0] << std::endl;
	std::cout << xGsy[1] << std::endl;

	PetscRandomDestroy(&r);
	return 0;
}



#undef __FUNCT__
#define __FUNCT__ "comp_inc_test"
int comp_inc_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	int data_dof = 2;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////

	int create_number = 8;
	pt_src_locs = equiplane(create_number,0,0.1);
	std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int N_s = pt_src_locs.size()/3;

	std::vector<double> d_locs = equiplane(create_number,0,0.9);
	std::cout << "Number gnereated=" << d_locs.size() << std::endl;
	int N_d = d_locs.size()/3;

	{
		coeffs.clear();
		for(int i=0;i<N_s*data_dof;i++){
			coeffs.push_back((i%2 == 0) ? 1 : 0 ); // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
		}
	}

	//////////////////////////////////////////////////////////////
	// Set up FMM stuff
	//////////////////////////////////////////////////////////////

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

	InvMedTree<FMM_Mat_t> *mask = new InvMedTree<FMM_Mat_t>(comm);
	mask->bndry = bndry;
	mask->kernel = kernel;
	mask->fn = mask_fn;
	mask->f_max = 1;

	InvMedTree<FMM_Mat_t> *eta = new InvMedTree<FMM_Mat_t>(comm);
	eta->bndry = bndry;
	eta->kernel = kernel;
	eta->fn = eta2_fn;
	eta->f_max = .01;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);
	temp->SetupFMM(fmm_mat);

	FMM_Mat_t *fmm_mat_c=new FMM_Mat_t;
	fmm_mat_c->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel_conj);
	temp_c->SetupFMM(fmm_mat_c);

	// Tree sizes
	PetscInt m = temp->m;
	PetscInt M = temp->M;
	PetscInt n = temp->n;
	PetscInt N = temp->N;

	//////////////////////////////////////////////////////////////
	// Set up incident field operators
	//////////////////////////////////////////////////////////////

	std::cout << "Set up incident field operator" << std::endl;
	Vec coeffs_vec;
	VecCreateMPI(comm,PETSC_DECIDE,N_s*data_dof,&coeffs_vec);

	IncidentData incident_data;
	incident_data.bndry = bndry;
	incident_data.kernel = kernel;
	incident_data.fn = phi_0_fn;
	incident_data.coeffs = &coeffs;
	incident_data.mask = mask;
	incident_data.comm = comm;

	// forward operator is direct evaluation
	Mat U_mat;
	MatCreateShell(comm,m,N_s*data_dof,M,N_s*data_dof,&incident_data,&U_mat); // not sure about the sizes here...
	MatShellSetOperation(U_mat,MATOP_MULT,(void(*)(void))incident_mult);

	IncidentTransData inc_trans_data;
	inc_trans_data.comm = comm;
	inc_trans_data.mask = mask;
	inc_trans_data.temp_c = temp_c;
	inc_trans_data.src_coord = pt_src_locs;

	// create the transpose of the incident field operator. This is a particle fmm
	Mat Ut_mat;
	MatCreateShell(comm,N_s*data_dof,m,N_s*data_dof,M,&inc_trans_data,&Ut_mat); // not sure about the sizes here
	MatShellSetOperation(Ut_mat,MATOP_MULT,(void(*)(void))incident_transpose_mult);

	//////////////////////////////////////////////////////////////
	// Randomized, low rank factorization of incident field operator
	//////////////////////////////////////////////////////////////


	std::cout << "Set up Randomized QR" << std::endl;
	// Need to define a process grid
	int size;
	ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
	El::Grid grid(comm,size);

	El::DistMatrix<double> Q(grid);
	El::DistMatrix<double> Q_tilde(grid);
	El::DistMatrix<double> R_tilde(grid);
	RandQRData randqrdata;
	randqrdata.A = &U_mat;
	randqrdata.Atrans = &Ut_mat;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = m;
	randqrdata.n = N_s*data_dof;
	randqrdata.M = M;
	randqrdata.N = N_s*data_dof;
	randqrdata.grid = &grid;

	std::cout << "Beginning Randomized QR" << std::endl;
	ierr = RandQR(&randqrdata, 1, .01, 1); CHKERRQ(ierr);
	std::cout << "Finished with Randomized QR" << std::endl;

	int l1 = Q.Height();
	assert(l1 == N);
	int R_s = Q.Width();
	std::cout << "R_s: " << R_s << std::endl;
	std::cout << "l1: " << l1 << std::endl;

	// let's see if the matrix Q is orthogonal
	/*
	El::DistMatrix<double> eye(grid);
	El::Zeros(eye,R_s,R_s);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,Q,Q,1.0,eye);
	El::Display(eye);
	*/

	std::cout << "Create U_hat" << std::endl;
	// Create the transformed incident field
	El::DistMatrix<double> U_hat(grid);
	El::Zeros(U_hat,N,R_s);
	El::Gemm(El::NORMAL,El::TRANSPOSE,1.0,Q,R_tilde,0.0,U_hat);



	//////////////////////////////////////////////////////////////
	// Create the scattered field
	//////////////////////////////////////////////////////////////
	// Set up the G and Gt operators
	std::vector<double> detector_samples = temp_c->ReadVals(d_locs); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* Gt_tree = temp_c->CreatePtFMMTree(d_locs, detector_samples, kernel_conj);

	InvMedData g_data;
	g_data.temp = temp;
	g_data.phi_0 = mask;
	g_data.src_coord = d_locs; // for G_mult src_coord is really the detector coordinates
	g_data.pt_tree = Gt_tree;
	g_data.filter = false; // REMEMBER TO SET FILTER TO TRUE LATER!!!!!!!!

	Mat G_mat;
	MatCreateShell(comm,N_d*data_dof,n,N_d*data_dof,N,&g_data,&G_mat); // not sure about the sizes here...
	MatShellSetOperation(G_mat,MATOP_MULT,(void(*)(void))G_mult);

	// create the transpose of the incident field operator. This is a particle fmm
	Mat Gt_mat;
	MatCreateShell(comm,n,N_d*data_dof,N,N_d*data_dof,&g_data,&Gt_mat); // not sure about the sizes here...
	MatShellSetOperation(Gt_mat,MATOP_MULT,(void(*)(void))Gt_mult);


	//first get the column vectors of U_hat
	std::cout << "Beginning computing transformed scattered field" << std::endl;
	std::vector<Vec> u_hat_vec;
	for(int i=0;i<R_s;i++){
			Vec u_hat;
			VecCreateMPI(comm,n,N,&u_hat);
			u_hat_vec.push_back(u_hat);
	}
	ElMat2Vecs(u_hat_vec,U_hat);
	std::cout << "dbg1" << std::endl;

	// then multiply by eta and apply G_operator, putting it in a new matrix as we go
	El::DistMatrix<double> Phi_hat(grid);
	El::Zeros(Phi_hat,N_d*data_dof,R_s);
	//std::vector<Vec> phi_hat_vec;
	std::cout << "dbg2" << std::endl;
	for(int i=0;i<R_s;i++){
		vec2tree(u_hat_vec[i],temp);
		temp->Multiply(eta,1);
		tree2vec(temp, u_hat_vec[i]);
		{
			Vec phi_hat_petsc;
			VecCreateMPI(comm,N_d*data_dof,N_d*data_dof,&phi_hat_petsc);
			ierr = MatMult(G_mat,u_hat_vec[i], phi_hat_petsc); CHKERRQ(ierr);
			ierr = VecView(phi_hat_petsc,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
			Vec2ElMatCol(phi_hat_petsc, Phi_hat, i);
		}
	std::cout << "dbg3" << std::endl;
	}
	El::Display(Phi_hat);


	//////////////////////////////////////////////////////////////
	// I guess now we need to factorize G
	//////////////////////////////////////////////////////////////
	g_data.filter = true; // NEEDS TO BE TRUE FOR THE FACTORIZATION!!!

	randqrdata.A = &G_mat;
	randqrdata.Atrans = &Gt_mat;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = N_d*data_dof;
	randqrdata.n = n;
	randqrdata.M = N_d*data_dof;
	randqrdata.N = N;
	randqrdata.grid = &grid;

	std::cout << "Beginning Randomized QR" << std::endl;
	ierr = RandQR(&randqrdata, 0, .01, 1); CHKERRQ(ierr);
	std::cout << "Finished with Randomized QR" << std::endl;

	//////////////////////////////////////////////////////////////
	// Rearrange to a more useable form
	//////////////////////////////////////////////////////////////
	std::cout << Q.Height() << std::endl;
	std::cout << Q.Width() << std::endl;
	int R_d = Q.Width();
	El::Display(Q);
	El::DistMatrix<double> G_hat(grid);
	El::Zeros(G_hat,R_d*data_dof,N);
	El::Gemm(El::TRANSPOSE,El::TRANSPOSE,1.0,R_tilde,Q_tilde,0.0,G_hat);

	El::DistMatrix<double> Phi_2hat(grid);
	El::Zeros(Phi_2hat,R_d,R_s); // THIS SIZE SEEMS SUSPICIOUS
	std::cout << Phi_hat.Height() << std::endl;
	std::cout << Phi_hat.Width() << std::endl;
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,Q,Phi_hat,0.0,Phi_2hat);
	El::Display(Phi_2hat);

	Phi_2hat.Resize(R_s,R_d);
	El::Display(Phi_2hat);
	

	//////////////////////////////////////////////////////////////
	// Solve with regularization
	//////////////////////////////////////////////////////////////




	/*
	std::cout << "dbg4" << std::endl;
	PetscRandomDestroy(&r);
	std::cout << "dbg5" << std::endl;
	for(int i=0;i<R_s;i++){
		VecDestroy(&u_hat_vec[i]);
		std::cout << "dbg6" << std::endl;
	}
	std::cout << "dbg7" << std::endl;
	delete fmm_mat;
	std::cout << "dbg7" << std::endl;
	delete fmm_mat_c;
	std::cout << "dbg7" << std::endl;

	//delete temp;
	std::cout << "dbg7" << std::endl;
	//delete temp_c;
	std::cout << "dbg7" << std::endl;
	//delete eta;
	std::cout << "dbg7" << std::endl;
	//delete mask;
	std::cout << "dbg7" << std::endl;
	*/


	return 0;
}

int filter_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;


	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	PetscInt m = temp->m;
	PetscInt M = temp->M;
	PetscInt n = temp->n;
	PetscInt N = temp->N;

	std::vector<double> coeff_scaling = {1};

	Vec x;
	std::cout << "n: " << n <<std::endl;
	std::cout << "N: " << N <<std::endl;
	ierr = VecCreateMPI(comm,n,N,&x); CHKERRQ(ierr);
	ierr = VecSetRandom(x,r); CHKERRQ(ierr);
	vec2tree(x,temp);
	temp->Write2File("../results/notsmoothed",0);
	temp->FilterChebTree(coeff_scaling);
	temp->Write2File("../results/smoothed",0);

	PetscRandomDestroy(&r);
	delete temp;
	return 0;

}

int el_test(MPI_Comm &comm){

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	PetscErrorCode ierr;

	InvMedTree<FMM_Mat_t> *temp = new InvMedTree<FMM_Mat_t>(comm);
	temp->bndry = bndry;
	temp->kernel = kernel;
	temp->fn = sc_fn;
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

	PetscInt m = temp->m;
	PetscInt M = temp->M;
	PetscInt n = temp->n;
	PetscInt N = temp->N;
	std::cout << M << std::endl;

	El::Matrix<El::Complex<double>> A;
	El::Zeros(A,M/2,1); // dividing by data_dof

	tree2elemental(temp,A);
	elemental2tree(A,temp1);

	temp->Write2File("../results/eltestA",0);
	temp1->Write2File("../results/eltestB",0);
	temp->Add(temp1,-1);
	temp->Write2File("../results/eltestC",0);



	delete temp;
	return 0;

}














int faims(MPI_Comm &comm){
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

	int create_number = 8;
	pt_src_locs = equiplane(create_number,0,0.1);
	//pt_src_locs = {.5,.5,.5};
	std::cout << "Number gnereated=" << pt_src_locs.size() << std::endl;
	int N_s = pt_src_locs.size()/3;

	std::vector<double> d_locs = equiplane(create_number,0,0.9);
	std::cout << "Number gnereated=" << d_locs.size() << std::endl;
	int N_d = d_locs.size()/3;

	{
		coeffs.clear();
		for(int i=0;i<N_s*data_dof;i++){
			coeffs.push_back((i%2 == 0) ? 1 : 0 ); // we push back all ones when we initially build the trees adaptively so we get a fine enough mesh
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

	G_data g_data;
	g_data.temp = temp;
	g_data.mask= mask;
	g_data.src_coord = d_locs;
	g_data.pt_tree = Gt_tree;

	int R_d = N_d/4;
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
	std::cout << "Incident Field Randomization" << std::endl;
	El::Matrix<double> rand_coeffs;
	El::Gaussian(rand_coeffs, N_s*data_dof, 1);
	coeffs.assign(rand_coeffs.Buffer(),rand_coeffs.Buffer()+N_s*data_dof);

	std::cout << "Printing Random Incident Field Coefficients" << std::endl;
	for(int i=0;i<coeffs.size();i++){
		std::cout << coeffs[i] << std::endl;
	}
	std::cout << "Done" << std::endl;

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
		El::Gaussian(RandomWeights, N_d, 1);
		El::Matrix<El::Complex<double>> Y_i = El::View(Y, 0, i, N_disc, 1);
		Gt_func(RandomWeights,Y_i,g_data);
		/*
		detector_values.assign(RandomWeights.Buffer(),RandomWeights.Buffer()+N_d*data_dof);

		Gt_tree->ClearFMMData();
		std::vector<double> trg_value;
		pvfmm::PtFMM_Evaluate(Gt_tree, trg_value, 0, &detector_values);

		// Insert the values back in
		temp->Trg2Tree(trg_value);
		temp->Multiply(mask,1);
		tree2elemental(temp, Y_i);
		*/
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
		El::Matrix<El::Complex<double>> Y_i = El::View(Y, 0, i, N_disc, 1);
		El::Matrix<El::Complex<double>> GY_i = El::View(GY, 0, i, N_d, 1);
		G_func(Y_i,GY_i,g_data);
		/*
		elemental2tree(Y_i,temp);
		temp->Multiply(mask,1);
		temp->ClearFMMData();
		temp->RunFMM();
		temp->Copy_FMMOutput();
		detector_values = temp->ReadVals(d_locs); // this should be size 2*N_d, we must now convert to a complex Elemental mat


		vec2elemental(detector_values,GY_i);
		*/
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
	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);

	std::cout << "alpha " << alpha << std::endl; 
	std::cout << "beta " << beta<< std::endl; 

	El::Gemm(El::ADJOINT,El::ADJOINT,alpha,V,Y,beta,Vt_tilde);

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

	for(int i=0;i<R_d;i++){
		El::Matrix<El::Complex<double>> V_i = El::View(Vt_tilde, i, 0, 1, N_disc);
		El::Matrix<El::Complex<double>> VU_i = El::View(Vt_tildeU, i, 0, 1, N_disc);
		El::Hadamard(Ut_rand,V_i,VU_i);
	}

	//El::Display(Vt_tildeU,"Vt_tildeU");
	std::cout << "Done w/ Vt_tildeU" << std::endl;

	/////////////////////////////////////////////////////////////////
	// Now we have Phi_hat = \Sigma Vt_tildeU_rand \eta
	/////////////////////////////////////////////////////////////////
	std::cout << "Rest of the Multiplcations"<<  std::endl;

	El::Matrix<El::Complex<double>> G_eps;
	El::Zeros(G_eps,R_d,N_disc);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,Sigma,Vt_tildeU,beta,G_eps);

	/////////////////////////////////////////////////////////////////
	// Solve
	/////////////////////////////////////////////////////////////////
	std::cout << "Solve"<<  std::endl;
	El::Matrix<El::Complex<double>> Eta_recon;
	El::Zeros(Eta_recon,N_disc,1);

	//std::vector<double> gamma(N_disc);
	//std::fill(gamma.begin(),gamma.end(),1.5);
	//El::Matrix<El::Complex<double>> Gamma;
	//El::Zeros(Gamma,N_disc,N_disc);
	//El::Diagonal(Gamma,gamma);
	
	El::Complex<double> gamma;
	El::SetRealPart(gamma, 1.5);
	El::SetImagPart(gamma, 0.0);

	//El::Tikhonov(G_eps, Phi_hat, Gamma, Eta_recon, El::TIKHONOV_CHOLESKY);
	//El::Ridge(G_eps, Phi_hat, 1.5, Eta_recon);
	
	// Truncated SVD
	int r = 0;
	int k = R_d/2;
	std::cout << "k: " << k << std::endl;
	El::Matrix<double> s_sol;
	El::Matrix<El::Complex<double>> V_sol;
	El::SVD(G_eps, s_sol, V_sol, El::SVDCtrl<double>());

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
	std::cout << "dbg2" << std::endl;

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
	temp->Write2File("../results/sol",0);
	temp->Add(eta,-1);
	std::cout << "Relative Error: " << temp->Norm2()/eta->Norm2() << std::endl;



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

	std::cout << "MINDEPTH: " << MINDEPTH << std::endl;

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
//	norm_test(comm);
//	add_test(comm);
//	multiply_test(comm);
//	multiply_test2(comm);
//	multiply_test3(comm);
	//conj_multiply_test(comm);
	//conj_multiply_test2(comm);
//	copy_test(comm);
//	int_test(comm);
//	int_test2(comm);
	//int_test3(comm);
//	ptfmm_trg2tree_test(comm);
//	mult_op_test(comm);
//	spectrum_test(comm);
//	mgs_test(comm);
//  compress_incident_test(comm);
	//factorize_G_test(comm);
	//randqr_test1(comm);
	//randqr_test2(comm);
//	GGt_test(comm);
//	UUt_test(comm);
  //comp_inc_test(comm);
	//filter_test(comm);
	//el_test(comm);
	faims(comm);
	//GfuncGtfunc_test(comm);
	El::Finalize();
	PetscFinalize();

	return 0;
}
