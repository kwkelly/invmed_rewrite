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

	double rel_norm = sc->Norm2()/sol->Norm2();

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

	double rel_norm = one->Norm2()/sol->Norm2();


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

	double rel_norm = test_fn->Norm2()/sol->Norm2();

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
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,D,D,1.0,eye);
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

	PetscRandom r;
	PetscRandomCreate(comm,&r);
	PetscRandomSetSeed(r,time(NULL));
	PetscRandomSetType(r,PETSCRAND48);

	int n_pt_srcs = 100;
	pt_src_locs = equisph(n_pt_srcs,1);

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
	temp->kernel = kernel;
	temp->fn = zero_fn;
	temp->f_max = 0;

	// initialize the trees
	InvMedTree<FMM_Mat_t>::SetupInvMed();

	bg->Write2File("results/bg",0);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(InvMedTree<FMM_Mat_t>::mult_order,InvMedTree<FMM_Mat_t>::cheb_deg,comm,kernel);

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
	MatCreateShell(comm,n,n_pt_srcs,N,n_pt_srcs,&incident_data,&inc_mat); // not sure about the sizes here...
	MatShellSetOperation(inc_mat,MATOP_MULT,(void(*)(void))incident_mult);

	// Transpose mult should we want it... not being used for now
	std::vector<double> src_samples = bg->ReadVals(pt_src_locs); //Not sure exactly what this will do...
	pvfmm::PtFMM_Tree* u_trans = bg->CreatePtFMMTree(pt_src_locs, src_samples, kernel_conj);
	IncidentTransData inc_trans_data;
	inc_trans_data.comm = comm;
	inc_trans_data.temp = temp;
	inc_trans_data.src_coord = pt_src_locs;
	inc_trans_data.pt_tree = u_trans;

	El::DistMatrix<double> Q;
	El::DistMatrix<double> Q_tilde;
	El::DistMatrix<double> R_tilde;
	RandQRData randqrdata;
	randqrdata.A = &inc_mat;
	randqrdata.Q = &Q;
	randqrdata.Q_tilde = &Q_tilde;
	randqrdata.R_tilde = &R_tilde;
	randqrdata.comm = comm;
	randqrdata.r = r;
	randqrdata.m = m;
	randqrdata.n = n;
	randqrdata.M = M;
	randqrdata.N = n_pt_srcs;;

	RandQR(&randqrdata, compress_tol);

	/*
	while((norm_val > compress_tol or n_times < 2) and num_trees < n_pt_srcs){
		num_trees++;
		{
			coeffs.clear();
			for(int i=0;i<pt_src_locs.size()/3;i++){
				VecSetValue(coeffs_vec,i,randn(0,1,r),INSERT_VALUES);
				//coeffs.push_back(randn(0,1,r));
			}
		}

		//InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		//t->bndry = bndry;
		//t->kernel = kernel;
		//t->fn = phi_0_fn;
		//t->f_max = 4;
		//t->CreateTree(false);
		//t->Write2File("results/t",0);
		//

		//temp->Copy(t);

		//temp->Multiply(bg,1);
		//temp->RunFMM();
		//temp->Copy_FMMOutput();
		//temp->Add(t,1);
		//t->Copy(temp);
		//t->Write2File("results/tthing",0);

		// create vector
		{
		Vec t2_vec;
		//Vec orig;
		VecCreateMPI(comm,n,PETSC_DETERMINE,&t2_vec);
		//VecDuplicate(t2_vec,&orig);
		//tree2vec(t,t2_vec);
		incident_mult(inc_mat, coeffs_vec, t2_vec);
		//VecView(t2_vec,PETSC_VIEWER_STDOUT_WORLD);
		//VecCopy(t2_vec,orig); // now that we have the copy we can mess with the old one;
		//orig_vec.push_back(orig);
		//VecView(t2_vec,	PETSC_VIEWER_STDOUT_WORLD);
		// Normalize it
		VecNorm(t2_vec,NORM_2,&norm_val);
		VecScale(t2_vec,1/norm_val);
		// project it
		if(ortho_vec.size() > 0){
			ortho_project(ortho_vec,t2_vec);
			VecNorm(t2_vec,NORM_2,&norm_val);
			// renormalize
			VecScale(t2_vec,1/norm_val);
			// add vector and tree to the vecs
		}
		std::cout << "norm_val " << norm_val << std::endl;
		ortho_vec.push_back(t2_vec);
		//delete t;
		//vec2tree(t2_vec,t);
		//treevec.push_back(t);
		}

		if(norm_val < compress_tol){
			n_times++;
		}
		std::cout << "num_trees: " << num_trees << std::endl;
	}
*/

	/*
	// Need to create the original matrix U containing all the different incident fields (NOT randomized)
	std::vector<Vec> u_vec;
	for(int j=0;j<n_pt_srcs ;j++){
		{
			coeffs.clear();
			for(int i=0;i<pt_src_locs.size()/3;i++){
				coeffs.push_back((j==i) ? 1 : 0);
			}
		}

		InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
		t->bndry = bndry;
		t->kernel = kernel;
		t->fn = phi_0_fn;
		t->f_max = 4;
		t->CreateTree(false);
		//t->Write2File("results/t",0);

		//temp->Copy(t);

		//temp->Multiply(bg,1);
		//temp->RunFMM();
		//temp->Copy_FMMOutput();
		//temp->Add(t,1);
		//t->Copy(temp);
		//t->Write2File("results/tthing",0);

		{
		Vec t2_vec;
		VecCreateMPI(comm,n,PETSC_DETERMINE,&t2_vec);
		tree2vec(t,t2_vec);
		u_vec.push_back(t2_vec);
		}
		delete t;
	}
	*/

	/*
	int l1 = ortho_vec.size();
	int m1;
	double mat_norm;
	VecGetSize(ortho_vec[0],&m1);
	El::DistMatrix<double> Q;
	Q.Resize(m1,l1);
	Vecs2ElMat(ortho_vec,Q);
	*/

	/*
	int n1 = u_vec.size();
	El::DistMatrix<double> U;
	U.Resize(m1,n1);
	Vecs2ElMat(u_vec,U);

	for(int i=0;i<n_pt_srcs;i++){
		VecDestroy(&u_vec[i]);
	}

	El::DistMatrix<double> A;
	El::Zeros(A,n1,l1);

	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,U,Q,1.0,A);
	El::DistMatrix<double> R;
	El::qr::Explicit(A,R,El::QRCtrl<double>());

	*/
	int l1 = Q.Height();
	int m1 = Q.Width();

	// transform U
	El::DistMatrix<double> U_hat;
	El::Zeros(U_hat,l1,m1);
	El::Gemm(El::NORMAL,El::TRANSPOSE,1.0,Q,R_tilde,1.0,U_hat);

	// transform the left side of the equations
	// By linearity we can just scatter the data now in U_hat
	std::vector<Vec> u_hat_vec;
	for(int i=0;i<l1;i++){
			Vec u_hat;
			VecCreateMPI(comm,n,PETSC_DETERMINE,&u_hat);
			u_hat_vec.push_back(u_hat);
	}
	ElMat2Vecs(u_hat_vec,U_hat);
	
	//ElMat2Vecs(ortho_vec,U_hat); //no longer orthogonal
	std::vector<Vec> phi_hat_vec;
	for(int i=0;i<l1;i++){
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
//	conj_multiply_test(comm);
//	copy_test(comm);
//	int_test(comm);
//	ptfmm_trg2tree_test(comm);
//	mult_op_test(comm);
//	spectrum_test(comm);
//	mgs_test(comm);
	compress_incident_test(comm);
	El::Finalize();

	return 0;
}
