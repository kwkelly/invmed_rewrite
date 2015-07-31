#include "invmed_tree.hpp"
#include "El.hpp"
#include <iostream>
#include "convert_elemental.hpp"
#include "operators.hpp"

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

/*
 * This function applies the operator G, which consists of a pointwise
 * multiplication with a mask function, a volume fmm evaluation and then
 * sampling values at predefined points
 */
int G_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, G_data *g_data){

	// This function simply computes the convolution of G with an input U
	// and then gets only the output at the detector locations
	InvMedTree<FMM_Mat_t>* temp = g_data->temp;
	InvMedTree<FMM_Mat_t>* mask = g_data->mask;
	std::vector<double> detector_coord = g_data->src_coord;
	bool filter = g_data->filter;
	MPI_Comm comm = g_data->comm;
	int rank;
	MPI_Comm_rank(comm,&rank);
	if(!rank) std::cout << "G function " << std::endl; 

	//std::cout << "Norm in" << El::OneNorm(x) << std::endl;
	
	elemental2tree(x,temp);
	//MPI_Barrier(comm);
	temp->Write2File("../results/g",6);
	temp->Multiply(mask,1);
	//MPI_Barrier(comm);
	temp->ClearFMMData();
	//temp->Write2File("../results/x_after_clear",6);
	MPI_Barrier(comm);
	temp->RunFMM();
	temp->Copy_FMMOutput();

	temp->Write2File("../results/gf_y",4);
	std::vector<double> detector_values = temp->ReadVals(detector_coord);

	vec2elemental(detector_values,y);
	return 0;
}


/*
 * This function is, of course, the operator G*, which is a particle fmm (at
 * the detector location) evaluated everywhere, which is then converted to an
 * fmm tree, multiplied pointwise by a mask and returned as an elemental
 * vector
 */
int Gt_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, G_data *g_data){

	std::vector<double> detector_coord = g_data->src_coord;
	pvfmm::PtFMM_Tree* Gt_tree = g_data->pt_tree;
	InvMedTree<FMM_Mat_t>* temp = g_data->temp;
	InvMedTree<FMM_Mat_t>* mask = g_data->mask;
	MPI_Comm comm = g_data->comm;
	int rank;
	MPI_Comm_rank(comm,&rank);

	if(!rank) std::cout << "G* function " << std::endl; 

	//int n = y.Height();
	int n = (temp->ChebPoints()).size()/3;
	int nd = detector_coord.size()*2/3;
	std::vector<double> detector_values(nd);

	elemental2vec(y,detector_values);

	Gt_tree->ClearFMMData();
	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(Gt_tree, trg_value, n, &detector_values);

	// Insert the values back in
	temp->Trg2Tree(trg_value);
	temp->Write2File("../results/gt",6);
	temp->Multiply(mask,1);
	temp->Write2File("../results/gtm",6);
	tree2elemental(temp, x);

	return 0;
}



/*
 * This function applies the operator U, which is almost exactly like the
 * operator for G*
 */
int U_func2(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, U_data *u_data){
	// u  is the vector containing the random coefficients for each of the point sources

	InvMedTree<FMM_Mat_t>* mask = u_data->mask;
	InvMedTree<FMM_Mat_t>* temp = u_data->temp;
	MPI_Comm comm               = u_data->comm;
	pvfmm::PtFMM_Tree* pt_tree  = u_data->pt_tree;
	int trg_coord_size          = u_data->trg_coord_size;

	int rank;
	MPI_Comm_rank(comm, &rank);

	if(!rank) std::cout << "U function" << std::endl;

	int n_local_pt_srcs = u_data->n_local_pt_srcs;
	std::vector<double> coeffs(n_local_pt_srcs*2);
	elemental2vec(x,coeffs);

	//std::vector<double> trg_coord = temp->ChebPoints();


	std::vector<double> trg_value;
	pt_tree->ClearFMMData();
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value, trg_coord_size,&coeffs);

	temp->Trg2Tree(trg_value);

	// convert the tree into a vector. This vector represents the function
	// that we passed into the tree constructor (which contains the current 
	// random coefficients).
	//temp->Write2File("../results/ufunc_b",8);
	temp->Multiply(mask,1);
	//temp->Write2File("../results/ufunc",8);
	tree2elemental(temp,y);

	MPI_Barrier(comm);

	return 0;
}

/*
 * This fucntion applies the operator for the U*, which is very similar to
 * the operator given by G
 */
int Ut_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, U_data *u_data){


	if((x.ColAlign() != y.ColAlign()) or (x.RowAlign() != y.RowAlign())){
		El::LogicError("x and y must have the same distribution");
	}

	InvMedTree<FMM_Mat_t>* temp_c = u_data->temp_c;
	InvMedTree<FMM_Mat_t>* mask = u_data->mask;
	std::vector<double> src_coord = u_data->src_coord;
	int rank;
	MPI_Comm_rank(u_data->comm,&rank);
	if(!rank) std::cout << "U* function" << std::endl;

	// get the input data
	elemental2tree(y,temp_c);
	temp_c->Multiply(mask,1);

	// integrate
	temp_c->ClearFMMData();
	temp_c->RunFMM();
	temp_c->Copy_FMMOutput();

	// read at srcs
	std::vector<double> src_values = temp_c->ReadVals(src_coord);
	//if(!rank){
	//	std::cout << src_values[0] << std::endl;
	//	std::cout << src_values[1] << std::endl;
	//}

	vec2elemental(src_values,x);	

	return 0;
}



/*
 * The B operators... I don't think are to be trusted at this point. I've been
 * neglecting them until I get the other stuff working correctly.
 */
int B_func(
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *S_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *V_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *W,
		InvMedTree<FMM_Mat_t>* temp,
		InvMedTree<FMM_Mat_t>* temp_c
		)
{
	int R_s = W->Width();
	int R_d = V_G->Width();
	int N_disc = W->Height();

	const El::Grid& g = x.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int rank;
	MPI_Comm_rank(comm,&rank);
	if(!rank) std::cout << "B function" << std::endl;

	elemental2tree(x,temp_c);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Wx(g);
	El::Zeros(Wx,N_disc,R_s);

	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> W_i = El::View(*W, 0, i, N_disc, 1);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Wx_i = El::View(Wx, 0, i, N_disc, 1);
		elemental2tree(W_i,temp);
		temp->Multiply(temp_c,1);
		tree2elemental(temp,Wx_i);
	}

	El::Complex<double> alpha = El::Complex<double>(1.0);
	El::Complex<double> beta  = El::Complex<double>(0.0);


	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VtWx(g);
	El::Zeros(VtWx,R_d,R_s);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,*V_G,Wx,beta,VtWx);

	El::DiagonalScale(El::LEFT,El::NORMAL,*S_G,VtWx);

	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> SVtWx2(g);
	SVtWx2 = VtWx;
	SVtWx2.Resize(R_s*R_d,1);
	VtWx = SVtWx2;
	y = VtWx;

	return 0;
}

/* 
 * Same as above
 */
int Bt_func(
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *S_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *V_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *W,
		InvMedTree<FMM_Mat_t>* temp,
		InvMedTree<FMM_Mat_t>* temp_c,
		InvMedTree<FMM_Mat_t>* temp2
		)
{

	const El::Grid& g = x.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;

	El::Complex<double> alpha = El::Complex<double>(1.0);
	El::Complex<double> beta  = El::Complex<double>(0.0);

	int rank;
	MPI_Comm_rank(comm,&rank);
	if(!rank) std::cout << "B* function" << std::endl;

	int R_s = W->Width();
	int R_d = V_G->Width();
	int N_disc = W->Height();

	//El::Display(x,"Bt vec");
	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> x2(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x_copy(g);
	El::Copy(x,x_copy);
	x2 = x_copy;
	x2.Resize(R_d,R_s);
	x_copy = x2;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Sx(g);
	El::Zeros(Sx,R_d,R_s);
	El::DiagonalScale(El::LEFT,El::ADJOINT,*S_G,x_copy);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VSx(g);
	El::Zeros(VSx,N_disc,R_s);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,*V_G,x_copy,beta,VSx);


	/*
	InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
	t->bndry = temp->bndry;
	t->kernel = temp->kernel;
	t->fn = zero_fn;
	t->f_max = 4;
	t->CreateTree(false);
	*/
	temp2->Zero();


	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VSx_i = El::View(VSx, 0, i, N_disc, 1);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> W_i = El::View(*W, 0, i, N_disc, 1);
		elemental2tree(W_i,temp_c);
		elemental2tree(VSx_i,temp);
		temp->ConjMultiply(temp_c,1);
		temp2->Add(temp,1);
	}
	tree2elemental(temp2,y);

	//delete t;

	return 0;
}
