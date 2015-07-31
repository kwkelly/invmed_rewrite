#include "invmed_tree.hpp"
#include "pvfmm.hpp"
#include "El.hpp"

#pragma once

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

/*
 * Some structs with data that we need for the U and G operators
 */
struct G_data{
	InvMedTree<FMM_Mat_t>* mask;
	InvMedTree<FMM_Mat_t>* temp;
	pvfmm::PtFMM_Tree* pt_tree;
	std::vector<double> src_coord;
	bool filter;
	MPI_Comm comm;
};

struct U_data{
	InvMedTree<FMM_Mat_t>* mask;
	InvMedTree<FMM_Mat_t>* temp;
	InvMedTree<FMM_Mat_t>* temp_c;
	std::vector<double> src_coord;
	pvfmm::BoundaryType bndry;
	const pvfmm::Kernel<double>* kernel;
	void (*fn)(const  double* coord, int n, double* out);
	std::vector<double> *coeffs;
	int trg_coord_size;
	MPI_Comm comm;
	pvfmm::PtFMM_Tree* pt_tree;
	int n_local_pt_srcs;
};


/*
 * This function applies the operator G, which consists of a pointwise
 * multiplication with a mask function, a volume fmm evaluation and then
 * sampling values at predefined points
 */
int G_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, G_data *g_data);

/*
 * This function is, of course, the operator G*, which is a particle fmm (at
 * the detector location) evaluated everywhere, which is then converted to an
 * fmm tree, multiplied pointwise by a mask and returned as an elemental
 * vector
 */
int Gt_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, G_data *g_data);

/*
 * This function applies the operator U, which is almost exactly like the
 * operator for G*
 */
int U_func2(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, U_data *u_data);

/*
 * This fucntion applies the operator for the U*, which is very similar to
 * the operator given by G
 */
int Ut_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, U_data *u_data);

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
		);

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
		);
