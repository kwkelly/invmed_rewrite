#include <mpi.h>
#include "invmed_tree.hpp"
#include "pvfmm.hpp"
#include "El.hpp"

#pragma once
typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

class G_op{
  public:
		G_op(std::vector<double> detector_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double> *kernel, pvfmm::BoundaryType bndry = pvfmm::FreeSpace, MPI_Comm comm = MPI_COMM_WORLD);
		~G_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &y);
	private:
		pvfmm::BoundaryType bndry;
		InvMedTree<FMM_Mat_t>* mask;
		InvMedTree<FMM_Mat_t>* temp;
		std::vector<double> det_coord;
		MPI_Comm comm;
};

class Gt_op{
  public:
		Gt_op(std::vector<double> detector_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double> *kernel, MPI_Comm comm = MPI_COMM_WORLD);
		~Gt_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &x);
	private:
		pvfmm::BoundaryType bndry;
		InvMedTree<FMM_Mat_t>* mask;
		InvMedTree<FMM_Mat_t>* temp;
		std::vector<double> detector_coord;
		std::vector<double> trg_coord;
		MPI_Comm comm;
		pvfmm::PtFMM_Tree* pt_tree = NULL;
		pvfmm::PtFMM* matrices = NULL;
		int local_cheb_points;
		std::vector<double> detector_values;
};

