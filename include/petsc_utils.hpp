#ifndef PETSC_UTILS_HPP
#define PETSC_UTILS_HPP
#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <pvfmm_common.hpp>
#include "typedefs.hpp"

namespace petsc_utils{

	struct InvMedData{
		InvMedTree<FMM_Mat_t>* phi_0;
		InvMedTree<FMM_Mat_t>* temp;
	};

	#undef __FUNCT__
	#define __FUNCT__ "mult"
	int mult(Mat M, Vec U, Vec Y);


	#undef __FUNCT__
	#define __FUNCT__ "tree2vec"
	template <class FMM_Mat_t>
	int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y);

	#undef __FUNCT__
	#define __FUNCT__ "vec2tree"
	template <class FMM_Mat_t>
	int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree);

#include "petsc_utils.txx"
}
#endif
