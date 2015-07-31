#include "El.hpp"
#include "invmed_tree.hpp"
#include <vector>

#pragma once

/*
 * Convert a pvfmm tree to en elemental vector whose entries are the Chebyshev coefficients
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */

template <class FMM_Mat_t, typename T>
int tree2elemental(InvMedTree<FMM_Mat_t> *tree, El::DistMatrix<T,El::VC,El::STAR> &Y);

/*
 * Convert an Elemental vector to a pvfmm tree. The opposite of the previous function
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */
template <class FMM_Mat_t, typename T>
int elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, InvMedTree<FMM_Mat_t> *tree);

/*
 * Convert a std::vector to an elemental vector of the given distribution. As before, thise well reorder the data
 */
int vec2elemental(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::VC,El::STAR > &Y);

/*
 * Convert an elemental vector to a std::vector of the given distribution. As before, thise well reorder the data
 */
int elemental2vec(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, std::vector<double> &vec);

/*
 * Convert an elemental STAR, STAR distributed vector toa  std::vector
 */
int elstar2vec(const El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> &Y, std::vector<double> &vec);

/*
 * Convert a std::vector to an elemental STAR, STAR distributed vector
 */
int vec2elstar(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::STAR,El::STAR > &Y);

/*
 * the sum op for a scan operation used in the below function.
 */
template <typename T>
void op(T& v1, const T& v2);

/*
 * Due to the different sizes and distributions of the data between pvfmm and elemental, we need to calculate how much data and which data is going to be sent to each plave
 */
int comp_alltoall_sizes(const std::vector<int> &input_sizes, const std::vector<int> &output_sizes, std::vector<int> &sendcnts, std::vector<int> &sdispls, std::vector<int> &recvcnts, std::vector<int> &rdispls, MPI_Comm comm);
