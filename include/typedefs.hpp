#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

//#include <pvfmm_common.hpp>

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

#endif
