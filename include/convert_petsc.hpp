#include "petscsys.h"
#include "invmed_tree.hpp"
#include "pvfmm.hpp"

#pragma once

template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y);

template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree);
