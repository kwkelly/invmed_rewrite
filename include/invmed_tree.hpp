#ifndef INVMED_TREE_HPP
#define INVMED_TREE_HPP

#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <petscksp.h>
#include <cassert>
#include <cstring>
#include <profile.hpp>
#include <mpi.h>
#include<set>

//typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
//typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
//typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;

template <class FMM_Mat_t>
class InvMedTree : public pvfmm::FMM_Tree<FMM_Mat_t>{
  public:
  typedef typename FMM_Mat_t::FMMNode_t FMM_Node_t;
  typedef typename FMM_Node_t::Node_t Node_t;
  typedef typename FMM_Mat_t::Real_t Real_t;

	typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;
  const pvfmm::Kernel<double>* kernel;
  FMM_Mat_t* fmm_mat;
  pvfmm::BoundaryType bndry;
  int mult_order;
  int cheb_deg;
  bool adap;
  double tol;
	int mindepth;
	int maxdepth;
	void (*fn)(double* coord, int n, double* out);
	double f_max;
	int dim;
	int data_dof;
	PetscInt m,M,n,N,l,L;
	static std::set< InvMedTree* > m_instances;
	
  //typedef typename FMMNode_t::NodeData tree_data;

  InvMedTree<FMM_Mat_t>(MPI_Comm c) : pvfmm::FMM_Tree<FMM_Mat_t>(c){
		InvMedTree::m_instances.insert(this);
	};
  virtual ~InvMedTree(){
		InvMedTree::m_instances.erase(this);		
	};
	void Initialize();
	void InitializeMat();
  static void Copy(InvMedTree<FMM_Mat_t> &new_tree, const InvMedTree<FMM_Mat_t> &other);
  static void Add(InvMedTree<FMM_Mat_t> &tree1, InvMedTree<FMM_Mat_t> &tree2);
  static void Multiply(InvMedTree<FMM_Mat_t> &tree1, InvMedTree<FMM_Mat_t> &tree2);
  static void CreateDiffFunction(InvMedTree<FMM_Mat_t> &tree1, InvMedTree<FMM_Mat_t> &tree2);

	std::vector<pvfmm::FMM_Node<pvfmm::Cheb_Node<double> >* > GetNGLNodes();
};

#include "invmed_tree.txx"

#endif
