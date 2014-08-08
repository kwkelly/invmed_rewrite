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
	void (*fn)(double* coord, int n, double* out);
	double f_max;

	PetscInt m,M,n,N,l,L;
	static std::set< InvMedTree* > m_instances;
	static std::vector<double> glb_pt_coord;
  static int mult_order;
  static int cheb_deg;
  static bool adap;
  static double tol;
	static int mindepth;
	static int maxdepth;
	static int dim;
	static int data_dof;

  //typedef typename FMMNode_t::NodeData tree_data;

  InvMedTree<FMM_Mat_t>(MPI_Comm c) : pvfmm::FMM_Tree<FMM_Mat_t>(c){
		InvMedTree::m_instances.insert(this);
	};
  virtual ~InvMedTree(){
		InvMedTree::m_instances.erase(this);		
	};
	static void SetupInvMed();
	void InitializeMat();
  void Add(InvMedTree<FMM_Mat_t>* other, double multiplier);
  void Multiply(InvMedTree<FMM_Mat_t>* other, double multiplier);
	void ScalarMultiply(double multiplier);
  void CreateNewTree();
	void Copy(InvMedTree<FMM_Mat_t>* other);

	std::vector<pvfmm::FMM_Node<pvfmm::Cheb_Node<double> >* > GetNGLNodes();
};

#include "invmed_tree.txx"

#endif
