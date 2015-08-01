#ifndef INVMED_TREE_HPP
#define INVMED_TREE_HPP

#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <pvfmm.hpp>
#include <cassert>
#include <cstring>
#include <profile.hpp>
#include <mpi.h>
#include <set>

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
	void (*fn)(const  double* coord, int n, double* out);
	double f_max;
	bool is_initialized;

	long long m,M,n,N,l,L;
	long long loc_octree_nodes;
	long long glb_octree_nodes;
	long long previous_octree_nodes;
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
		this->is_initialized = false;
		InvMedTree::m_instances.insert(this);
	};

  InvMedTree<FMM_Mat_t>(void (*fn_)(const  double* coord, int n, double* out), double fmax_, const pvfmm::Kernel<double>* kernel_, pvfmm::BoundaryType bndry_ = pvfmm::FreeSpace , MPI_Comm c = MPI_COMM_WORLD) : pvfmm::FMM_Tree<FMM_Mat_t>(c){
		this->fn = fn_;
		this->f_max = fmax_;
		this->kernel=kernel_;
		this->bndry=bndry_;
		this->is_initialized = false;
		InvMedTree::m_instances.insert(this);
	};

  virtual ~InvMedTree(){
		InvMedTree::m_instances.erase(this);		
	};
	static void SetupInvMed();
	void InitializeMat();
  void Add(InvMedTree<FMM_Mat_t>* other, double multiplier);
  void Multiply(InvMedTree<FMM_Mat_t>* other, double multiplier);
  void ConjMultiply(InvMedTree<FMM_Mat_t>* other, double multiplier);
	void ScalarMultiply(double multiplier);
  void CreateTree(bool adap);
	void Copy(InvMedTree<FMM_Mat_t>* other);
	void FilterChebTree(std::vector<double>& coeff_scaling);
	void Zero();
	void FMMSetup();
	pvfmm::PtFMM_Tree* CreatePtFMMTree(std::vector<double> &src_coord, std::vector<double> &src_value, const pvfmm::Kernel<double>* kernel);
	void Trg2Tree(std::vector<double> &trg_value);
	std::vector<double> ReadVals(std::vector<double> &coord);
	static void SetSrcValues(const std::vector<double> coords, const std::vector<double> values, pvfmm::PtFMM_Tree* tree);
  double Norm2();
  double Norm2c();
	std::vector<double> Integrate();
	std::vector<double> ChebPoints();
	std::vector<pvfmm::FMM_Node<pvfmm::Cheb_Node<double> >* > GetNGLNodes();
};

#include "invmed_tree.txx"

#endif
