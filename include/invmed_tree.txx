#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <mpi.h>
#include <profile.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <iterator>

template <class FMM_Mat_t>
int InvMedTree<FMM_Mat_t>::mult_order;
template <class FMM_Mat_t>
int InvMedTree<FMM_Mat_t>::cheb_deg;
template <class FMM_Mat_t>
bool InvMedTree<FMM_Mat_t>::adap;
template <class FMM_Mat_t>
double InvMedTree<FMM_Mat_t>::tol;
template <class FMM_Mat_t>
int InvMedTree<FMM_Mat_t>::mindepth;
template <class FMM_Mat_t>
int InvMedTree<FMM_Mat_t>::maxdepth;
template <class FMM_Mat_t>
int InvMedTree<FMM_Mat_t>::dim;
template <class FMM_Mat_t>
int InvMedTree<FMM_Mat_t>::data_dof;

template <class FMM_Mat_t>
std::set< InvMedTree<FMM_Mat_t>* > InvMedTree<FMM_Mat_t>::m_instances;
//typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
//typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
//typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::SetupInvMed(){

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typename FMMNode_t::NodeData tree_data;
	//Various parameters.

		int myrank, np;
		MPI_Comm_rank(*((*(InvMedTree::m_instances.begin()))->Comm()), &myrank);
		MPI_Comm_size(*((*(InvMedTree::m_instances.begin()))->Comm()),&np);
	// Set original source coordinates on a regular grid 
	// at minepth with one point per octree node.
	// This gets refined when FMM_Init is called with adaptivity on.
	// We then use the new generated coordinates as the original sources
	// for a new tree, and so on and so forth. Once all trees have been generated
	// we create all new ones without adaptivity so that they all have the same structure.

	std::vector<double> pt_coord;// = InvMedTree<FMM_Mat_t>::glb_pt_coord;
	{ 
		size_t NN=ceil(pow((double)np,1.0/3.0));
		NN=std::max<size_t>(NN,pow(2.0,InvMedTree<FMM_Mat_t>::mindepth));
		size_t N_total=NN*NN*NN;
		size_t start= myrank   *N_total/np;
		size_t end  =(myrank+1)*N_total/np;
		for(size_t i=start;i<end;i++){
			pt_coord.push_back(((double)((i/  1    )%NN)+0.5)/NN);
			pt_coord.push_back(((double)((i/ NN    )%NN)+0.5)/NN);
			pt_coord.push_back(((double)((i/(NN*NN))%NN)+0.5)/NN);
		}
		tree_data.pt_coord=pt_coord;
	}

	std::cout << "pt_coord len: " << pt_coord.size() << std::endl;

	tree_data.max_pts=1; // Points per octant.
	tree_data.dim=InvMedTree<FMM_Mat_t>::dim;
	tree_data.max_depth=InvMedTree<FMM_Mat_t>::maxdepth;
	tree_data.cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	tree_data.data_dof=InvMedTree<FMM_Mat_t>::data_dof;


	typename std::set<InvMedTree<FMM_Mat_t>* >::iterator it; // not sure why this is here...
	for (it = InvMedTree::m_instances.begin(); it!=InvMedTree::m_instances.end(); ++it){
		//Set input function pointer
		tree_data.input_fn=(*it)->fn;
		tree_data.tol=(InvMedTree<FMM_Mat_t>::tol)*((*it)->f_max);


		//Create Tree and initialize with input data.
		(*it)->Initialize(&tree_data);
		(*it)->InitFMM_Tree(InvMedTree<FMM_Mat_t>::adap,(*it)->bndry);
		std::cout << ((*it)->GetNodeList()).size() << std::endl;

		// This loop gets the new coordinates or the centers of all the nodes in the octree,
		// replacing the old starting coordinates.
		pt_coord.clear();
		std::vector<FMMNode_t*> nlist=(*it)->GetNodeList();
		for(size_t i=0;i<nlist.size();i++){
			if(nlist[i]->IsLeaf() && !nlist[i]->IsGhost()){
				double s=pow(0.5,nlist[i]->Depth()+1);
				double* c=nlist[i]->Coord();
				pt_coord.push_back(c[0]+s);
				pt_coord.push_back(c[1]+s);
				pt_coord.push_back(c[2]+s);
			}
		}
		tree_data.pt_coord=pt_coord;

/* Not actually sure what this does.
		{ //Output max tree depth.
			std::vector<size_t> all_nodes(InvMedTree<FMM_Mat_t>::maxdepth+1,0);
			std::vector<size_t> leaf_nodes(InvMedTree<FMM_Mat_t>::maxdepth+1,0);
			std::vector<FMMNode_t*>& nodes=InvMedTree<FMM_Mat_t>::GetNodeList();
			for(size_t i=0;i<nodes.size();i++){
				FMMNode_t* n=nodes[i];
				if(!n->IsGhost()) all_nodes[n->Depth()]++;
				if(!n->IsGhost() && n->IsLeaf()) leaf_nodes[n->Depth()]++;
			}

			if(!myrank) std::cout<<"All  Nodes: ";
			for(int i=0;i<InvMedTree<FMM_Mat_t>::maxdepth;i++){
				int local_size=all_nodes[i];
				int global_size;
				MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, *((*it)->Comm()));
				if(global_size==0) InvMedTree<FMM_Mat_t>::maxdepth=i;
				if(!myrank) std::cout<<global_size<<' ';
			}
			if(!myrank) std::cout<<'\n';

			if(!myrank) std::cout<<"Leaf Nodes: ";
			for(int i=0;i<InvMedTree<FMM_Mat_t>::maxdepth;i++){
				int local_size=leaf_nodes[i];
				int global_size;
				MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, *((*it)->Comm()));
				if(!myrank) std::cout<<global_size<<' ';
			}
			if(!myrank) std::cout<<'\n';
		}
*/
		std::cout << "here" << std::endl;


		int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
		size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
		size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
		{ // Get local and global size
			long long loc_size=0, glb_size=0;
			long long loc_nodes=0, glb_nodes=0;
			std::vector<FMMNode_t*> nlist=(*it)->GetNodeList();
			for(size_t i=0;i<nlist.size();i++){
				if(nlist[i]->IsLeaf() && !nlist[i]->IsGhost()){
					loc_size+=n_coeff3; //nlist[i]->ChebData().Dim();
					loc_nodes+=n_nodes3;
				}
			}
			MPI_Allreduce(&loc_size, &glb_size, 1, MPI_LONG_LONG , MPI_SUM, *((*it)->Comm()));
			MPI_Allreduce(&loc_nodes, &glb_nodes, 1, MPI_LONG_LONG, MPI_SUM, *((*it)->Comm()));
			(*it)->n=loc_size*(*it)->kernel->ker_dim[0];
			(*it)->N=glb_size*(*it)->kernel->ker_dim[0];
			(*it)->m=loc_size*(*it)->kernel->ker_dim[1];
			(*it)->M=glb_size*(*it)->kernel->ker_dim[1];
			(*it)->l=loc_nodes*(*it)->kernel->ker_dim[0];
			(*it)->L=glb_nodes*(*it)->kernel->ker_dim[0];
		}
	}

	// Now we loop through all the trees again, reinitialize them without adaptivity so that 
	// all of the trees have the structure given be the common adaptively selected coordinates of
	// them all.
	InvMedTree<FMM_Mat_t>::adap = false;
	for (it = InvMedTree::m_instances.begin(); it!=InvMedTree::m_instances.end(); ++it){
		//Set input function pointer
		tree_data.input_fn=(*it)->fn;
		tree_data.tol=(InvMedTree<FMM_Mat_t>::tol)*((*it)->f_max);

		//Create Tree and initialize with input data.
		(*it)->Initialize(&tree_data);
		(*it)->InitFMM_Tree(InvMedTree<FMM_Mat_t>::adap,(*it)->bndry);
		std::cout << ((*it)->GetNodeList()).size() << std::endl;
	}
	return;
}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Multiply(InvMedTree* other, double multiplier){

	//  may need to change this
	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=this->Comm();
	int omp_p=omp_get_max_threads();
	//int omp_p = 1;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	int data_dof=InvMedTree<FMM_Mat_t>::data_dof;

	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);

	std::vector<FMMNode_t*> nlist1 = this->GetNGLNodes();
	std::vector<FMMNode_t*> nlist2 = other->GetNGLNodes();

	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
	#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist1.size()* tid   )/omp_p;
		size_t i_end  =(nlist1.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> coeff_vec1(n_coeff3*data_dof);
		pvfmm::Vector<double> coeff_vec2(n_coeff3*data_dof);
		pvfmm::Vector<double> val_vec1(n_nodes3*data_dof);
		pvfmm::Vector<double> val_vec2(n_nodes3*data_dof);
		//	std::cout << "val_vec2.Dim() " << val_vec2.Dim() << std::endl;
		for(size_t i=i_start;i<i_end;i++){
			double s=std::pow(2.0,COORD_DIM*nlist1[i]->Depth()*0.5*SCAL_EXP);
			coeff_vec1 = nlist1[i]->ChebData();
			coeff_vec2 = nlist2[i]->ChebData();

			// val_vec: Evaluate coeff_vec at Chebyshev node points
			cheb_eval(coeff_vec1, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec1);
			cheb_eval(coeff_vec2, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec2);
			//			std::cout << "dim :" << val_vec2.Dim() << std::endl;
			//		std::cout << "dim :" << val_vec1.Dim() << std::endl;


			for(size_t j0=0;j0<n_nodes3;j0++){
				//for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					//real*real - im*im
					val_vec1[0*n_nodes3+j0]=multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]-val_vec1[1*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]);
					// real*im + im*real
					val_vec1[1*n_nodes3+j0]=multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]+val_vec1[1*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]);
			//	}
			}


			{ // Compute Chebyshev approx
				pvfmm::Vector<double>& coeff_vec=nlist1[i]->ChebData();
				if(coeff_vec.Dim()!=(data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6){
					coeff_vec.ReInit((data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
				}
				pvfmm::cheb_approx<double,double>(&val_vec1[0], cheb_deg, data_dof, &coeff_vec[0]);
				nlist1[i]->DataDOF()=data_dof;
			}
		}
	}

	return;
}

	
template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::ScalarMultiply(double multiplier){

	//  may need to change this
	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=this->Comm();
	int omp_p=omp_get_max_threads();
	//int omp_p = 1;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	int data_dof=InvMedTree<FMM_Mat_t>::data_dof;

	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);

	std::vector<FMMNode_t*> nlist1 = this->GetNGLNodes();

	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
	#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist1.size()* tid   )/omp_p;
		size_t i_end  =(nlist1.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> coeff_vec1(n_coeff3*data_dof);
		pvfmm::Vector<double> val_vec1(n_nodes3*data_dof);
		//	std::cout << "val_vec2.Dim() " << val_vec2.Dim() << std::endl;
		for(size_t i=i_start;i<i_end;i++){
			double s=std::pow(2.0,COORD_DIM*nlist1[i]->Depth()*0.5*SCAL_EXP);
			coeff_vec1 = nlist1[i]->ChebData();

			// val_vec: Evaluate coeff_vec at Chebyshev node points
			cheb_eval(coeff_vec1, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec1);

			for(size_t j0=0;j0<n_nodes3;j0++){
				for(size_t j1=0;j1<data_dof;j1++){
					val_vec1[j1*n_nodes3+j0]*=multiplier
				}
			}


			{ // Compute Chebyshev approx
				pvfmm::Vector<double>& coeff_vec=nlist1[i]->ChebData();
				if(coeff_vec.Dim()!=(data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6){
					coeff_vec.ReInit((data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
				}
				pvfmm::cheb_approx<double,double>(&val_vec1[0], cheb_deg, data_dof, &coeff_vec[0]);
				nlist1[i]->DataDOF()=data_dof;
			}
		}
	}

	return;
}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Add(InvMedTree* other, double multiplier){

	//  may need to change this
	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=this->Comm();
	int omp_p=omp_get_max_threads();
	//int omp_p = 1;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	int data_dof=InvMedTree<FMM_Mat_t>::data_dof;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)/6;

	std::vector<FMMNode_t*> nlist1 = this->GetNGLNodes();
	std::vector<FMMNode_t*> nlist2 = other->GetNGLNodes();

	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
	#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist1.size()* tid   )/omp_p;
		size_t i_end  =(nlist1.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> coeff_vec1(n_coeff3*data_dof);
		pvfmm::Vector<double> coeff_vec2(n_coeff3*data_dof);
		pvfmm::Vector<double> val_vec1(n_nodes3*data_dof);
		pvfmm::Vector<double> val_vec2(n_nodes3*data_dof);

		for(size_t i=i_start;i<i_end;i++){
			double s=std::pow(2.0,COORD_DIM*nlist1[i]->Depth()*0.5*SCAL_EXP);
			coeff_vec1 = nlist1[i]->ChebData();
			coeff_vec2 = nlist2[i]->ChebData();

			// val_vec: Evaluate coeff_vec at Chebyshev node points
			cheb_eval(coeff_vec1, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec1);
			cheb_eval(coeff_vec2, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec2);

			for(size_t j0=0;j0<n_nodes3;j0++){
				for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					val_vec1[j1*n_nodes3+j0]+=multiplier*(val_vec2[j1*n_nodes3+j0]);
				}
			}

			{ // Compute Chebyshev approx
				pvfmm::Vector<double>& coeff_vec=nlist1[i]->ChebData();
				if(coeff_vec.Dim()!=(data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6){
					coeff_vec.ReInit((data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
				}
				pvfmm::cheb_approx<double,double>(&val_vec1[0], cheb_deg, data_dof, &coeff_vec[0]);
				nlist1[i]->DataDOF()=data_dof;
			}
		}
	}

	return;
}

template<class FMM_Mat_t>
std::vector<pvfmm::FMM_Node<pvfmm::Cheb_Node<double> >* > InvMedTree<FMM_Mat_t>::GetNGLNodes(){
	std::vector<pvfmm::FMM_Node<pvfmm::Cheb_Node<double> >*> nlist;
	{ 
		std::vector<FMM_Node_t*>& nlist_=this->GetNodeList();
		for(size_t i=0;i<nlist_.size();i++){
			if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
				nlist.push_back(nlist_[i]);
			}
		}
	}
	
	return nlist;
}


template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::InitializeMat(){
	this->fmm_mat = new FMM_Mat_t;
  this->fmm_mat->Initialize(this->mult_order,this->cheb_deg,*(this->Comm()),this->kernel);

	return;
}

template <class FMM_Mat_t>
void CreateNewTree(){
	// TODO
	//
	return;
}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Copy(InvMedTree<FMM_Mat_t>* other){

	//  may need to change this
	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=this->Comm();

	typename FMMNode_t::NodeData tree_data;
	tree_data.max_pts=1; // Points per octant.
	tree_data.dim=InvMedTree<FMM_Mat_t>::dim;
	tree_data.max_depth=InvMedTree<FMM_Mat_t>::maxdepth;
	tree_data.cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	tree_data.data_dof=InvMedTree<FMM_Mat_t>::data_dof;

	// Copy data over from old tree
	this->f_max = other->f_max;
	this->bndry = other->bndry;
	this->fn    = other->fn;
	tree_data.input_fn=this->fn;
	tree_data.tol=(InvMedTree<FMM_Mat_t>::tol)*(this->f_max);

	// Get point coords of other
	std::vector<double> pt_coord;
	pt_coord.clear();
	std::vector<FMMNode_t*> nlist=other->GetNodeList();
	for(size_t i=0;i<nlist.size();i++){
		if(nlist[i]->IsLeaf() && !nlist[i]->IsGhost()){
			double s=pow(0.5,nlist[i]->Depth()+1);
			double* c=nlist[i]->Coord();
			pt_coord.push_back(c[0]+s);
			pt_coord.push_back(c[1]+s);
			pt_coord.push_back(c[2]+s);
		}
	}
	tree_data.pt_coord=pt_coord;

	InvMedTree<FMM_Mat_t>::adap = false;
	//Create Tree and initialize with input data.
	this->Initialize(&tree_data);
	this->InitFMM_Tree(InvMedTree<FMM_Mat_t>::adap,this->bndry);
	std::cout << (this->GetNodeList()).size() << std::endl;


	//int omp_p=omp_get_max_threads();
	int omp_p = 1;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	int data_dof=InvMedTree<FMM_Mat_t>::data_dof;

	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);

	std::vector<FMMNode_t*> nlist1 = this->GetNGLNodes();
	std::vector<FMMNode_t*> nlist2 = other->GetNGLNodes();

	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
	#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist1.size()* tid   )/omp_p;
		size_t i_end  =(nlist1.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> coeff_vec1(n_coeff3*data_dof);
		pvfmm::Vector<double> coeff_vec2(n_coeff3*data_dof);
		pvfmm::Vector<double> val_vec1(n_nodes3*data_dof);
		pvfmm::Vector<double> val_vec2(n_nodes3*data_dof);
		//	std::cout << "val_vec2.Dim() " << val_vec2.Dim() << std::endl;
		for(size_t i=i_start;i<i_end;i++){
			double s=std::pow(2.0,COORD_DIM*nlist1[i]->Depth()*0.5*SCAL_EXP);
			coeff_vec1 = nlist1[i]->ChebData();
			coeff_vec2 = nlist2[i]->ChebData();

			// val_vec: Evaluate coeff_vec at Chebyshev node points
			cheb_eval(coeff_vec1, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec1);
			cheb_eval(coeff_vec2, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec2);

			for(size_t j0=0;j0<n_nodes3;j0++){
				for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					val_vec1[j1*n_nodes3+j0]=val_vec2[j1*n_nodes3+j0];
					//std::cout << val_vec1[j1*n_nodes3+j0] << " : " << val_vec2[j1*n_nodes3+j0] << std::endl;
				}
			}


			{ // Compute Chebyshev approx
				pvfmm::Vector<double>& coeff_vec=nlist1[i]->ChebData();
				if(coeff_vec.Dim()!=(data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6){
					coeff_vec.ReInit((data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
				}
				pvfmm::cheb_approx<double,double>(&val_vec1[0], cheb_deg, data_dof, &coeff_vec[0]);
				nlist1[i]->DataDOF()=data_dof;
			}
		}
	}

	return;
}

