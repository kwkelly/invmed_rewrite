#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <mpi.h>
#include <profile.hpp>
#include <pvfmm.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <iterator>
#include <iomanip>

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
//		std::cout << "here" << std::endl;


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

			double temp_real;
			double temp_im;
			for(size_t j0=0;j0<n_nodes3;j0++){
				//for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					//real*real - im*im
					temp_real = multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]-val_vec1[1*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]);
					//val_vec1[0*n_nodes3+j0]=multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]-val_vec1[1*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]);
					// real*im + im*real
					temp_im = multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]+val_vec1[1*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]);
					//val_vec1[1*n_nodes3+j0]=multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]+val_vec1[1*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]);
					//std::cout << temp_real << std::endl;
					//std::cout << temp_im << std::endl;
					val_vec1[0*n_nodes3+j0] = temp_real;
					val_vec1[1*n_nodes3+j0] = temp_im;
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
					val_vec1[j1*n_nodes3+j0]*=multiplier;
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

	double max = 0;

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

			for(size_t j1=0;j1<data_dof;j1++){
				for(size_t j0=0;j0<n_nodes3;j0++){
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

template <class FMM_Mat_t>
pvfmm::PtFMM_Tree* InvMedTree<FMM_Mat_t>::CreatePtFMMTree(std::vector<double> &src_coord, std::vector<double> &src_value, const pvfmm::Kernel<double>* kernel){

	int SCAL_EXP = 1;
	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm_const=this->Comm();
	MPI_Comm* comm = const_cast<MPI_Comm*>(comm_const);
	///////////////
	// Get trg_coord. These are the locations of the chebyshev nodes in each of the nodes of the octree.
	///////////////
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int mult_order = InvMedTree<FMM_Mat_t>::mult_order;
	std::vector<double> cheb_node_coord3=pvfmm::cheb_nodes<double>(cheb_deg, 3);
	//for(int i=0)
	size_t n_chebnodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	
	std::vector<double> trg_coord;
	std::vector<FMMNode_t*> nlist=this->GetNodeList();
	for(size_t i=0;i<nlist.size();i++){
		if(nlist[i]->IsLeaf() && !nlist[i]->IsGhost()){

			double s=pow(0.5,nlist[i]->Depth());
			double* c=nlist[i]->Coord();
			for(int j=0;j<n_chebnodes3*3;j=j+3){
				trg_coord.push_back(c[0] + cheb_node_coord3[j+0]*s);
				trg_coord.push_back(c[1] + cheb_node_coord3[j+1]*s);
				trg_coord.push_back(c[2] + cheb_node_coord3[j+2]*s);
			}
		}
	}

	// Now we can create the new octree
  pvfmm::PtFMM_Tree* pt_tree=pvfmm::PtFMM_CreateTree(src_coord, src_value, trg_coord, *comm );
	std::cout << "after new tree" << std::endl;
  // Load matrices.
  pvfmm::PtFMM* matrices = new pvfmm::PtFMM;
	std::cout << "after matrices" << std::endl;
  matrices->Initialize(mult_order, *comm, kernel);
	std::cout << "after init" << std::endl;

  // FMM Setup
  pt_tree->SetupFMM(matrices);
	std::cout << "after setup" << std::endl;

	return pt_tree;
}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Trg2Tree(std::vector<double> &trg_value){

	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=this->Comm();
	int omp_p=omp_get_max_threads();
	//int omp_p = 1;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	int data_dof=InvMedTree<FMM_Mat_t>::data_dof;

	int n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	int n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	
	std::vector<FMMNode_t*> nlist = this->GetNGLNodes();
	
	#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist.size()* tid   )/omp_p;
		size_t i_end  =(nlist.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> val_vec(n_nodes3*data_dof);
		//	std::cout << "val_vec2.Dim() " << val_vec2.Dim() << std::endl;
		for(size_t i=i_start;i<i_end;i++){
			for(size_t j0=0;j0<n_nodes3;j0++){
					// In a given node of the octree, trg_value has the real and imaginary parts
					// next to each other. (real, imag, real, imag) for each point that we evaluated
					// at. val_vec needs to have all the real parts, then all the imaginary parts.
					// (real, real, real, imag, imag imag). Thus we have to reorder on insert.
					// Get real part and then place it in the first n_nodes part of val_vec
					val_vec[0*n_nodes3+j0] = trg_value[i*n_nodes3*data_dof+j0*data_dof + 0];

					//std::cout <<  trg_value[i*n_nodes3*data_dof+j0*data_dof + 0] << std::endl;
					//std::cout << i*n_nodes3*data_dof+j0*data_dof + 0 << std::endl;
					// Get the complex part and place those values in the second n_nodes3 part of val_vec
					val_vec[1*n_nodes3+j0] = trg_value[i*n_nodes3*data_dof+j0*data_dof + 1];
			}

			{ // Compute Chebyshev approx
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				if(coeff_vec.Dim()!=(data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6){
					coeff_vec.ReInit((data_dof*(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3))/6);
				}
				pvfmm::cheb_approx<double,double>(&val_vec[0], cheb_deg, data_dof, &coeff_vec[0]);
				nlist[i]->DataDOF()=data_dof;
			}
		}
	}

	return;
}

template <class FMM_Mat_t>
std::vector<double> InvMedTree<FMM_Mat_t>::ReadVals(std::vector<double> &coord){

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;


	int dim=InvMedTree<FMM_Mat_t>::dim;
	int data_dof=InvMedTree<FMM_Mat_t>::data_dof;
	assert(coord.size()%dim == 0);
	int n_pts = coord.size() / 3;
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> z;
	std::vector<double> val(n_pts*data_dof);

	FMMNode_t* r_node=static_cast<FMMNode_t*>(this->RootNode());
	double *v = val.data();

	for(int i=0;i<n_pts;i++){
		x.push_back(coord[i*dim + 0]);
		y.push_back(coord[i*dim + 1]);
		z.push_back(coord[i*dim + 2]);
		r_node->ReadVal(x,y,z, &v[i*data_dof], true);

		x.clear();
		y.clear();
		z.clear();
	}

	return val;

}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::SetSrcValues(const std::vector<double> coords, const std::vector<double> values, pvfmm::PtFMM_Tree* tree){

	typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > MPI_Node_t;
	int dim = InvMedTree<FMM_Mat_t>::dim;
	int data_dof = InvMedTree<FMM_Mat_t>::data_dof;

	std::vector<MPI_Node_t*> nlist=tree->GetNodeList();
	std::vector<MPI_Node_t*> src_nodes;
	std::cout << "n_list size" << nlist.size() << std::endl;
	//std::vector<pvfmm::MortonId> mins=pt_tree->GetMins();
//	for(int i=0;i<mins.size();i++){
//		std::cout << mins[i] << std::endl;
//	}
	for(int i=0;i<nlist.size();i++){
//		pvfmm::Vector<double> *sv = &(nlist[i]->src_value);
//		Add the nodes with src_coords to the src_node vector
		pvfmm::Vector<double> *sc = &(nlist[i]->src_coord);
		if(sc->Capacity() >0){
			src_nodes.push_back(nlist[i]);
		}
	}

	// Now we loop through the nodes with nonempty src_coord vectors, and overwrite the src_values with 
	// those values in the matching values vector.
	for(int i=0;i<src_nodes.size();i++){
		pvfmm::Vector<double> *sv = &(src_nodes[i]->src_value);
		pvfmm::Vector<double> *sc = &(src_nodes[i]->src_coord);
		for(int j=0;j<(coords.size())/dim;j++){
			bool match = 1;
			for(int k=0;k<dim;k++){
				if(fabs(coords[j*dim+k]-(sc[0])[k])>.000000001){
					match*=0;
				}
			}
			if(match){
				//std::cout << "j: " << j << std::endl;
				//std::cout << "i: " << i << std::endl;
				for(int k=0;k<data_dof;k++){
					(sv[0][k]) = values[j*data_dof+k];
			//		std::cout << sv[0][k] << std::endl;
				}
			}
		}
	}
	return;
}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::ConjMultiply(InvMedTree* other, double multiplier){

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
			double temp_real;
			double temp_im;


			for(size_t j0=0;j0<n_nodes3;j0++){
				//for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					//real*real - im*im
					temp_real=multiplier*(val_vec1[0*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]+val_vec1[1*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]);
					// real*im + im*real
					temp_im=multiplier*(val_vec2[0*n_nodes3+j0]*val_vec1[1*n_nodes3+j0]-val_vec2[1*n_nodes3+j0]*val_vec1[0*n_nodes3+j0]);
					val_vec1[0*n_nodes3+j0] = temp_real;
					val_vec1[1*n_nodes3+j0] = temp_im;
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
double InvMedTree<FMM_Mat_t>::Norm2(){

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

	std::vector<FMMNode_t*> nlist = this->GetNGLNodes();

	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
	#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	double out = 0;

	#pragma omp parallel for reduction(+:out)
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist.size()* tid   )/omp_p;
		size_t i_end  =(nlist.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> coeff_vec(n_coeff3*data_dof);
		pvfmm::Vector<double> val_vec(n_nodes3*data_dof);

		for(size_t i=i_start;i<i_end;i++){
			double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);
			coeff_vec = nlist[i]->ChebData();

			// val_vec: Evaluate coeff_vec at Chebyshev node points
			cheb_eval(coeff_vec, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec);

			for(size_t j1=0;j1<data_dof;j1++){
				for(size_t j0=0;j0<n_nodes3;j0++){
					out+=(val_vec[j1*n_nodes3+j0])*(val_vec[j1*n_nodes3+j0]);
				}
			}
		}
	}

	return sqrt(out);
}
