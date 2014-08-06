#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <mpi.h>
#include <profile.hpp>
#include <iostream>
#include <vector>



template <class FMM_Mat_t>
std::set< InvMedTree<FMM_Mat_t>* > InvMedTree<FMM_Mat_t>::m_instances;
//typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
//typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
//typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;

//template <class FMM_Mat_t>
//InvMedTree<FMM_Mat_t> ::InvMedTree(MPI_Comm c) : pvfmm::FMM_Tree<FMM_Mat_t>(c){};
template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Initialize(){

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typename FMMNode_t::NodeData tree_data;
	//Various parameters.
	tree_data.dim=this->dim;
	tree_data.max_depth=this->maxdepth;;
	tree_data.cheb_deg=this->cheb_deg;

	//Set input function pointer
	tree_data.input_fn=this->fn;
	tree_data.data_dof=this->data_dof;
	tree_data.tol=(this->tol)*(this->f_max);

	int myrank, np;
	MPI_Comm_rank(*(this->Comm()), &myrank);
	MPI_Comm_size(*(this->Comm()),&np);

	std::vector<double> pt_coord;
	{ //Set source coordinates.
		size_t NN=ceil(pow((double)np,1.0/3.0));
		NN=std::max<size_t>(NN,pow(2.0,this->mindepth));
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
	tree_data.max_pts=1; // Points per octant.

	//Create Tree and initialize with input data.
	pvfmm::FMM_Tree<FMM_Mat_t>::Initialize(&tree_data);

	std::cout << "got hereq" << std::endl;
	std::cout << "got here" << std::endl;
	pvfmm::FMM_Tree<FMM_Mat_t>::InitFMM_Tree(this->adap,this->bndry);
	std::cout << (this->GetNodeList()).size() << std::endl;
	std::cout << "We are here" << std::endl;

	// this is getting the center coordinates of each of the nodes of the octree?
	pt_coord.clear();
	std::vector<FMMNode_t*> nlist=this->GetNodeList();
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


	{ //Output max tree depth.
		std::vector<size_t> all_nodes(this->maxdepth+1,0);
		std::vector<size_t> leaf_nodes(this->maxdepth+1,0);
		std::vector<FMMNode_t*>& nodes=this->GetNodeList();
		for(size_t i=0;i<nodes.size();i++){
			FMMNode_t* n=nodes[i];
			if(!n->IsGhost()) all_nodes[n->Depth()]++;
			if(!n->IsGhost() && n->IsLeaf()) leaf_nodes[n->Depth()]++;
		}

		if(!myrank) std::cout<<"All  Nodes: ";
		for(int i=0;i<this->maxdepth;i++){
			int local_size=all_nodes[i];
			int global_size;
			MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, *(this->Comm()));
			if(global_size==0) this->maxdepth=i;
			if(!myrank) std::cout<<global_size<<' ';
		}
		if(!myrank) std::cout<<'\n';

		if(!myrank) std::cout<<"Leaf Nodes: ";
		for(int i=0;i<this->maxdepth;i++){
			int local_size=leaf_nodes[i];
			int global_size;
			MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, *(this->Comm()));
			if(!myrank) std::cout<<global_size<<' ';
		}
		if(!myrank) std::cout<<'\n';
	}

	size_t n_coeff3=(this->cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(this->cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	{ // Get local and global size
		long long loc_size=0, glb_size=0;
		long long loc_nodes=0, glb_nodes=0;
		std::vector<FMMNode_t*> nlist=this->GetNodeList();
		for(size_t i=0;i<nlist.size();i++){
			if(nlist[i]->IsLeaf() && !nlist[i]->IsGhost()){
				loc_size+=n_coeff3; //nlist[i]->ChebData().Dim();
				loc_nodes+=n_nodes3;
			}
		}
		MPI_Allreduce(&loc_size, &glb_size, 1, MPI_LONG_LONG , MPI_SUM, *(this->Comm()));
		MPI_Allreduce(&loc_nodes, &glb_nodes, 1, MPI_LONG_LONG, MPI_SUM, *(this->Comm()));
		this->n=loc_size*kernel->ker_dim[0];
		this->N=glb_size*kernel->ker_dim[0];
		this->m=loc_size*kernel->ker_dim[1];
		this->M=glb_size*kernel->ker_dim[1];
		this->l=loc_nodes*kernel->ker_dim[0];
		this->L=glb_nodes*kernel->ker_dim[0];
	}

	return;
}
	

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Copy(InvMedTree &new_tree, const InvMedTree &other){
	//const MPI_Comm comm = *(other.Comm());
  new_tree.kernel = other.kernel;
	new_tree.bndry = other.bndry;
	new_tree.cheb_deg = other.cheb_deg;
	new_tree.mult_order = other.mult_order;
	new_tree.tol = other.tol;
	new_tree.mindepth = other.mindepth;
	new_tree.maxdepth = other.maxdepth;
	new_tree.fn = other.fn;
	new_tree.f_max = other.f_max;
	new_tree.dim = other.dim;;
	new_tree.adap = other.adap;

	//new_tree.Initialize();

	return;

}

template <class FMM_Mat_t>
void InvMedTree<FMM_Mat_t>::Multiply(InvMedTree &tree1, InvMedTree &tree2){

	//  may need to change this
	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=tree1.Comm();
	//int omp_p=omp_get_max_threads();
	int omp_p = 1;
	int cheb_deg=tree1.cheb_deg;
	int data_dof=tree1.data_dof;

	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);

	std::vector<FMMNode_t*> nlist1 = tree1.GetNGLNodes();
	std::vector<FMMNode_t*> nlist2 = tree2.GetNGLNodes();

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
					val_vec1[0*n_nodes3+j0]=val_vec1[0*n_nodes3+j0]*val_vec2[0*n_nodes3+j0]-val_vec1[1*n_nodes3+j0]*val_vec2[1*n_nodes3+j0];
					// real*im + im*real
					val_vec1[1*n_nodes3+j0]=val_vec1[0*n_nodes3+j0]*val_vec2[1*n_nodes3+j0]+val_vec1[1*n_nodes3+j0]*val_vec2[0*n_nodes3+j0];
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
void InvMedTree<FMM_Mat_t>::Add(InvMedTree &tree1, InvMedTree &tree2){

	//  may need to change this
	int SCAL_EXP = 1;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=tree1.Comm();
	//int omp_p=omp_get_max_threads();
	int omp_p = 1;
	int cheb_deg=tree1.cheb_deg;
	int data_dof=tree1.data_dof;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)/6;

	std::vector<FMMNode_t*> nlist1 = tree1.GetNGLNodes();
//	std::cout << nlist1.size() << std::endl;
	std::vector<FMMNode_t*> nlist2 = tree2.GetNGLNodes();
//	std::cout << nlist2.size() << std::endl;

//	std::cout << nlist1[1]->ChebData() << std::endl;
//	std::cout << nlist1[1]->DataDOF() << std::endl;
//	std::cout << n_coeff3 <<std::endl;
//	std::cout << n_nodes3 <<std::endl;
	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
	#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}
	/*
	{	
		pvfmm::Vector<double> val_vec1(n_nodes3*data_dof);
		pvfmm::Vector<double> val_vec2(n_nodes3*data_dof);
		cheb_eval(nlist1[1]->ChebData(), cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec1);
		cheb_eval(nlist2[1]->ChebData(), cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec2);
		std::cout << val_vec1 << std::endl;
		for(size_t j0=0;j0<n_nodes3;j0++){
			for(size_t j1=0;j1<data_dof;j1++){
				//why is this like this?
				val_vec1[j1*n_nodes3+j0]+=val_vec2[j1*n_nodes3+j0];
			}
		}
		std::cout << val_vec1 << std::endl;
	}*/
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
				for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					val_vec1[j1*n_nodes3+j0]+=val_vec2[j1*n_nodes3+j0];
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
void InvMedTree<FMM_Mat_t>::CreateDiffFunction(InvMedTree &tree1, InvMedTree &tree2){

	(*tmp_fn)(double* coord, int n, double* out);
	tmp_fn = tree_2.fn;
	tree_2.fn=tree_1.fn;
	//  may need to change this
	int SCAL_EXP = 1;

	std::vector<FMMNode_t*> nlist1 = tree1.GetNGLNodes();

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	const MPI_Comm* comm=tree1.Comm();
	//int omp_p=omp_get_max_threads();
	int omp_p = 1;
	int cheb_deg=tree1.cheb_deg;
	int data_dof=tree1.data_dof;

	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);

	std::vector<FMMNode_t*> nlist1 = tree1.GetNGLNodes();
	std::vector<FMMNode_t*> nlist2 = tree2.GetNGLNodes();

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
				for(size_t j1=0;j1<data_dof;j1++){
					//why is this like this?
					val_vec1[j1*n_nodes3+j0]*=val_vec2[j1*n_nodes3+j0];
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
