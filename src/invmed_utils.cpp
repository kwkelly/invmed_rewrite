#include <petscksp.h>
#include <cassert>
#include <cstring>
#include <profile.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include "invmed_utils.hpp"


PetscInt  VTK_ORDER=0;
PetscInt  INPUT_DOF=2;
PetscReal  SCAL_EXP=1.0;
PetscBool  PERIODIC=PETSC_FALSE;
PetscBool TREE_ONLY=PETSC_FALSE;

PetscInt  MAXDEPTH  =MAX_DEPTH;// Maximum tree depth
PetscInt  MINDEPTH   =4;       // Minimum tree depth
PetscReal       TOL  =1e-3;    // Tolerance
PetscReal GMRES_TOL  =1e-6;    // Fine mesh GMRES tolerance

PetscInt  CHEB_DEG  =14;       // Fine mesh Cheb. order
PetscInt MUL_ORDER  =10;       // Fine mesh mult  order

PetscInt MAX_ITER  =200;

PetscReal f_max=1;
PetscReal eta_=1;

double time_ksp;
int    iter_ksp;
size_t num_oct;


int tree2vec(FMMData fmm_data, Vec& Y){
	PetscErrorCode ierr;
	FMM_Tree_t* tree=fmm_data.tree;
	int cheb_deg=fmm_data.fmm_mat->ChebDeg();

	std::vector<FMMNode_t*> nlist;
	{ // Get non-ghost, leaf nodes.
		std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
		for(size_t i=0;i<nlist_.size();i++){
			if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
				nlist.push_back(nlist_[i]);
			}
		}
	}
	assert(nlist.size()>0);

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());

		PetscScalar *Y_ptr;
		ierr = VecGetArray(Y, &Y_ptr);

#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				double s=std::pow(0.5,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++) Y_ptr[j+Y_offset]=coeff_vec[j]*s;
			}
		}
		ierr = VecRestoreArray(Y, &Y_ptr);
	}

	return 0;
}

int vec2tree(Vec& Y, FMMData fmm_data){
	PetscErrorCode ierr;
	FMM_Tree_t* tree=fmm_data.tree;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg=fmm_data.fmm_mat->ChebDeg();

	std::vector<FMMNode_t*> nlist;
	{ // Get non-ghost, leaf nodes.
		std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
		for(size_t i=0;i<nlist_.size();i++){
			if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
				nlist.push_back(nlist_[i]);
			}
		}
	}

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());

		const PetscScalar *Y_ptr;
		ierr = VecGetArrayRead(Y, &Y_ptr);

#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++) coeff_vec[j]=PetscRealPart(Y_ptr[j+Y_offset])*s;
				nlist[i]->DataDOF()=data_dof;
			}
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////

int FMM_Init(MPI_Comm& comm, FMMData *fmm_data){
	int myrank, np;
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm,&np);

	FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	FMM_Tree_t* tree=new FMM_Tree_t(comm);

	//Kernel function
	const pvfmm::Kernel<double>* kernel;
	if(INPUT_DOF==1) kernel=pvfmm::LaplaceKernel<double>::potn_ker;
	else if(INPUT_DOF==2) kernel=&pvfmm::ker_helmholtz;
	else if(INPUT_DOF==3) kernel=&pvfmm::ker_stokes_vel;

	//Setup FMM data structure.
	int mult_order=MUL_ORDER;
	int cheb_deg=CHEB_DEG;
	bool adap=true;
	double tol=TOL;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	if(PERIODIC==PETSC_TRUE) bndry=pvfmm::Periodic;

	typename FMMNode_t::NodeData tree_data;
	{ // Tree Construction (eta).
		FMM_Tree_t* tree=new FMM_Tree_t(comm);
		//Various parameters.
		tree_data.dim=COORD_DIM;
		tree_data.max_depth=MAXDEPTH;
		tree_data.cheb_deg=cheb_deg;

		//Set input function pointer
		tree_data.input_fn=eta;
		tree_data.data_dof=1;
		tree_data.tol=tol*fabs(eta_);

		std::vector<double> pt_coord;
		{ //Set source coordinates.
			size_t NN=ceil(pow((double)np,1.0/3.0));
			NN=std::max<size_t>(NN,pow(2.0,MINDEPTH));
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
		tree->Initialize(&tree_data);
		tree->InitFMM_Tree(adap,bndry);

		// this is getting the center coordinates of each of the nodes of the octree?
		pt_coord.clear();
		std::vector<FMMNode_t*> nlist=tree->GetNodeList();
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

		tree->Write2File("results/eta",VTK_ORDER);
		delete tree;
	}

	//new

	//  typename FMMNode_t::NodeData tree_data;
	//Various parameters.
	//  tree_data.dim=COORD_DIM;
	//  tree_data.max_depth=MAXDEPTH;
	//  tree_data.cheb_deg=cheb_deg;

	{ // Tree Construction.
		bool adap=true;

		// Manually setting the coordinates where we evaluate the function?
		std::vector<double> pt_coord;
		{ //Set source coordinates.
			size_t NN=ceil(pow((double)np,1.0/3.0));
			NN=std::max<size_t>(NN,pow(2.0,MINDEPTH));
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

		//old
		//
		//Set input function pointer
		tree_data.input_fn=fn_input;
		tree_data.data_dof=kernel->ker_dim[0];
		tree_data.tol=tol*f_max;

		//Create Tree and initialize with input data.
		tree->Initialize(&tree_data);
		tree->InitFMM_Tree(adap,bndry);

		//std::vector<double> pt_coord; Again getting the point coordinates of the centers of the nodes of the octree?
		pt_coord.clear();
		std::vector<FMMNode_t*> nlist=tree->GetNodeList();
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
			std::vector<size_t> all_nodes(MAXDEPTH+1,0);
			std::vector<size_t> leaf_nodes(MAXDEPTH+1,0);
			std::vector<FMMNode_t*>& nodes=tree->GetNodeList();
			for(size_t i=0;i<nodes.size();i++){
				FMMNode_t* n=nodes[i];
				if(!n->IsGhost()) all_nodes[n->Depth()]++;
				if(!n->IsGhost() && n->IsLeaf()) leaf_nodes[n->Depth()]++;
			}

			if(!myrank) std::cout<<"All  Nodes: ";
			for(int i=0;i<MAXDEPTH;i++){
				int local_size=all_nodes[i];
				int global_size;
				MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
				if(global_size==0) MAXDEPTH=i;
				if(!myrank) std::cout<<global_size<<' ';
			}
			if(!myrank) std::cout<<'\n';

			if(!myrank) std::cout<<"Leaf Nodes: ";
			for(int i=0;i<MAXDEPTH;i++){
				int local_size=leaf_nodes[i];
				int global_size;
				MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
				if(!myrank) std::cout<<global_size<<' ';
			}
			if(!myrank) std::cout<<'\n';
		}
	}

	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	PetscInt m=0,n=0,M=0,N=0,l=0,L=0;
	{ // Get local and global size
		long long loc_size=0, glb_size=0;
		long long loc_nodes=0, glb_nodes=0;
		std::vector<FMMNode_t*> nlist=tree->GetNodeList();
		for(size_t i=0;i<nlist.size();i++){
			if(nlist[i]->IsLeaf() && !nlist[i]->IsGhost()){
				loc_size+=n_coeff3; //nlist[i]->ChebData().Dim();
				loc_nodes+=n_nodes3;
			}
		}
		MPI_Allreduce(&loc_size, &glb_size, 1, MPI_LONG_LONG , MPI_SUM, comm);
		MPI_Allreduce(&loc_nodes, &glb_nodes, 1, MPI_LONG_LONG, MPI_SUM, comm);
		n=loc_size*kernel->ker_dim[0];
		N=glb_size*kernel->ker_dim[0];
		m=loc_size*kernel->ker_dim[1];
		M=glb_size*kernel->ker_dim[1];
		l=loc_nodes*kernel->ker_dim[0];
		L=glb_nodes*kernel->ker_dim[0];
		std::cout << L << std::endl;
		num_oct=glb_size/n_coeff3;
	}
	if(TREE_ONLY) return 0;

	//Initialize FMM_Mat.
	fmm_mat->Initialize(mult_order,cheb_deg,comm,kernel);

	fmm_data->kernel =kernel ;
	fmm_data->fmm_mat=fmm_mat;
	fmm_data->tree   =tree   ;
	fmm_data->bndry  =bndry  ;
	fmm_data->m=m;
	fmm_data->n=n;
	fmm_data->M=M;
	fmm_data->N=N;
	fmm_data->l=l;
	fmm_data->L=L;

	return 0;
}

int FMMCreateShell(FMMData *fmm_data, Mat *A){


	FMM_Tree_t*   tree=fmm_data->tree   ;
	const MPI_Comm& comm=*tree->Comm();
	PetscInt m,n,M,N;
	m=fmm_data->m;
	n=fmm_data->n;
	M=fmm_data->M;
	N=fmm_data->N;
	/*
	   { // Evaluate eta at Chebyshev node points.
	   std::vector<double>& eta_vec=fmm_data->eta;

	   std::vector<FMMNode_t*> nlist;
	   { // Get non-ghost, leaf nodes.
	   std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
	   for(size_t i=0;i<nlist_.size();i++){
	   if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
	   nlist.push_back(nlist_[i]);
	   }
	   }
	   }

	   int cheb_deg=fmm_data->fmm_mat->ChebDeg();
	   size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	   eta_vec.resize(n_nodes3*nlist.size());

	   std::vector<double> cheb_node_coord3=pvfmm::cheb_nodes<double>(cheb_deg, 3);
	   int omp_p=omp_get_max_threads();
#pragma omp parallel for
for(size_t tid=0;tid<omp_p;tid++){
size_t i_start=(nlist.size()* tid   )/omp_p;
size_t i_end  =(nlist.size()*(tid+1))/omp_p;

std::vector<double> cheb_node_coord3_(n_nodes3*3);
std::vector<double> eta_val(n_nodes3);
for(size_t i=i_start;i<i_end;i++){
	// Shift Cheb node points and evaluate eta
	double* coord=nlist[i]->Coord();
	double s=pow(0.5,nlist[i]->Depth());
	for(size_t j=0;j<n_nodes3;j++){
	cheb_node_coord3_[j*3+0]=cheb_node_coord3[j*3+0]*s+coord[0];
	cheb_node_coord3_[j*3+1]=cheb_node_coord3[j*3+1]*s+coord[1];
	cheb_node_coord3_[j*3+2]=cheb_node_coord3[j*3+2]*s+coord[2];
	}
	eta(&cheb_node_coord3_[0], n_nodes3, &eta_val[0]);

	size_t vec_offset=i*n_nodes3;
	for(size_t j=0;j<n_nodes3;j++){
	eta_vec[vec_offset+j]=eta_val[j];
	}
	}
	}
	}
	*/
	eval_function_at_nodes(fmm_data, eta, fmm_data->eta);
	return MatCreateShell(comm,m,n,M,N,fmm_data,A);
}

int FMMDestroy(FMMData *fmm_data){
	delete fmm_data->fmm_mat;
	delete fmm_data->tree;
	return 1;
}

////////////////////////////////////////////////////////////////////////////////

int PtWiseTreeMult(FMMData &fmm_data, FMM_Tree_t &tree2){
	FMM_Tree_t* tree1=fmm_data.tree;
	const MPI_Comm* comm=tree1->Comm();
	int omp_p=omp_get_max_threads();
	int cheb_deg=fmm_data.fmm_mat->ChebDeg();

	std::vector<FMMNode_t*> nlist1;
	std::vector<FMMNode_t*> nlist2;
	{ // Get non-ghost, leaf nodes for BOTH trees. They must have the same structure.
		std::vector<FMMNode_t*>& nlist1_=tree1->GetNodeList();
		std::vector<FMMNode_t*>& nlist2_=tree2.GetNodeList();
		for(size_t i=0;i<nlist1_.size();i++){
			if(nlist1_[i]->IsLeaf() && !nlist1_[i]->IsGhost()){
				nlist1.push_back(nlist1_[i]);
				nlist2.push_back(nlist2_[i]);
			}
		}
	}
	assert(nlist1.size()>0);

	// Cheb node points
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	// PtWise Mult
	pvfmm::Profile::Tic("FMM_PtWise_Mult",comm,true);
	{
		/*
		   PetscInt U_size;
		   PetscInt phi_0_size;
		   ierr = VecGetLocalSize(U, &U_size);
		   ierr = VecGetLocalSize(phi_0_vec, &phi_0_size);
		   int data_dof=U_size/(n_coeff3*nlist.size());
		   assert(data_dof*n_coeff3*nlist.size()==U_size);
		   */
		int data_dof = fmm_data.kernel->ker_dim[0];
		/*
		   PetscScalar *U_ptr;
		   PetscScalar* phi_0_ptr;
		   ierr = VecGetArray(U, &U_ptr);
		   ierr = VecGetArray(phi_0_vec, &phi_0_ptr);

*/
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
				/*
				   { // coeff_vec: Cheb coeff data for this node
				   size_t U_offset=i*n_coeff3*data_dof;
				   size_t phi_0_offset = i*n_nodes3*data_dof;
				   for(size_t j=0;j<n_coeff3*data_dof;j++){
				   coeff_vec[j]=PetscRealPart(U_ptr[j+U_offset])*s;
				   }
				   for(size_t j=0;j<n_nodes3*data_dof;j++){
				   phi_0_part[j]=PetscRealPart(phi_0_ptr[j+phi_0_offset]);
				   }
				   }
				   */
				{ // coeff_vecs: Cheb coeffs for this node for each tree!
					coeff_vec1 = nlist1[i]->ChebData();
					coeff_vec2 = nlist2[i]->ChebData();
				}
				// val_vec: Evaluate coeff_vec at Chebyshev node points
				cheb_eval(coeff_vec1, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec1);
				cheb_eval(coeff_vec2, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec2);

				{// phi_0_part*val_vec
					for(size_t j0=0;j0<data_dof;j0++){
						double* vec1=&val_vec1[j0*n_nodes3];
						double* vec2=&val_vec2[j0*n_nodes3];
						for(size_t j1=0;j1<n_nodes3;j1++) vec1[j1]*=vec2[j1];
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
	}
	pvfmm::Profile::Toc();

	return 0;
}


int mult(Mat M, Vec U, Vec Y){
	PetscErrorCode ierr;
	FMMData* fmm_data=NULL;
	MatShellGetContext(M, &fmm_data);
	FMM_Tree_t* tree=fmm_data->tree;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg=fmm_data->fmm_mat->ChebDeg();
	std::vector<double>& eta_vec=fmm_data->eta;
	Vec& phi_0_vec=fmm_data->phi_0_vec;
	int omp_p=omp_get_max_threads();
	//int omp_p=1;
	pvfmm::Profile::Tic("FMM_Mul",comm,true);

	std::vector<FMMNode_t*> nlist;
	{ // Get non-ghost, leaf nodes.
		std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
		for(size_t i=0;i<nlist_.size();i++){
			if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
				nlist.push_back(nlist_[i]);
			}
		}
	}
	assert(nlist.size()>0);

	// Cheb node points
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}

	// Input Vector ( \phi_0_vec * U )
	pvfmm::Profile::Tic("FMM_Input",comm,true);
	{
		PetscInt U_size;
		PetscInt phi_0_size;
		ierr = VecGetLocalSize(U, &U_size);
		ierr = VecGetLocalSize(phi_0_vec, &phi_0_size);
		int data_dof=U_size/(n_coeff3*nlist.size());
		assert(data_dof*n_coeff3*nlist.size()==U_size);

		PetscScalar *U_ptr;
		PetscScalar* phi_0_ptr;
		ierr = VecGetArray(U, &U_ptr);
		ierr = VecGetArray(phi_0_vec, &phi_0_ptr);
#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			pvfmm::Vector<double> coeff_vec(n_coeff3*data_dof);
			pvfmm::Vector<double> val_vec(n_nodes3*data_dof);
			pvfmm::Vector<double> phi_0_part(n_nodes3*data_dof);
			for(size_t i=i_start;i<i_end;i++){
				double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				{ // coeff_vec: Cheb coeff data for this node
					size_t U_offset=i*n_coeff3*data_dof;
					size_t phi_0_offset = i*n_nodes3*data_dof;
					for(size_t j=0;j<n_coeff3*data_dof;j++){
						coeff_vec[j]=PetscRealPart(U_ptr[j+U_offset])*s;
					}
					for(size_t j=0;j<n_nodes3*data_dof;j++){
						phi_0_part[j]=PetscRealPart(phi_0_ptr[j+phi_0_offset]);
					}
				}
				// val_vec: Evaluate coeff_vec at Chebyshev node points
				cheb_eval(coeff_vec, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec);

				{// phi_0_part*val_vec
					for(size_t j0=0;j0<data_dof;j0++){
						double* vec=&val_vec[j0*n_nodes3];
						double* phi_0=&phi_0_part[j0*n_nodes3];
						for(size_t j1=0;j1<n_nodes3;j1++) vec[j1]*=phi_0[j1];
					}
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
	}
	pvfmm::Profile::Toc();

	// Run FMM ( Compute: G[ \eta * u ] )
	tree->ClearFMMData();
	tree->RunFMM();

	// Copy data from tree to Y
	pvfmm::Profile::Tic("tree2vec",comm,true);
	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());

		PetscScalar *Y_ptr;
		ierr = VecGetArray(Y, &Y_ptr);

#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=((FMM_Mat_t::FMMData*) nlist[i]->FMMData())->cheb_out;
				double s=std::pow(0.5,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++) Y_ptr[j+Y_offset]=coeff_vec[j]*s;
			}
		}

		ierr = VecRestoreArray(Y, &Y_ptr);
	}
	pvfmm::Profile::Toc();
	// Regularize

	Vec alpha;
	PetscScalar sca = (PetscScalar).01;
	VecDuplicate(Y,&alpha);
	VecSet(alpha,sca);
	ierr = VecPointwiseMult(alpha,alpha,U);
	ierr = VecAXPY(Y,1,alpha);
	// Output Vector ( Compute:  U + G[ \eta * U ] )
	// pvfmm::Profile::Tic("FMM_Output",comm,true);
	// ierr = VecAXPY(Y,1,U);CHKERRQ(ierr);
	// pvfmm::Profile::Toc();
	ierr = VecDestroy(&alpha); CHKERRQ(ierr);

	pvfmm::Profile::Toc();
	return 0;
}

/*
   int mult(Mat M, Vec U, Vec Y){
   PetscErrorCode ierr;
   FMMData* fmm_data=NULL;
   MatShellGetContext(M, &fmm_data);
   FMM_Tree_t* tree=fmm_data->tree;
   const MPI_Comm* comm=tree->Comm();
   int cheb_deg=fmm_data->fmm_mat->ChebDeg();
   std::vector<double>& eta_vec=fmm_data->eta;
   int omp_p=omp_get_max_threads();
   pvfmm::Profile::Tic("FMM_Mul",comm,true);

   std::vector<FMMNode_t*> nlist;
   { // Get non-ghost, leaf nodes.
   std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
   for(size_t i=0;i<nlist_.size();i++){
   if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
   nlist.push_back(nlist_[i]);
   }
   }
   }
   assert(nlist.size()>0);

// Cheb node points
size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);
#pragma omp parallel for
for(size_t i=0;i<cheb_node_coord1.size();i++){
cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
}

// Input Vector ( \eta * U )
pvfmm::Profile::Tic("FMM_Input",comm,true);
{
PetscInt U_size;
ierr = VecGetLocalSize(U, &U_size);
int data_dof=U_size/(n_coeff3*nlist.size());
assert(data_dof*n_coeff3*nlist.size()==U_size);

PetscScalar *U_ptr;
ierr = VecGetArray(U, &U_ptr);
#pragma omp parallel for
for(size_t tid=0;tid<omp_p;tid++){
size_t i_start=(nlist.size()* tid   )/omp_p;
size_t i_end  =(nlist.size()*(tid+1))/omp_p;
pvfmm::Vector<double> coeff_vec(n_coeff3*data_dof);
pvfmm::Vector<double> val_vec(n_nodes3*data_dof);
for(size_t i=i_start;i<i_end;i++){
double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

{ // coeff_vec: Cheb coeff data for this node
size_t U_offset=i*n_coeff3*data_dof;
for(size_t j=0;j<n_coeff3*data_dof;j++) coeff_vec[j]=PetscRealPart(U_ptr[j+U_offset])*s;
}

// val_vec: Evaluate coeff_vec at Chebyshev node points
cheb_eval(coeff_vec, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, val_vec);

{// eta*val_vec
double* eta=&eta_vec[i*n_nodes3];
for(size_t j0=0;j0<data_dof;j0++){
double* vec=&val_vec[j0*n_nodes3];
for(size_t j1=0;j1<n_nodes3;j1++) vec[j1]*=eta[j1];
}
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
}
pvfmm::Profile::Toc();

// Run FMM ( Compute: G[ \eta * u ] )
tree->ClearFMMData();
tree->RunFMM();

// Copy data from tree to Y
pvfmm::Profile::Tic("tree2vec",comm,true);
{
	PetscInt Y_size;
	ierr = VecGetLocalSize(Y, &Y_size);
	int data_dof=Y_size/(n_coeff3*nlist.size());

	PetscScalar *Y_ptr;
	ierr = VecGetArray(Y, &Y_ptr);

#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist.size()* tid   )/omp_p;
		size_t i_end  =(nlist.size()*(tid+1))/omp_p;
		for(size_t i=i_start;i<i_end;i++){
			pvfmm::Vector<double>& coeff_vec=((FMM_Mat_t::FMMData*) nlist[i]->FMMData())->cheb_out;
			double s=std::pow(0.5,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

			size_t Y_offset=i*n_coeff3*data_dof;
			for(size_t j=0;j<n_coeff3*data_dof;j++) Y_ptr[j+Y_offset]=coeff_vec[j]*s;
		}
	}

	ierr = VecRestoreArray(Y, &Y_ptr);
}
pvfmm::Profile::Toc();

// Output Vector ( Compute:  U + G[ \eta * U ] )
pvfmm::Profile::Tic("FMM_Output",comm,true);
ierr = VecAXPY(Y,1,U);CHKERRQ(ierr);
pvfmm::Profile::Toc();

pvfmm::Profile::Toc();
return 0;
}
*/

int eval_function_at_nodes(FMMData *fmm_data, void (*func)(double* coord, int n, double* out), std::vector<double> &func_vec){
	FMM_Tree_t*   tree=fmm_data->tree   ;
	const MPI_Comm& comm=*tree->Comm();
	PetscInt m,n,M,N;
	m=fmm_data->m;
	n=fmm_data->n;
	M=fmm_data->M;
	N=fmm_data->N;

	{ // Evaluate func at Chebyshev node points.
		//  std::vector<double> func_vec;

		std::vector<FMMNode_t*> nlist;
		{ // Get non-ghost, leaf nodes.
			std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
			for(size_t i=0;i<nlist_.size();i++){
				if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
					nlist.push_back(nlist_[i]);
				}
			}
		}

		int cheb_deg=fmm_data->fmm_mat->ChebDeg();
		size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
		func_vec.resize(n_nodes3*nlist.size());

		std::vector<double> cheb_node_coord3=pvfmm::cheb_nodes<double>(cheb_deg, 3);
		int omp_p=omp_get_max_threads();
#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;

			std::vector<double> cheb_node_coord3_(n_nodes3*3);
			std::vector<double> func_val(n_nodes3);
			for(size_t i=i_start;i<i_end;i++){
				// Shift Cheb node points and evaluate func
				double* coord=nlist[i]->Coord();
				double s=pow(0.5,nlist[i]->Depth());
				for(size_t j=0;j<n_nodes3;j++){
					cheb_node_coord3_[j*3+0]=cheb_node_coord3[j*3+0]*s+coord[0];
					cheb_node_coord3_[j*3+1]=cheb_node_coord3[j*3+1]*s+coord[1];
					cheb_node_coord3_[j*3+2]=cheb_node_coord3[j*3+2]*s+coord[2];
				}
				func(&cheb_node_coord3_[0], n_nodes3, &func_val[0]);

				size_t vec_offset=i*n_nodes3;
				for(size_t j=0;j<n_nodes3;j++){
					func_vec[vec_offset+j]=func_val[j];
				}
			}
		}
	}


	return 1;

}


int eval_cheb_at_nodes(FMMData *fmm_data, Vec &val_vec){

	FMM_Tree_t* tree=fmm_data->tree;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg=fmm_data->fmm_mat->ChebDeg();
	PetscInt       m,n, M,N;
	PetscErrorCode ierr;
	// Cheb node points
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
	std::vector<double> cheb_node_coord1=pvfmm::cheb_nodes<double>(cheb_deg, 1);

	int omp_p=omp_get_max_threads();
	//int omp_p=1;
#pragma omp parallel for
	for(size_t i=0;i<cheb_node_coord1.size();i++){
		cheb_node_coord1[i]=cheb_node_coord1[i]*2.0-1.0;
	}


	std::vector<FMMNode_t*> nlist;
	{ // Get non-ghost, leaf nodes.
		std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
		for(size_t i=0;i<nlist_.size();i++){
			if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
				nlist.push_back(nlist_[i]);
			}
		}
	}

	m=fmm_data->m; // local rows
	n=fmm_data->n; // local columns
	M=fmm_data->M; // global rows
	N=fmm_data->N; // global columns

	//create coeff vec
	Vec coeff_vec;
	VecCreateMPI(*comm,m,PETSC_DETERMINE,&coeff_vec);

	tree2vec(*fmm_data,coeff_vec);

	PetscInt coeff_vec_size;
	ierr = VecGetLocalSize(coeff_vec, &coeff_vec_size);
	int data_dof=coeff_vec_size/(n_coeff3*nlist.size());
	assert(data_dof*n_coeff3*nlist.size()==coeff_vec_size);

	PetscScalar *coeff_vec_ptr;
	PetscScalar *val_vec_ptr;
	ierr = VecGetArray(coeff_vec, &coeff_vec_ptr);
	ierr = VecGetArray(val_vec, &val_vec_ptr);
#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist.size()* tid   )/omp_p;
		size_t i_end  =(nlist.size()*(tid+1))/omp_p;
		pvfmm::Vector<double> single_node_coeff_vec(n_coeff3*data_dof);
		pvfmm::Vector<double> single_node_val_vec(n_nodes3*data_dof);
		for(size_t i=i_start;i<i_end;i++){
			double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

			{ // coeff_vec: Cheb coeff data for this node
				size_t coeff_vec_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++) single_node_coeff_vec[j]=PetscRealPart(coeff_vec_ptr[j+coeff_vec_offset])*s;
			}

			// val_vec: Evaluate coeff_vec at Chebyshev node points
			cheb_eval(single_node_coeff_vec, cheb_deg, cheb_node_coord1, cheb_node_coord1, cheb_node_coord1, single_node_val_vec);
			//std::cout << "here" << std::endl; 
			{ // val_vec: places the values into the vector
				size_t val_vec_offset=i*n_nodes3*data_dof;
				for(size_t j=0;j<n_nodes3*data_dof;j++){
					//std::cout << single_node_val_vec[j] << std::endl;
					val_vec_ptr[j+val_vec_offset] = single_node_val_vec[j];
				}
			}
			//std::cout << "not here though" << std::endl;
		}
	}
	ierr = VecRestoreArray(coeff_vec, &coeff_vec_ptr);
	ierr = VecRestoreArray(val_vec, &val_vec_ptr);

	return 1;

}

int CompPhiUsingBorn(Vec &true_sol, FMMData &fmm_data){

	// Initialize and get data about the problem
	PetscErrorCode ierr;
	PetscInt       m,n,l, M,N,L;
	m=fmm_data.m; // local rows
	n=fmm_data.n; // local columns
	M=fmm_data.M; // global rows
	N=fmm_data.N; // global columns
	l=fmm_data.l; // local val vec length
	L=fmm_data.L; // global val vec length
	FMM_Tree_t *tree=fmm_data.tree;
	const MPI_Comm comm=*(tree->Comm());
	int cheb_deg=fmm_data.fmm_mat->ChebDeg();
	int omp_p=omp_get_max_threads();

	// Get nodes (I think that these are just the ones local to this compute node)
	std::vector<FMMNode_t*> nlist;
	{ // Get non-ghost, leaf nodes.
		std::vector<FMMNode_t*>& nlist_=tree->GetNodeList();
		for(size_t i=0;i<nlist_.size();i++){
			if(nlist_[i]->IsLeaf() && !nlist_[i]->IsGhost()){
				nlist.push_back(nlist_[i]);
			}
		}
	}
	assert(nlist.size()>0);

	// Pointwise multiplication of \eta and \phi_0
	Vec prod;
	VecCreateMPI(comm,l,PETSC_DETERMINE,&prod);

	ierr = VecPointwiseMult(prod, fmm_data.eta_val,fmm_data.phi_0_vec);
	CHKERRQ(ierr);

	// Get cheb coeffs from the values and store them in the tree

	{
		size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
		size_t n_nodes3=(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
		PetscInt prod_size;
		ierr = VecGetLocalSize(prod, &prod_size);
		int data_dof=prod_size/(n_nodes3*nlist.size());
		assert(data_dof*n_nodes3*nlist.size()==prod_size);

		PetscScalar *prod_ptr;
		ierr = VecGetArray(prod, &prod_ptr);
#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			pvfmm::Vector<double> coeff_vec(n_coeff3*data_dof);
			pvfmm::Vector<double> val_vec(n_nodes3*data_dof);
			for(size_t i=i_start;i<i_end;i++){

				{ // copy values of the product vector into val_vec which contains the values in the current node
					size_t prod_offset=i*n_nodes3*data_dof;
					for(size_t j=0;j<n_nodes3*data_dof;j++) val_vec[j]=PetscRealPart(prod_ptr[j+prod_offset]);
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
	}

	//run fmm
	tree->ClearFMMData();
	tree->RunFMM();
	tree->Copy_FMMOutput();

	tree2vec(fmm_data,true_sol);
	VecAXPY(true_sol,1,fmm_data.phi_0);
	vec2tree(true_sol,fmm_data);

	return 1;        
}
