#include "invmed_tree.hpp"
#include <iostream>
#include <petscksp.h>
#include <profile.hpp>
#include "funcs.hpp"
#include <pvfmm_common.hpp>
#include "typedefs.hpp"


#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	InvMedData invmed_data;
	MatShellGetContext(M, &invmed_data);
	std::cout << "before reassignment" << std::endl;
	std::cout << (invmed_data.phi_0->GetNodeList()).size() << std::endl;
	//InvMedTree<FMM_Mat_t>* phi_0 = invmed_data.phi_0;
	//InvMedTree<FMM_Mat_t>* temp = invmed_data.temp;
	std::cout << "phi_0 size " << std::endl;
	std::cout << ((invmed_data.phi_0)->GetNodeList()).size() << std::endl;
	std::cout << "temp size " << std::endl;
	std::cout << ((invmed_data.temp)->GetNodeList()).size() << std::endl;

	const MPI_Comm* comm=invmed_data.phi_0->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int omp_p=omp_get_max_threads();
	//int omp_p=1;
	std::cout << "Does this get called?" << std::endl;
//	pvfmm::Profile::Tic("FMM_Mul",comm,true);

	std::cout << "Does this get called?" << std::endl;
	//VecView(U, PETSC_VIEWER_STDOUT_SELF);
	vec2tree(U,invmed_data.temp);
/*
	std::vector<FMMNode_t*> nlist = phi_0;
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
	pvfmm::Profile::Toc();A
		*/

	std::cout << "Does this get called?" << std::endl;
	invmed_data.temp->Multiply(invmed_data.phi_0,1);
	std::cout << "Does this get called?" << std::endl;

	// Run FMM ( Compute: G[ \eta * u ] )
	invmed_data.temp->ClearFMMData();
	invmed_data.temp->RunFMM();
/*
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
*/

	// Regularize
	//double alpha = .00001;
	//temp->ScalarMultiply(alpha);
	petsc_utils::tree2vec(invmed_data.temp,Y);


	
	//Vec alpha;
	PetscScalar alpha = (PetscScalar).00001;
	//VecDuplicate(Y,&alpha);
	//VecSet(alpha,sca);
	//ierr = VecPointwiseMult(alpha,alpha,U);
	ierr = VecAXPY(Y,alpha,U);CHKERRQ(ierr);
	// Output Vector ( Compute:  U + G[ \eta * U ] )
	//ierr = VecAXPY(Y,1,U);CHKERRQ(ierr);
	//ierr = VecDestroy(&alpha); CHKERRQ(ierr);
	

	//pvfmm::Profile::Toc();
	return 0;
}
		
#undef __FUNCT__
#define __FUNCT__ "tree2vec"
template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y){
	PetscErrorCode ierr;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());
		int SCAL_EXP = 1;

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

#undef __FUNCT__
#define __FUNCT__ "vec2tree"
template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree){
	std::cout << (tree->GetNodeList()).size() << std::endl;
	std::cout << "in the vec2tree" << std::endl;
	PetscErrorCode ierr;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	std::cout << "in the vec2tree" << std::endl;
	std::cout << "in the vec2tree1" << std::endl;
	std::cout << "in the vec2tree2" << std::endl;
	std::cout << "in the vec2tree3" << std::endl;
	std::cout << "in the vec2tree4" << std::endl;
	std::cout << "in the vec2tree5" << std::endl;
	std::cout << "in the vec2tree6" << std::endl;
	std::cout << "in the vec2tree7" << std::endl;
	std::cout << "in the vec2tree8" << std::endl;

	//std::vector<FMMNode_t*> nlist1 = tree->GetNodeList();
	std::cout << "in the vec2tree" << std::endl;
	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	std::cout << "in the vec2tree" << std::endl;
	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	std::cout << "in the vec2tree" << std::endl;
	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());
		int SCAL_EXP = 1;

		const PetscScalar *Y_ptr;
		ierr = VecGetArrayRead(Y, &Y_ptr);

		std::cout << "in the vec2tree" << std::endl;
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
