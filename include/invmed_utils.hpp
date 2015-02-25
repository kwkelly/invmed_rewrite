#include <cmath>
#include <cstdlib>
#include "petscsys.h"
#include "El.hpp"

#pragma once

/////////////////////////////////////////
// Declarations
/////////////////////////////////////////
struct InvMedData{
	InvMedTree<FMM_Mat_t>* phi_0;
	InvMedTree<FMM_Mat_t>* temp;
	pvfmm::PtFMM_Tree* pt_tree;
	std::vector<double> src_coord;
	PetscReal alpha;
};


struct ScatteredData{
	InvMedTree<FMM_Mat_t>* eta;
	InvMedTree<FMM_Mat_t>* temp;
	PetscReal alpha;
};

struct QRData{
	Mat *A;
	Mat *Q;
	Mat *R;
};

#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y);

#undef __FUNCT__
#define __FUNCT__ "fullmult"
int fullmult(Mat M, Vec U, Vec Y);

#undef __FUNCT__
#define __FUNCT__ "scattermult"
int scattermult(Mat M, Vec U, Vec Y);

#undef __FUNCT__
#define __FUNCT__ "tree2vec"
template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y);

#undef __FUNCT__
#define __FUNCT__ "vec2tree"
template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree);

#undef __FUNCT__
#define __FUNCT__ "mgs"
PetscErrorCode mgs(QRData &qr_data);


#undef __FUNCT__
#define __FUNCT__ "MatSetColumnVector"
PetscErrorCode  MatSetColumnVector(Mat A,Vec yy,PetscInt col);

void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k);
void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void nonsingular_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

std::vector<double> randsph(int n_points, double rad);
std::vector<double> randunif(int n_points);
std::vector<double> equicube(int n_points);


const pvfmm::Kernel<double> helm_kernel=pvfmm::BuildKernel<double, helm_kernel_fn>("helm_kernel", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> helm_kernel_conj=pvfmm::BuildKernel<double, helm_kernel_conj_fn>("helm_kernel_conj", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> nonsingular_kernel=pvfmm::BuildKernel<double, nonsingular_kernel_fn>("nonsingular_kernel", 3, std::pair<int,int>(2,2));

std::vector<double> test_pts();

/////////////////////////////////////////
// Definitions
/////////////////////////////////////////

void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k){
#ifndef __MIC__
	pvfmm::Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(24*dof));
#endif
	for(int t=0;t<trg_cnt;t++){
		for(int i=0;i<dof;i++){
			double p[2]={0,0};
			for(int s=0;s<src_cnt;s++){
				double dX_reg=r_trg[3*t ]-r_src[3*s ];
				double dY_reg=r_trg[3*t+1]-r_src[3*s+1];
				double dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
				double R = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
				if (R!=0){
					R = sqrt(R);
					double invR=1.0/R;
					invR = invR/(4*const_pi<double>());
					double G[2]={cos(k*R)*invR, sin(k*R)*invR};
					p[0] += v_src[(s*dof+i)*2+0]*G[0] - v_src[(s*dof+i)*2+1]*G[1];
					p[1] += v_src[(s*dof+i)*2+0]*G[1] + v_src[(s*dof+i)*2+1]*G[0];
				}
			}
			k_out[(t*dof+i)*2+0] += p[0];
			k_out[(t*dof+i)*2+1] += p[1];
		}
	}
}

void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, 1);
};

void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, -1);
};


void nonsingular_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
#ifndef __MIC__
	pvfmm::Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(24*dof));
#endif
	for(int t=0;t<trg_cnt;t++){
		for(int i=0;i<dof;i++){
			double p[2]={0,0};
			for(int s=0;s<src_cnt;s++){
				double dX_reg=r_trg[3*t ]-r_src[3*s ];
				double dY_reg=r_trg[3*t+1]-r_src[3*s+1];
				double dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
				double R = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
				if (R!=0){
					R = sqrt(R);
					double invR=1.0/R;
					invR = invR/(4*const_pi<double>());
					double G[2]={cos(R), sin(R)};
					p[0] += v_src[(s*dof+i)*2+0]*G[0] - v_src[(s*dof+i)*2+1]*G[1];
					p[1] += v_src[(s*dof+i)*2+0]*G[1] + v_src[(s*dof+i)*2+1]*G[0];
				}
			}
			k_out[(t*dof+i)*2+0] += p[0];
			k_out[(t*dof+i)*2+1] += p[1];
		}
	}
}

std::vector<double> randsph(int n_points, double rad){
	std::vector<double> src_points;
	std::vector<double> z;
	std::vector<double> phi;
	std::vector<double> theta;
	double val;

	for (int i = 0; i < n_points; i++) {
		val = 2*rad*((double)std::rand()/(double)RAND_MAX) - rad;
		z.push_back(val);
		theta.push_back(asin(z[i]/rad));
		val = 2*M_PI*((double)std::rand()/(double)RAND_MAX);
		phi.push_back(val);
		src_points.push_back(rad*cos(theta[i])*cos(phi[i])+.5);
		src_points.push_back(rad*cos(theta[i])*sin(phi[i])+.5);
		src_points.push_back(z[i]+.5);
	}

	return src_points;
}

std::vector<double> randunif(int n_points){
	double val;
	std::vector<double> src_points;

	for (int i = 0; i < n_points; i++) {
		val = ((double)std::rand()/(double)RAND_MAX);
		src_points.push_back(val);
		val = ((double)std::rand()/(double)RAND_MAX);
		src_points.push_back(val);
		val = ((double)std::rand()/(double)RAND_MAX);
		src_points.push_back(val);
	}

	return src_points;
}


std::vector<double> equicube(int n_points){
	double val;
	std::vector<double> src_points;
	double spacing = 1.0/(n_points + 1);

	for (int i = 0; i < n_points; i++) {
	for (int j = 0; j < n_points; j++) {
	for (int k = 0; k < n_points; k++) {
		src_points.push_back(spacing*(i+1));
		src_points.push_back(spacing*(j+1));
		src_points.push_back(spacing*(k+1));
	}
	}
	}

	return src_points;
}


std::vector<double> test_pts(){
	std::vector<double> pts;
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.3000);

	pts.push_back( 0.5000);
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
	
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.7000);
	
	pts.push_back( 0.5000);
	pts.push_back( 0.7000);
	pts.push_back( 0.5000);
	
	pts.push_back( 0.7000);
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	return pts;
}


std::vector<double> equisph(int n_points, double rad){
	// Generate n_points equidistributed points on the
	// surface of a sphere of radius rad
	// http://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
	std::vector<double> points;
	
	int n_count = 0;
	double a = 4*M_PI*rad*rad/n_points;
	double d = sqrt(a);
	double M_theta  = round(M_PI/d);
	double d_theta = M_PI/M_theta;
	double d_phi = a/d_theta;
	for(int m=0;m<M_theta;m++){
		double theta = M_PI*(m+0.5)/M_theta;
		double M_phi = round(2*M_PI*sin(theta)/d_phi);
		for(int n=0;n<M_phi;n++){
			double phi = 2*M_PI*n/M_phi;
			points.push_back(rad*sin(theta)*cos(phi)+.5);
			points.push_back(rad*sin(theta)*sin(phi)+.5);
			points.push_back(rad*cos(theta)+.5);
			n_count++;
		}
	}
	std::cout << "This many points were created: " << n_count << std::endl;
	return points;
			
}



#undef __FUNCT__
#define __FUNCT__ "fullmult"
int fullmult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	// Get context
	InvMedData *invmed_data = NULL;
	MatShellGetContext(M, &invmed_data);
	InvMedTree<FMM_Mat_t>* phi_0 = invmed_data->phi_0;
	InvMedTree<FMM_Mat_t>* temp = invmed_data->temp;
	PetscReal alpha = invmed_data->alpha;

	const MPI_Comm* comm=invmed_data->phi_0->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	//int omp_p=omp_get_max_threads();
	vec2tree(U,temp);

	temp->Multiply(phi_0,1);

	// Run FMM ( Compute: G[ \eta * u ] )
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();

	// Regularize
	tree2vec(temp,Y);

	ierr = VecAXPY(Y,alpha,U);CHKERRQ(ierr);

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "scattermult"
int scattermult(Mat M, Vec U, Vec Y){

	// This function computes (I+N)u where 
	// Nu = \int G(x-y)k^2\eta(y)u(y)dy

	PetscErrorCode ierr;
	// Get context
	ScatteredData *scattered_data = NULL;
	MatShellGetContext(M, &scattered_data);
	InvMedTree<FMM_Mat_t>* eta = scattered_data->eta;
	InvMedTree<FMM_Mat_t>* temp = scattered_data->temp;
	PetscReal alpha = scattered_data->alpha;

	const MPI_Comm* comm=scattered_data->eta->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	//int omp_p=omp_get_max_threads();
	temp->ClearFMMData(); // not sure if this one is necessary... but it couldn't hurt, right?
	vec2tree(U,temp);

	temp->Multiply(eta,1);

	// Run FMM ( Compute: G[ \eta * u ] )
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();

	// Add u ... no regularization at this time.
	tree2vec(temp,Y);

	ierr = VecAXPY(Y,1,U);CHKERRQ(ierr);

	return 0;
}




#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	// Get context
	InvMedData *invmed_data = NULL;
	MatShellGetContext(M, &invmed_data);
	InvMedTree<FMM_Mat_t>* phi_0 = invmed_data->phi_0;
	InvMedTree<FMM_Mat_t>* temp = invmed_data->temp;
	pvfmm::PtFMM_Tree* pt_tree = invmed_data->pt_tree;
	std::vector<double> src_coord = invmed_data->src_coord;
	PetscReal alpha = invmed_data->alpha;

	const MPI_Comm* comm=invmed_data->phi_0->Comm();
	pvfmm::Profile::Tic("Mult",comm,true);
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	
	vec2tree(U,temp);

	temp->Multiply(phi_0,1);

	// Run FMM ( Compute: G[ \eta * u ] )
	pvfmm::Profile::Tic("Volume_FMM",comm,true);
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();
	pvfmm::Profile::Toc();

	// Sample at the points in src_coord, then apply the transpose
	// operator.
	std::vector<double> src_values = temp->ReadVals(src_coord);
	
	pvfmm::Profile::Tic("Particle_FMM",comm,true);
	pt_tree->ClearFMMData();
	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value, 0, &src_values);
	pvfmm::Profile::Toc();

	// Insert the values back in
	temp->Trg2Tree(trg_value);
	
	// Ptwise multiply by the conjugate of phi_0
	temp->ConjMultiply(phi_0,1);

	temp->Write2File("results/aftermult",0);


	tree2vec(temp,Y);

	// Regularize
	ierr = VecAXPY(Y,alpha,U);CHKERRQ(ierr);
	pvfmm::Profile::Toc();

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
		//std::cout << "TREE2VEC HERE " << n_coeff3 << " " << Y_size << " " << nlist.size() << " " << data_dof << " " << n_coeff3 << std::endl;
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
				for(size_t j=0;j<n_coeff3*data_dof;j++){
					Y_ptr[j+Y_offset]=coeff_vec[j]*s;
		//			std::cout << coeff_vec[j]*s << std::endl;
				}
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
	PetscErrorCode ierr;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		PetscInt Y_size;
		ierr = VecGetLocalSize(Y, &Y_size);
		int data_dof=Y_size/(n_coeff3*nlist.size());
		int SCAL_EXP = 1;

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

#undef __FUNCT__
#define __FUNCT__ "MatSetColumnVector"
PetscErrorCode  MatSetColumnVector(Mat A,Vec yy,PetscInt col)
{
	PetscScalar       *y;
	const PetscScalar *v;
	PetscErrorCode    ierr;
	PetscInt          i,j,nz,N,Rs,Re,rs,re;
	const PetscInt    *idx;

	PetscFunctionBegin;
	if (col < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Requested negative column: %D",col);
	ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
	if (col >= N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Requested column %D larger than number columns in matrix %D",col,N);
	ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);

	ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
	for(i=0;i<re -rs;i++){
		std::cout << i << std::endl;
		ierr= MatSetValue(A,i+rs,col,y[i], INSERT_VALUES);CHKERRQ(ierr);
	}

	ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}



PetscErrorCode mgs(QRData &qr_data){
	// Modified gram schmidt
	PetscErrorCode ierr;
	MPI_Comm comm;
	ierr = PetscObjectGetComm((PetscObject)*(qr_data.A),&comm);CHKERRQ(ierr);
	PetscInt m,n;
	MatGetSize(*(qr_data.A),&m,&n);
	std::vector<Vec> vectors(n);
	for(int i=0;i<n;i++){
		Vec v;
		VecCreateMPI(comm, PETSC_DECIDE,m,&v);
		vectors[i] = v;
		MatGetColumnVector(*(qr_data.A),vectors[i],i);
	}

	
	MatAssemblyBegin(*(qr_data.R),MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(*(qr_data.Q),MAT_FINAL_ASSEMBLY);
	Vec q_i;
	VecCreateMPI(comm,PETSC_DECIDE,m,&q_i);
	VecDuplicate(vectors[0],&q_i); // should all have the same size and partitioning
	for(int i=0; i< n; i++){
		std::cout << i << std::endl;
		PetscReal r_ii;
		ierr = VecNorm(vectors[i],NORM_2,&r_ii);CHKERRQ(ierr);
		ierr = VecCopy(vectors[i],q_i);CHKERRQ(ierr);
		ierr = VecScale(q_i,1/r_ii);
		ierr = VecScale(vectors[i],1/r_ii);
		MatSetValue(*(qr_data.R),i,i,r_ii,INSERT_VALUES);
		
		for(int j=i+1;j<n;j++){
			PetscScalar r_ij;
			ierr = VecDot(vectors[j],q_i,&r_ij);CHKERRQ(ierr);
			MatSetValue(*(qr_data.R),i,j,r_ij,INSERT_VALUES);
			ierr = VecAXPY(vectors[j],-r_ij,q_i);
		}
		ierr = MatSetColumnVector(*(qr_data).Q,vectors[i],i);CHKERRQ(ierr);
	}
	VecDestroy(&q_i);
	MatAssemblyEnd(*(qr_data.R),MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*(qr_data.Q),MAT_FINAL_ASSEMBLY);
	return ierr;
}

PetscErrorCode ortho_project(const std::vector<Vec> &ortho_set, Vec &other_vec){
	// Given an orthogonal set of vectors in ortho_set, project other vector onto
	// a subspace orthogonal to the one spanned by the ortho_set
	PetscErrorCode ierr;
	MPI_Comm comm;
	ierr = PetscObjectGetComm((PetscObject)ortho_set[0],&comm);CHKERRQ(ierr);
	int n = ortho_set.size();
	for(int i=0; i< n; i++){
		PetscScalar r_ij;
		ierr = VecDot(other_vec,ortho_set[i],&r_ij);CHKERRQ(ierr);
		ierr = VecAXPY(other_vec,-r_ij,ortho_set[i]);
	}
	//PetscReal r_ii;
	//ierr = VecNorm(other_vec,NORM_2,&r_ii);CHKERRQ(ierr);
	//ierr = VecScale(other_vec,1/r_ii);
	return ierr;
}

double randn(double mu, double sigma, const PetscRandom &r){

  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
			PetscReal val1;
			PetscReal val2;
			PetscRandomGetValueReal(r,&val1);
			PetscRandomGetValueReal(r,&val2);
      //U1 = -1 + ((double) val1 / RAND_MAX) * 2;
			U1 = -1 + val1*2;
      //U2 = -1 + ((double) val2 / RAND_MAX) * 2;
			U2 = -1 + val2*2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);

}



#undef __FUNCT__
#define __FUNCT__ "scatter_born"
template <class FMM_Mat_t>
void scatter_born(InvMedTree<FMM_Mat_t>* phi_0, InvMedTree<FMM_Mat_t>* scatterer, InvMedTree<FMM_Mat_t> *phi){
	// phi_0 is the incident field, phi is the scattered field


	std::cout << "does it go here" << std::endl;
	// -------------------------------------------------------------------
	// Compute phi using the Born approximation - u = u_0 - \int G(x-y)k^2\eta(y)u_0(y)dy
	// -------------------------------------------------------------------
	phi->Multiply(scatterer,-1);  
	phi->RunFMM();
	phi->Copy_FMMOutput();
	phi->Add(phi_0,1);
	//phi->Write2File("results/phi",0);
	return;
}

#undef __FUNCT__
#define __FUNCT__ "scatter_solve"
template <class FMM_Mat_t>
PetscErrorCode scatter_solve(InvMedTree<FMM_Mat_t>* phi_0, ScatteredData &scattered_data, InvMedTree<FMM_Mat_t> *phi){
	// -------------------------------------------------------------------
	// Compute phi by solving  u = u_0 - \int G(x-y)k^2\eta(y)u(y)dy
	// for u
	// -------------------------------------------------------------------


	const MPI_Comm* comm_const=phi_0->Comm();
	MPI_Comm* comm_ = const_cast<MPI_Comm*>(comm_const);
	MPI_Comm comm = *comm_;
	PetscErrorCode ierr;
	PetscInt m = phi_0->m;
	PetscInt M = phi_0->M;
	PetscInt n = phi_0->n;
	PetscInt N = phi_0->N;
	Mat A;
	MatCreateShell(comm,m,n,M,N,&scattered_data,&A);
	MatShellSetOperation(A,MATOP_MULT,(void(*)(void))scattermult);
	PetscReal TOL = 1e-6; // maybe make this one an input parameter
	PetscInt MAX_ITER = 100;

	Vec sol ,rhs;
	VecCreateMPI(comm,n,PETSC_DETERMINE,&sol);
	VecCreateMPI(comm,n,PETSC_DETERMINE,&rhs);
	tree2vec(phi_0,rhs);
	//VecView(rhs, PETSC_VIEWER_STDOUT_SELF);

	KSP ksp;
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
	//ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

	KSPSetType(ksp  ,KSPGMRES);
	//KSPSetNormType(ksp  , KSP_NORM_UNPRECONDITIONED);
	/*
	 * PetscErrorCode  KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits)
	 *
	 * ksp 	- the Krylov subspace context
	 * rtol 	- the relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm
	 * abstol 	- the absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm
	 * dtol 	- the divergence tolerance, amount (possibly preconditioned) residual norm can increase before KSPConvergedDefault() concludes that the method is diverging
	 * maxits 	- maximum number of iterations to use
	 */
	ierr = KSPSetTolerances(ksp  ,TOL  ,PETSC_DEFAULT,PETSC_DEFAULT,MAX_ITER); CHKERRQ(ierr);
	// What type of CG should this be... I think hermitian??

	// SET CG OR GMRES OPTIONS
	ierr = KSPGMRESSetRestart(ksp  , MAX_ITER  ); CHKERRQ(ierr);
	ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization); CHKERRQ(ierr);
	//ierr = KSPSetFromOptions(ksp  );CHKERRQ(ierr);
	ierr = KSPMonitorSet(ksp, KSPMonitorDefault, NULL, NULL); CHKERRQ(ierr);

	double time_ksp;
	int    iter_ksp;
	// -------------------------------------------------------------------
	// Solve the linear system
	// -------------------------------------------------------------------
	pvfmm::Profile::Tic("KSPSolve",&comm,true);
	time_ksp=-omp_get_wtime();
	ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);
	MPI_Barrier(comm);
	time_ksp+=omp_get_wtime();
	pvfmm::Profile::Toc();

	KSPConvergedReason reason;
	ierr=KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
	std::cout << reason << std::endl;
	ierr=PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);CHKERRQ(ierr);

	// View info about the solver
	KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	vec2tree(sol,phi);
	phi->Write2File("results/phi",0);

	VecDestroy(&rhs);
	VecDestroy(&sol);
	MatDestroy(&A);

	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "vec2elmatcol"
PetscErrorCode Vec2ElMatCol(const Vec &v, El::DistMatrix<double> &A, const int col){
	PetscErrorCode ierr;
	int local_sz;
	int high;
	int low;
	double *vec_arr;
	ierr = VecGetArray(v,&vec_arr); CHKERRQ(ierr);
	ierr = VecGetLocalSize(v,&local_sz); CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(v,&low,&high); CHKERRQ(ierr);
	#pragma omp parallel for
	for(int i=0;i<local_sz;i++){
		A.Set(low+i,col,vec_arr[i]); // global set
	}
	ierr = VecRestoreArray(v,&vec_arr);
	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "elmatcol2vec"
PetscErrorCode ElMatCol2Vec(Vec &v, const El::DistMatrix<double> &A, const int col){
	PetscErrorCode ierr;
	int local_sz;
	int high;
	int low;
	double *vec_arr;
	double val;
	int gl_sz;
	ierr = VecGetSize(v,&gl_sz);
	ierr = VecGetArray(v,&vec_arr); CHKERRQ(ierr);
	ierr = VecGetLocalSize(v,&local_sz); CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(v,&low,&high); CHKERRQ(ierr);
	#pragma omp parallel for
	for(int i=0;i<local_sz;i++){
		val = A.Get(low+i,col); //global get
		std::cout << val << std::endl;
		vec_arr[i] = val;
	}
	ierr = VecRestoreArray(v,&vec_arr);
	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "elmat2vecs"
PetscErrorCode ElMat2Vecs(std::vector<Vec> &v, const El::DistMatrix<double> &A){
	PetscErrorCode ierr;

	MPI_Comm comm;
	int rank;
	ierr = PetscObjectGetComm((PetscObject)v[0],&comm);CHKERRQ(ierr);
	MPI_Comm_rank(comm, &rank);
	int n_vecs = v.size();
	for(int i=0;i<n_vecs;i++){
		ElMatCol2Vec(v[i], A, i);
	}
	return ierr;
}
