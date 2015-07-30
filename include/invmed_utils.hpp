#include <cmath>
#include <cstdlib>
#include "petscsys.h"
#include "El.hpp"
#include "par_scan/gen_scan.hpp"
//#include "math_utils.hpp"

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
	bool filter;
};


struct G_data{
	InvMedTree<FMM_Mat_t>* mask;
	InvMedTree<FMM_Mat_t>* temp;
	pvfmm::PtFMM_Tree* pt_tree;
	std::vector<double> src_coord;
	bool filter;
	MPI_Comm comm;
};

struct U_data{
	InvMedTree<FMM_Mat_t>* mask;
	InvMedTree<FMM_Mat_t>* temp;
	InvMedTree<FMM_Mat_t>* temp_c;
	std::vector<double> src_coord;
	pvfmm::BoundaryType bndry;
	const pvfmm::Kernel<double>* kernel;
	void (*fn)(const  double* coord, int n, double* out);
	std::vector<double> *coeffs;
	int trg_coord_size;
	MPI_Comm comm;
	pvfmm::PtFMM_Tree* pt_tree;
	int n_local_pt_srcs;
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


int comp_alltoall_sizes(const std::vector<int> &input_sizes, const std::vector<int> &output_sizes, std::vector<int> &sendcnts, std::vector<int> &sdispls, std::vector<int> &recvcnts, std::vector<int> &rdispls, MPI_Comm comm);

template <typename T>
void op(T& v1, const T& v2);


struct RandQRData{
	Mat *A;
	Mat *Atrans;
	El::DistMatrix<double>*  Q;
	El::DistMatrix<double>*  R_tilde;
	El::DistMatrix<double>*  Q_tilde;
	PetscRandom r;
	MPI_Comm comm;
	El::Grid *grid;
	int m;
	int n;
	int M;
	int N;
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


void helm_kernel3_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void helm_kernel_conj3_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);


void helm_kernel_low_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void helm_kernel_conj_low_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);




void nonsingular_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

std::vector<double> randsph(int n_points, double rad);
std::vector<double> randunif(int n_points);
std::vector<double> equicube(int n_points);
std::vector<double> equiplane(int n_points, int plane, double pos);

const pvfmm::Kernel<double> helm_kernel=pvfmm::BuildKernel<double, helm_kernel_fn>("helm_kernel", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> helm_kernel_conj=pvfmm::BuildKernel<double, helm_kernel_conj_fn>("helm_kernel_conj", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> helm_kernel3=pvfmm::BuildKernel<double, helm_kernel3_fn>("helm_kernel3", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> helm_kernel_conj3=pvfmm::BuildKernel<double, helm_kernel_conj3_fn>("helm_kernel_conj3", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> nonsingular_kernel=pvfmm::BuildKernel<double, nonsingular_kernel_fn>("nonsingular_kernel", 3, std::pair<int,int>(2,2));

const pvfmm::Kernel<double> helm_kernel_low=pvfmm::BuildKernel<double, helm_kernel_low_fn>("helm_kernel_low", 3, std::pair<int,int>(2,2));
const pvfmm::Kernel<double> helm_kernel_conj_low=pvfmm::BuildKernel<double, helm_kernel_conj_low_fn>("helm_kernel_conj_low", 3, std::pair<int,int>(2,2));

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
					invR = invR/(4*pvfmm::const_pi<double>());
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


void helm_kernel3_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, 3);
};

void helm_kernel_conj3_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, -3);
};


void helm_kernel_low_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, 0.1);
};

void helm_kernel_conj_low_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr){
	helm_kernel_fn_var(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, k_out, mem_mgr, -0.1);
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
					invR = invR/(4*M_PI);
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

std::vector<double> randunif(int n_points, double lmin, double lmax){
	double val;
	std::vector<double> src_points;

	for (int i = 0; i < n_points; i++) {
		val = ((double)std::rand()/(double)RAND_MAX);
		val = val*(lmax-lmin) + lmin;
		src_points.push_back(val);
		val = ((double)std::rand()/(double)RAND_MAX);
		val = val*(lmax-lmin) + lmin;
		src_points.push_back(val);
		val = ((double)std::rand()/(double)RAND_MAX);
		val = val*(lmax-lmin) + lmin;
		src_points.push_back(val);
	}

	return src_points;
}

enum DistribType{
  UnifGrid,
  RandUnif,
  RandGaus,
  RandElps,
  RandSphr
};


template <class Real_t>
std::vector<Real_t> unif_point_distrib(size_t N, Real_t lmin, Real_t lmax, MPI_Comm comm){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  std::vector<Real_t> coord;
    {
      size_t NN=(size_t)round(pow((double)N,1.0/3.0));
      size_t N_total=NN*NN*NN;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;
      for(size_t i=start;i<end;i++){
        coord.push_back( (lmax-lmin) * (((Real_t)((i/  1    )%NN)+0.5)/NN) + lmin);
        coord.push_back( (lmax-lmin) * (((Real_t)((i/ NN    )%NN)+0.5)/NN) + lmin);
        coord.push_back( (lmax-lmin) * (((Real_t)((i/(NN*NN))%NN)+0.5)/NN) + lmin);
      }
    }
	return coord;
}


template <class Real_t>
std::vector<Real_t> rand_unif_point_distrib(size_t N, Real_t lmin, Real_t lmax, MPI_Comm comm){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);
  static size_t seed=myrank+1; seed+=np;
  srand48(seed);

  std::vector<Real_t> coord;
    {
      size_t N_total=N;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;
      size_t N_local=end-start;
      coord.resize(N_local*3);

      for(size_t i=0;i<N_local*3;i++)
        coord[i]=(lmax-lmin)*((Real_t)drand48()) + lmin;
    }
		return coord;
}

template <class Real_t>
std::vector<Real_t> rand_sph_point_distrib(size_t N, Real_t rad, MPI_Comm comm){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);
  static size_t seed=myrank+1; seed+=np;
  srand48(seed);

	std::vector<Real_t> src_points;
	std::vector<Real_t> z;
	std::vector<Real_t> phi;
	std::vector<Real_t> theta;
	Real_t val;

	size_t N_total=N;
	size_t start= myrank   *N_total/np;
	size_t end  =(myrank+1)*N_total/np;
	size_t N_local=end-start;

	for (int i = 0; i < N_local; i++) {
		val = 2*rad*((size_t)drand48()) - rad;
		z.push_back(val);
		theta.push_back(asin(z[i]/rad));
		val = 2*M_PI*((size_t)drand48());
		phi.push_back(val);
		src_points.push_back(rad*cos(theta[i])*cos(phi[i])+.5);
		src_points.push_back(rad*cos(theta[i])*sin(phi[i])+.5);
		src_points.push_back(z[i]+.5);
	}

	return src_points;
}



std::vector<double> equicube(int n_points, double lmin, double lmax){
	double val;
	std::vector<double> src_points;
	double spacing = (lmax-lmin)/(n_points);
	double offset = lmin/n_points;

	for (int i = 0; i < n_points; i++) {
	for (int j = 0; j < n_points; j++) {
	for (int k = 0; k < n_points; k++) {
		src_points.push_back(offset + spacing*(i+1));
		src_points.push_back(offset + spacing*(j+1));
		src_points.push_back(offset + spacing*(k+1));
	}
	}
	}

	return src_points;
}

std::vector<double> equicube(int n_points){
	return equicube(n_points,0,1);
}


template <class Real_t>
std::vector<Real_t> unif_plane(size_t N, int plane, Real_t pos, MPI_Comm comm){
	int np, myrank;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	assert(plane == 0 or plane == 1 or plane == 2);

	// generate n_points points sources equally spaced on a plane
	// plane == 0 = yz plane
	// plane == 1 = xz plane
	// plane == 2 = xy plane

	std::vector<Real_t> coord;
	{
		size_t NN=(size_t)round(pow((double)N,1.0/2.0));
		double spacing = 1.0/(NN);
		size_t N_total=NN*NN;
		size_t start= myrank   *N_total/np;
		size_t end  =(myrank+1)*N_total/np;
		switch(plane){
			case 0:
				for(size_t i=start;i<end;i++){
					coord.push_back( pos                              );
					coord.push_back( (((Real_t)((i/ 1  )%NN)+0.5)/NN) );
					coord.push_back( (((Real_t)((i/(NN))%NN)+0.5)/NN) );
				}
				break;
			case 1:
				for(size_t i=start;i<end;i++){
					coord.push_back( (((Real_t)((i/  1 )%NN)+0.5)/NN) );
					coord.push_back( pos                               );
					coord.push_back( (((Real_t)((i/(NN))%NN)+0.5)/NN) );
				}
				break;
			case 2:
				for(size_t i=start;i<end;i++){
					coord.push_back( (((Real_t)((i/  1  )%NN)+0.5)/NN) );
					coord.push_back( (((Real_t)((i/ NN  )%NN)+0.5)/NN) );
					coord.push_back( pos                               );
				}
				break;
		}
	}
	return coord;
}




std::vector<double> equiplane(int n_points, int plane, double pos){
	// generate n_points^2 points sources equally spaced on a plane
	// plane == 0 = yz plane
	// plane == 1 = xz plane
	// plane == 2 = xy plane
	assert(plane == 0 or plane == 1 or plane == 2);
	std::vector<double> src_points;
	double spacing = 1.0/(n_points + 1);

  size_t n=(size_t)round(pow((double)n_points,1.0/2.0));
	n_points = n;

	if(plane == 0){
		for (int i = 0; i < n_points; i++) {
			for (int j = 0; j < n_points; j++) {
				src_points.push_back(pos);
				src_points.push_back(spacing*(i+1));
				src_points.push_back(spacing*(j+1));
			}
		}
	}
	else if(plane == 1){
		for (int i = 0; i < n_points; i++) {
			for (int j = 0; j < n_points; j++) {
				src_points.push_back(spacing*(i+1));
				src_points.push_back(pos);
				src_points.push_back(spacing*(j+1));
			}
		}
	}
	else if(plane == 2){
		for (int i = 0; i < n_points; i++) {
			for (int j = 0; j < n_points; j++) {
				src_points.push_back(spacing*(i+1));
				src_points.push_back(spacing*(j+1));
				src_points.push_back(pos);
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
	temp->Write2File("../results/mult_temp",0);
	
	for(int i = 0;i<src_coord.size();i++){
		std::cout << src_coord[i] << std::endl;
	}
	std::cout << "========================" << std::endl;
	std::vector<double> src_values = temp->ReadVals(src_coord);
	for(int i = 0;i<src_values.size();i++){
		std::cout << src_values[i] << std::endl;
	}
	
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
	//std::cout << "v2t" << std::endl;
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

				//std::cout << n_coeff3 << std::endl;
				//std::cout << data_dof << std::endl;
				size_t Y_offset=i*n_coeff3*data_dof;
				for(size_t j=0;j<n_coeff3*data_dof;j++){
					//std::cout << j << " : " << PetscRealPart(Y_ptr[j+Y_offset])*s << std::endl;
					coeff_vec[j]=PetscRealPart(Y_ptr[j+Y_offset])*s;
				}
				nlist[i]->DataDOF()=data_dof;
			}
		}
		//ierr = VecRestoreArrayRead(Y, &Y_ptr); // wasnt here to begin with
	}
	//std::cout << "v2t end" << std::endl;

	return 0;
}


template<typename T>
std::ostream &operator<<(std::ostream &stream, std::vector<T> ob)
{
	for(int i=0;i<ob.size();i++){
		stream << ob[i] << '\n';
	}
	return stream;
}


std::vector<int> exscan(std::vector<int> x){
	int ll = x.size();
	std::vector<int> y(ll);
	y[0] = 0;

	for(int i=1;i<ll;i++){
		y[i] = y[i-1] + x[i-1];
	}

	return y;
}


/*
#undef __FUNCT__
#define __FUNCT__ "elemental2tree"
template <class FMM_Mat_t, typename T>
int elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, InvMedTree<FMM_Mat_t> *tree){
	
	assert((Y.DistData().colDist == El::STAR) and (Y.DistData().rowDist == El::VC));

	int data_dof=2;
	int SCAL_EXP = 1;

	int nlocal, nstart; // petsc vec info
	double *pt_array,*pt_perm_array;
	int r,q,ll,rq; // el vec info
	int nbigs; //Number of large recv (i.e. recv 1 extra data point)
	int pstart; // p_id of nstart
	int p = El::mpi::WorldRank(); //p_id
	int recv_size; // base recv size
	bool print = p == -1; 

	// Get el vec info
	ll = Y.Height();
	const El::Grid* g = &(Y.Grid());
	r = g->Height();
	q = g->Width();
	MPI_Comm comm = (g->Comm()).comm;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
	
	// Get petsc vec params
	//VecGetLocalSize(pt_vec,&nlocal);
	nlocal = (tree->m)/data_dof;
	//VecGetArray(pt_vec,&pt_array);
	//VecGetOwnershipRange(pt_vec,&nstart,NULL);
	MPI_Exscan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,comm);

	// Determine who owns the first element we want
	rq = r * q;
	pstart = nstart % rq;
	nbigs = nlocal % rq;
	recv_size = nlocal / rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "recv_size: " << recv_size << std::endl;
	}

	// Make recv sizes
	std::vector<int> recv_lengths(rq);
	std::fill(recv_lengths.begin(),recv_lengths.end(),recv_size);
	if(nbigs >0){
		for(int i=0;i<nbigs;i++){
			recv_lengths[(pstart + i) % rq] += 1;
		}
	}

	// Make recv disps
	std::vector<int> recv_disps = exscan(recv_lengths);

	// All2all to get send sizes
	std::vector<int> send_lengths(rq);
	MPI_Alltoall(&recv_lengths[0], 1, MPI_INT, &send_lengths[0], 1, MPI_INT,comm);

	// Scan to get send_disps
	std::vector<int> send_disps = exscan(send_lengths);

	// Do all2allv to get data on correct processor
	std::vector<El::Complex<double>> recv_data(nlocal);
	std::vector<El::Complex<double>> recv_data_ordered(nlocal);
	//MPI_Alltoallv(el_vec.Buffer(),&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
			&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);
	El::mpi::AllToAll(Y.LockedBuffer(), &send_lengths[0], &send_disps[0], &recv_data[0],&recv_lengths[0],&recv_disps[0],comm);
	
	if(print){
		//std::cout << "Send data: " <<std::endl << *el_vec.Buffer() <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}
	
	// Reorder the data so taht it is in the right order for the fmm tree
	for(int p=0;p<rq;p++){
		int base_idx = (p - pstart + rq) % rq;
		int offset = recv_disps[p];
		for(int i=0;i<recv_lengths[p];i++){
			recv_data_ordered[base_idx + rq*i] = recv_data[offset + i];
		}
	}

	// loop through and put the data into the tree

	#pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist.size()* tid   )/omp_p;
		size_t i_end  =(nlist.size()*(tid+1))/omp_p;
		for(size_t i=i_start;i<i_end;i++){
			pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
			double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

			size_t Y_offset=i*n_coeff3;
			for(size_t j=0;j<n_coeff3;j++){
				double real = El::RealPart(recv_data_ordered[j+Y_offset])*s;
				double imag = El::ImagPart(recv_data_ordered[j+Y_offset])*s;
				coeff_vec[j]=real;
				coeff_vec[j+n_coeff3]=imag;
			}
			nlist[i]->DataDOF()=2;
		}
	}

	if(print){std::cout <<"here?"<<std::endl;}

	return 0;

}

*/

#undef __FUNCT__
#define __FUNCT__ "elemental2tree"
template <class FMM_Mat_t, typename T>
int elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, InvMedTree<FMM_Mat_t> *tree){
	
	assert((Y.DistData().colDist == El::STAR) and (Y.DistData().rowDist == El::VC));

	PetscErrorCode ierr;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int rank, size;
	MPI_Comm_size(*comm,&size);
	MPI_Comm_rank(*comm,&rank);

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;


	{
		int data_dof=2;
		int SCAL_EXP = 1;

		// get the input and output sizes for everyone
		std::vector<int> input_sizes(size);
		std::vector<int> output_sizes(size);
		int m = (tree->m)/data_dof;
		int el_l_sz = Y.LocalHeight();

		MPI_Allgather(&el_l_sz, 1, MPI_INT,&input_sizes[0], 1, MPI_INT,*comm);
		MPI_Allgather(&m, 1, MPI_INT,&output_sizes[0], 1, MPI_INT,*comm);

		std::vector<int> sendcnts;
		std::vector<int> sdispls;
		std::vector<int> recvcnts;
		std::vector<int> rdispls;

		std::vector<El::Complex<double>> indata(input_sizes[rank]);
		indata.assign(Y.LockedBuffer(),Y.LockedBuffer()+indata.size());
		std::vector<El::Complex<double>> outdata(output_sizes[rank]);

		comp_alltoall_sizes(input_sizes, output_sizes, sendcnts, sdispls, recvcnts, rdispls, *comm);

		El::mpi::AllToAll(&indata[0], &sendcnts[0], &sdispls[0], &outdata[0],&recvcnts[0],&rdispls[0],*comm);

		#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				double s=std::pow(2.0,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=i*n_coeff3;
				for(size_t j=0;j<n_coeff3;j++){
					double real = El::RealPart(outdata[j+Y_offset])*s;
					double imag = El::ImagPart(outdata[j+Y_offset])*s;
					coeff_vec[j]=real;
					coeff_vec[j+n_coeff3]=imag;
				}
				nlist[i]->DataDOF()=2;
			}
		}
	}

	MPI_Barrier(*comm);
	return 0;
}

template <typename T>
void op(T& v1, const T& v2){
	v1+=v2;
}

int comp_alltoall_sizes(const std::vector<int> &input_sizes, const std::vector<int> &output_sizes, std::vector<int> &sendcnts, std::vector<int> &sdispls, std::vector<int> &recvcnts, std::vector<int> &rdispls, MPI_Comm comm){
	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// compute size differences
	std::vector<int> size_diff(size);

	#pragma omp parallel for
	for(int i=0;i<size;i++){
		size_diff[i] = input_sizes[i] - output_sizes[i];
	}

	// first we compute the sendcnts
	sendcnts.clear();
	sendcnts.resize(size);
	std::fill(sendcnts.begin(),sendcnts.end(),0);

	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			if(rank == i && i == j) sendcnts[j] = (output_sizes[j] < input_sizes[j]) ? output_sizes[j] : input_sizes[j];
			else{ // then we can take away from this one
				if((size_diff[i] >= 0) && size_diff[j] < 0){
					int snd = std::min(abs(size_diff[j]),abs(size_diff[i]));
					size_diff[i] -= snd;
					size_diff[j] += snd;
					if(i == rank){
						sendcnts[j] = snd;
					}
				}
			}
		}
	}

	// reset the difference array
	#pragma omp parallel for
	for(int i=0;i<size;i++){
		size_diff[i] = input_sizes[i] - output_sizes[i];
	}
	recvcnts.clear();
	recvcnts.resize(size);
	std::fill(recvcnts.begin(),recvcnts.end(),0);

	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			if(rank == i && i == j) recvcnts[j] = (output_sizes[j] < input_sizes[j]) ? output_sizes[j] : input_sizes[j];
			else{ // then we can take away from this one
				if((size_diff[i] < 0) && size_diff[j] > 0){
					int recv = std::min(abs(size_diff[j]),abs(size_diff[i]));
					size_diff[i] += recv;
					size_diff[j] -= recv;
					if(i == rank){
						recvcnts[j] = recv;
					}
				}
			}
		}
	}

	sdispls = sendcnts;
	ex_scan(sdispls);

	rdispls = recvcnts;
	ex_scan(rdispls);

	/*
	if(!rank){
		for(int i=0;i<size;i++) std::cout << input_sizes[i] << " ";
		std::cout << std::endl;
		for(int i=0;i<size;i++) std::cout << output_sizes[i] << " ";
		std::cout << std::endl;
		for(int i=0;i<size;i++) std::cout << sendcnts[i] << " ";
		std::cout << std::endl;
		for(int i=0;i<size;i++) std::cout << recvcnts[i] << " ";
		std::cout << std::endl;
		for(int i=0;i<size;i++) std::cout << sdispls[i] << " ";
		std::cout << std::endl;
		for(int i=0;i<size;i++) std::cout << rdispls[i] << " ";
		std::cout << std::endl;
	}
	*/

	return 0;
}

/*
#undef __FUNCT__
#define __FUNCT__ "tree2elemental"
template <class FMM_Mat_t, typename T>
int tree2elemental(InvMedTree<FMM_Mat_t> *tree, El::DistMatrix<T,El::VC,El::STAR> &Y){

	int data_dof=2;
	int SCAL_EXP = 1;

	int nlocal,nstart,gsize; //local elements, start p_id, global size
	double *pt_array; // will hold local array
	int r,q,rq; //Grid sizes
	int nbigs; //Number of large sends (i.e. send 1 extra data point)
	int pstart; // p_id of nstart
	int p = El::mpi::WorldRank(); //p_id
	int send_size; // base send size
	bool print = p == -1; 


	// Get Grid and associated params
	const El::Grid* g = &(Y.Grid());
	r = g->Height();
	q = g->Width();
	MPI_Comm comm = (g->Comm()).comm;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	// Get sizes, array in petsc 
	//VecGetSize(pt_vec,&gsize);
	gsize = tree->M/data_dof;
	nlocal = tree->m/data_dof;
	//VecGetLocalSize(pt_vec,&nlocal);
	//VecGetArray(pt_vec,&pt_array);
	MPI_Exscan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,comm);
	//VecGetOwnershipRange(pt_vec,&nstart,NULL);

	//Find processor that nstart belongs to, number of larger sends
	rq = r * q;
	pstart = nstart % rq; //int div
	nbigs = nlocal % rq;
	send_size = nlocal/rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "send_size: " << send_size << std::endl;
	}

	// Make send_lengths
	std::vector<int> send_lengths(rq);
	std::fill(send_lengths.begin(),send_lengths.end(),send_size);
	if(nbigs >0){
		for(int j=0;j<nbigs;j++){
			send_lengths[(pstart + j) % rq] += 1;
		}
	}

	// Make send_disps
	std::vector<int> send_disps = exscan(send_lengths);

	std::vector<El::Complex<double>> indata(nlocal);
	// copy the data from an ffm tree to into a local vec of complex data for sending #pragma omp parallel for
	for(size_t tid=0;tid<omp_p;tid++){
		size_t i_start=(nlist.size()* tid   )/omp_p;
		size_t i_end  =(nlist.size()*(tid+1))/omp_p;
		for(size_t i=i_start;i<i_end;i++){
			pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
			double s=std::pow(0.5,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

			size_t Y_offset=(i)*n_coeff3;
			for(size_t j=0;j<n_coeff3;j++){
				double real = coeff_vec[j]*s; // local indices as in the pvfmm trees
				double imag = coeff_vec[j+n_coeff3]*s;
				El::Complex<double> val;
				El::SetRealPart(val,real);
				El::SetImagPart(val,imag);

				indata[Y_offset+j] = val;
			}
		}
	}


	// Make send_data
	std::vector<El::Complex<double>> send_data(nlocal);
	for(int proc=0;proc<rq;proc++){
		int offset = send_disps[proc];
		int base_idx = (proc - pstart + rq) % rq; 
		for(int j=0; j<send_lengths[proc]; j++){
			int idx = base_idx + (j * rq);
			send_data[offset + j] = indata[idx];
		}
	}

	// Do all2all to get recv_lengths
	std::vector<int> recv_lengths(rq);
	MPI_Alltoall(&send_lengths[0], 1, MPI_INT, &recv_lengths[0], 1, MPI_INT,comm);

	// Scan to get recv_disps
	std::vector<int> recv_disps = exscan(recv_lengths);

	// Do all2allv to get data on correct processor
	El::Complex<double> * recv_data = Y.Buffer();
	//MPI_Alltoallv(&send_data[0],&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
	//		&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);
	El::mpi::AllToAll(&send_data[0], &send_lengths[0], &send_disps[0], &recv_data[0],&recv_lengths[0],&recv_disps[0],comm);

	if(print){
		std::cout << "Send data: " <<std::endl << send_data <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}

	return 0;
}
*/

#undef __FUNCT__
#define __FUNCT__ "tree2elemental"
template <class FMM_Mat_t, typename T>
int tree2elemental(InvMedTree<FMM_Mat_t> *tree, El::DistMatrix<T,El::VC,El::STAR> &Y){

	assert((Y.DistData().colDist == El::STAR) and (Y.DistData().rowDist == El::VC));

	PetscErrorCode ierr;
	int cheb_deg=InvMedTree<FMM_Mat_t>::cheb_deg;
	const MPI_Comm* comm=tree->Comm();
	int rank;
	int size;
	MPI_Comm_rank(*comm,&rank);
	MPI_Comm_size(*comm,&size);

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;


	{
		int data_dof=2;
		int SCAL_EXP = 1;

		// get the input and output sizes for everyone
		std::vector<int> input_sizes(size);
		std::vector<int> output_sizes(size);
		int m = (tree->m)/data_dof;
		int el_l_sz = Y.LocalHeight();

		MPI_Allgather(&m, 1, MPI_INT,&input_sizes[0], 1, MPI_INT,*comm);
		MPI_Allgather(&el_l_sz, 1, MPI_INT,&output_sizes[0], 1, MPI_INT,*comm);

		std::vector<int> sendcnts;
		std::vector<int> sdispls;
		std::vector<int> recvcnts;
		std::vector<int> rdispls;

		std::vector<El::Complex<double>> indata(input_sizes[rank]);
		std::vector<El::Complex<double>> outdata(output_sizes[rank]);

		comp_alltoall_sizes(input_sizes, output_sizes, sendcnts, sdispls, recvcnts, rdispls, *comm);

		#pragma omp parallel for
		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();
				double s=std::pow(0.5,COORD_DIM*nlist[i]->Depth()*0.5*SCAL_EXP);

				size_t Y_offset=(i)*n_coeff3;
				for(size_t j=0;j<n_coeff3;j++){
					double real = coeff_vec[j]*s; // local indices as in the pvfmm trees
					double imag = coeff_vec[j+n_coeff3]*s;
					El::Complex<double> val;
					El::SetRealPart(val,real);
					El::SetImagPart(val,imag);

					indata[Y_offset+j] = val;
				}
			}
		}

		El::mpi::AllToAll(&indata[0], &sendcnts[0], &sdispls[0], &outdata[0],&recvcnts[0],&rdispls[0],*comm);

		for(int i=0;i<outdata.size();i++){
			Y.Set(i*size+rank,0,outdata[i]);
		}
	}
	MPI_Barrier(*comm);
	return 0;
}



#undef __FUNCT__
#define __FUNCT__ "vec2elemental"
int vec2elemental(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::VC,El::STAR > &Y){

	const El::Grid& g = Y.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int data_dof = 2;

	// get the input and output sizes for everyone
	std::vector<int> input_sizes(size);
	std::vector<int> output_sizes(size);
	int m = (vec.size())/data_dof;
	int el_l_sz = Y.LocalHeight();
	int t_sz_el;
	int t_sz_vec;
	MPI_Reduce(&el_l_sz, &t_sz_el, 1, MPI_INT,MPI_SUM, 0, comm);
	MPI_Reduce(&m, &t_sz_vec, 1, MPI_INT,MPI_SUM, 0, comm);
	if(!rank) assert(t_sz_el == t_sz_vec);

	// check the the total size doesn't change

	MPI_Allgather(&m, 1, MPI_INT,&input_sizes[0], 1, MPI_INT,comm);
	MPI_Allgather(&el_l_sz, 1, MPI_INT,&output_sizes[0], 1, MPI_INT,comm);

	std::vector<int> sendcnts;
	std::vector<int> sdispls;
	std::vector<int> recvcnts;
	std::vector<int> rdispls;

	std::vector<El::Complex<double>> indata(input_sizes[rank]);
	#pragma omp parallel for
	for(int i=0;i<m;i++){
		double real = vec[2*i+0];
		double imag = vec[2*i+1];
		El::Complex<double> val;
		El::SetRealPart(val,real);
		El::SetImagPart(val,imag);

		indata[i] = val;
	}
	std::vector<El::Complex<double>> outdata(output_sizes[rank]);

	comp_alltoall_sizes(input_sizes, output_sizes, sendcnts, sdispls, recvcnts, rdispls, comm);

	El::mpi::AllToAll(&indata[0], &sendcnts[0], &sdispls[0], &outdata[0],&recvcnts[0],&rdispls[0],comm);

	El::Complex<double> *Y_ptr = Y.Buffer();
	int sz = outdata.size();
	#pragma omp parallel for
	for(int i=0;i<sz; i++){
		//double real = outdata[2*i];
		//double imag = outdata[2*i+1];
		//El::SetRealPart(Y_ptr[i],real);
		//El::SetImagPart(Y_ptr[i],imag);
		Y_ptr[i] = outdata[i];
	}

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "elemental2vec"
int elemental2vec(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, std::vector<double> &vec){

	const El::Grid& g = Y.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int data_dof = 2;
	// get the input and output sizes for everyone
	std::vector<int> input_sizes(size);
	std::vector<int> output_sizes(size);
	int m = (vec.size())/data_dof;
	int el_l_sz = Y.LocalHeight();
	// test the total size doesnt change
	int t_sz_el;
	int t_sz_vec;
	MPI_Reduce(&el_l_sz, &t_sz_el, 1, MPI_INT,MPI_SUM, 0, comm);
	MPI_Reduce(&m, &t_sz_vec, 1, MPI_INT,MPI_SUM, 0, comm);
	if(!rank) assert(t_sz_el == t_sz_vec);

	MPI_Allgather(&el_l_sz, 1, MPI_INT,&input_sizes[0], 1, MPI_INT,comm);
	MPI_Allgather(&m, 1, MPI_INT,&output_sizes[0], 1, MPI_INT,comm);

	std::vector<int> sendcnts;
	std::vector<int> sdispls;
	std::vector<int> recvcnts;
	std::vector<int> rdispls;

	std::vector<El::Complex<double>> indata(input_sizes[rank]);
	indata.assign(Y.LockedBuffer(),Y.LockedBuffer()+indata.size());
	std::vector<El::Complex<double>> outdata(output_sizes[rank]);

	comp_alltoall_sizes(input_sizes, output_sizes, sendcnts, sdispls, recvcnts, rdispls, comm);

	El::mpi::AllToAll(&indata[0], &sendcnts[0], &sdispls[0], &outdata[0],&recvcnts[0],&rdispls[0],comm);


	//El::Complex<double> *Y_ptr = Y.Buffer();
	int sz = outdata.size();
	#pragma omp parallel for
	for(int i=0;i<sz; i++){
		vec[2*i] = El::RealPart(outdata[i]);
		vec[2*i+1] = El::ImagPart(outdata[i]);
	}

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "elstar2vec"
int elstar2vec(const El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> &Y, std::vector<double> &vec){

	const El::Complex<double> *Y_ptr = Y.LockedBuffer();
	int sz = vec.size()/2;
	#pragma omp parallel for
	for(int i=0;i<sz; i++){
		vec[2*i] = El::RealPart(Y_ptr[i]);
		vec[2*i+1] = El::ImagPart(Y_ptr[i]);
	}

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "vec2elstar"
int vec2elstar(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::STAR,El::STAR > &Y){

	El::Complex<double> *Y_ptr = Y.Buffer();
	int sz = vec.size()/2;
	#pragma omp parallel for
	for(int i=0;i<sz; i++){
		double real = vec[2*i];
		double imag = vec[2*i+1];
		El::SetRealPart(Y_ptr[i],real);
		El::SetImagPart(Y_ptr[i],imag);
	}

	return 0;
}


template <class FMM_Mat_t>
int printcoeffs(InvMedTree<FMM_Mat_t> *tree){
	std::cout << "START Print Coeffs============================" << std::endl;
	const MPI_Comm* comm=tree->Comm();
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;

	std::vector<FMMNode_t*> nlist = tree->GetNGLNodes();

	int omp_p=omp_get_max_threads();
	size_t n_coeff3=(cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;

	{
		int SCAL_EXP = 1;

		for(size_t tid=0;tid<omp_p;tid++){
			size_t i_start=(nlist.size()* tid   )/omp_p;
			size_t i_end  =(nlist.size()*(tid+1))/omp_p;
			for(size_t i=i_start;i<i_end;i++){
				pvfmm::Vector<double>& coeff_vec=nlist[i]->ChebData();

				for(size_t j=0;j<n_coeff3*2;j++){
					std::cout << coeff_vec[j] << std::endl;
				}
			}
		}
	}

	std::cout << "END Print Coeffs============================" << std::endl;
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
		//std::cout << i << std::endl;
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

	// -------------------------------------------------------------------
	// Compute phi using the Born approximation - u = u_0 + \int G(x-y)k^2\scatterer(y)u_0(y)dy
	// -------------------------------------------------------------------
	phi->Multiply(scatterer,1);  
	phi->RunFMM();
	phi->Copy_FMMOutput();
	phi->Add(phi_0,1);
	//phi->Write2File("results/phi",0);
	return;
}


#undef __FUNCT__
#define __FUNCT__ "scatter_born"
template <class FMM_Mat_t>
void scatter_born_scattered(InvMedTree<FMM_Mat_t>* phi, InvMedTree<FMM_Mat_t>* scatterer){
	// phi_0 is the incident field, phi is the scattered field

	// -------------------------------------------------------------------
	// Compute phi using the Born approximation - u_s = \int G(x-y)k^2\scatterer(y)u_0(y)dy
	// -------------------------------------------------------------------
	phi->Multiply(scatterer,1);  
	phi->RunFMM();
	phi->Copy_FMMOutput();
	//phi->Write2File("results/phi",0);
	return;
}


#undef __FUNCT__
#define __FUNCT__ "scatter_solve"
template <class FMM_Mat_t>
PetscErrorCode scatter_solve(InvMedTree<FMM_Mat_t>* phi_0, ScatteredData &scattered_data, InvMedTree<FMM_Mat_t> *phi){
	// -------------------------------------------------------------------
	// Compute phi by solving  u = u_0 + \int G(x-y)k^2\eta(y)u(y)dy
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
	MPI_Comm comm;
	ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

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
	ierr = VecRestoreArray(v,&vec_arr); CHKERRQ(ierr);
	MPI_Barrier(comm);

	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "elmatcol2vec"
PetscErrorCode ElMatCol2Vec(Vec &v, const El::DistMatrix<double> &A, const int col){
	PetscErrorCode ierr;
	MPI_Comm comm;
	ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

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
		vec_arr[i] = val;
	}
	ierr = VecRestoreArray(v,&vec_arr); CHKERRQ(ierr);
	MPI_Barrier(comm);
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "elmat2vecs"
PetscErrorCode ElMat2Vecs(std::vector<Vec> &v, const El::DistMatrix<double> &A){
	PetscErrorCode ierr;

	MPI_Comm comm;
	int rank;
	ierr = PetscObjectGetComm((PetscObject)(v[0]),&comm);CHKERRQ(ierr);
	MPI_Comm_rank(comm, &rank);
	int n_vecs = v.size();
	for(int i=0;i<n_vecs;i++){
		ElMatCol2Vec(v[i], A, i);
	}
	MPI_Barrier(comm);
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "vecs2elmat"
PetscErrorCode Vecs2ElMat(const std::vector<Vec> &v, El::DistMatrix<double> &A){
	PetscErrorCode ierr;

	MPI_Comm comm;
	int rank;
	ierr = PetscObjectGetComm((PetscObject)(v[0]),&comm);CHKERRQ(ierr);
	MPI_Comm_rank(comm, &rank);
	int n_vecs = v.size();
	for(int i=0;i<n_vecs;i++){
		Vec2ElMatCol(v[i], A, i);
	}
	MPI_Barrier(comm);
	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "randqr"
PetscErrorCode RandQR(RandQRData* randqrdata, const int q, const double compress_tol, const int total_vecs){
	// q describes the number of iterations of power iteration to perform.
	// compress_tol determines when to stop creating new trees
	// n_vecs, if set, determines the number of random vectors to project, i.e. the size of the random matrix that
	// we are projecting into the range of the operator whose randomized factorization we are trying to determine
	
	PetscErrorCode ierr;
	Mat *A = randqrdata->A;
	Mat *Atrans = randqrdata->Atrans;
	El::DistMatrix<double>* R_tilde = randqrdata->R_tilde;
	El::DistMatrix<double>* Q_tilde = randqrdata->Q_tilde;
	El::DistMatrix<double>* Q = randqrdata->Q;
	PetscRandom r = randqrdata->r;
	MPI_Comm comm = randqrdata->comm;
	El::Grid* grid = randqrdata->grid;
	int m = randqrdata->m;
	int n = randqrdata->n;
	int M = randqrdata->M;
	int N = randqrdata->N;

	assert(total_vecs <= N);

	std::vector<Vec> ortho_vec;

	// if compress_tol > 1, that means it is really the number of vectors we'd like to use.
	// if compress_tol < 1, then it really is the tolerance.
	double ct;
	int max_vecs;
	if(compress_tol >= 1){
		ct = 0;
		max_vecs = compress_tol;
	}
	else{
		ct = compress_tol;
		max_vecs = std::min(N,M);
	}
	std::cout << "ct: " << ct << std::endl;
	std::cout << "max_vecs: " << max_vecs << std::endl;


	int num_vecs = 0;
	int n_times = 0;
	//PetscInt M;
	//PetscInt N;
	//MatGetSize(*A,&M,&N); //obviously Atrans should be N by M then

	// create random vector
	Vec coeffs_vec;
	VecCreateMPI(comm,PETSC_DECIDE,N,&coeffs_vec); // local sizes determined by PETSc, may need fixing in the future

	double norm_val = 1;

	while((norm_val > ct or n_times < 2) and num_vecs < max_vecs){
		num_vecs++;
		// iniitialize the random vector
		for(int i=0;i<N;i++){
			// George says to only set the real parts
			VecSetValue(coeffs_vec,i,((i%2==0) ? randn(0,1,r) : 0),INSERT_VALUES); // will probably need fixing for distributed
			//ierr = VecAssemblyBegin(coeffs_vec); CHKERRQ(ierr);
			//ierr = VecAssemblyEnd(coeffs_vec); CHKERRQ(ierr);
		}

		{
			// create vector
			Vec t2_vec;
			ierr = VecCreateMPI(comm,m,PETSC_DETERMINE,&t2_vec); CHKERRQ(ierr);
			PetscInt v1,v2,v3,v4;
			VecGetSize(t2_vec,&v1);
			VecGetLocalSize(t2_vec,&v2);
			VecGetSize(coeffs_vec,&v3);
			VecGetLocalSize(coeffs_vec,&v4);
			std::cout << "v1: " << v1 << std::endl;
			std::cout << "v2: " << v2 << std::endl;
			std::cout << "v3: " << v3 << std::endl;
			std::cout << "v4: " << v4 << std::endl;

			// y=Ax
			ierr = MatMult(*A, coeffs_vec, t2_vec); CHKERRQ(ierr);
			for(int p=0;p<q;p++){
				// Need to disable the filtering
				std::cout << "tm" << std::endl;
				ierr = MatMult(*Atrans, t2_vec, coeffs_vec); CHKERRQ(ierr);
				std::cout << "m" << std::endl;
				ierr = MatMult(*A, coeffs_vec, t2_vec); CHKERRQ(ierr);
			}

			// Normalize it
			ierr = VecNorm(t2_vec,NORM_2,&norm_val);CHKERRQ(ierr);
			ierr = VecScale(t2_vec,1/norm_val); CHKERRQ(ierr);

			// project it
			if(ortho_vec.size() > 0){
				ortho_project(ortho_vec,t2_vec);
				ierr = VecNorm(t2_vec,NORM_2,&norm_val);CHKERRQ(ierr);

				// renormalize
				ierr = VecScale(t2_vec,1/norm_val); CHKERRQ(ierr);
			}
			std::cout << "norm_val " << norm_val << std::endl;
			ortho_vec.push_back(t2_vec);
		}

		if(norm_val < compress_tol){
			n_times++;
		}
		std::cout << "num_vecs: " << num_vecs << std::endl;
	}
	// Now we have created and orthogonalized all the trees that we're going too

	// stuff them into an elemental matrix
	int l1 = ortho_vec.size();
	int m1;
	double mat_norm;
	VecGetSize(ortho_vec[0],&m1);
	//El::DistMatrix<double> Q;
	Q->Resize(m1,l1);
	Vecs2ElMat(ortho_vec,*Q);

	/*
	// Now we need to create all the original columns of 
	// the input matrix. This is the same as A*e_i for all i.
	std::vector<Vec> u_vec;
	for(int j=0;j<N ;j++){
		for(int i=0;i<N;i++){
			VecSetValue(coeffs_vec,i,((j==i) ? 1 : 0),INSERT_VALUES);
		}
		{
			Vec t2_vec;
			VecCreateMPI(comm,m,PETSC_DETERMINE,&t2_vec);
			ierr = MatMult(*A,coeffs_vec,t2_vec); CHKERRQ(ierr);
			u_vec.push_back(t2_vec);
		}
	}
	*/
	// Let's instead create U_trans directly
	std::cout << "dbg-1" << std::endl;
	std::vector<Vec> ut_vec;
	for(int j=0;j<ortho_vec.size() ;j++){
		{
			Vec t2_vec;
			ierr = VecCreateMPI(comm,PETSC_DECIDE,N,&t2_vec); CHKERRQ(ierr);
			ierr = MatMult(*Atrans,ortho_vec[j],t2_vec); CHKERRQ(ierr);
			ut_vec.push_back(t2_vec);
		}
	}

	std::cout << "dbg0" << std::endl;
	// and put the vectors in a matrix
	/*
	int n1 = u_vec.size();
	El::DistMatrix<double> U(*grid);
	U.Resize(m1,n1);
	Vecs2ElMat(u_vec,U);
	*/
	std::cout << "dbg1" << std::endl;
	int n1 = ut_vec.size();
	El::DistMatrix<double> Ut(*grid);
	std::cout << "m1,n " << m1 << " " << N << std::endl;
	Ut.Resize(N,n1);
	Vecs2ElMat(ut_vec,Ut);

	std::cout << "dbg2" << std::endl;
	/*
	for(int i=0;i<N;i++){
		VecDestroy(&u_vec[i]);
	}
	*/
	for(int i=0;i<ut_vec.size();i++){
		VecDestroy(&ut_vec[i]);
	}

	std::cout << "dbg3" << std::endl;


	// compute Q_tilde
	// The matrix names here are a little weird because eventually
	// when we compute the QR factorization it stores the Q matrix 
	// where the inpute matrix was.
	// First compute A = U*Q
	//El::DistMatrix<double> Q_tilde;

	/*
	El::Zeros(*Q_tilde,n1,l1);
	El::Gemm(El::TRANSPOSE,El::NORMAL,1.0,U,*Q,0.0,*Q_tilde);
	*/

	std::cout << "n1,l1 " << n1 << " " << l1 << std::endl;
	//El::Zeros(*Q_tilde,n1,l1);
	//El::Transpose(Ut,*Q_tilde);
	El::Copy(Ut,*Q_tilde);

	// compute QtildeR = A
	//El::DistMatrix<double> R;
	El::qr::Explicit(*Q_tilde,*R_tilde,El::QRCtrl<double>());



	/*
	// let's test if we get a good output matrix
	int m2 = R_tilde->Width();
	int n2 = Q_tilde->Height();
	std::cout << "m2: " << m2 << std::endl;
	std::cout << "n2: " << n2 << std::endl;
	El::DistMatrix<double> RQ_tilde(*grid);
	El::Zeros(RQ_tilde,m2,n2);
	El::Gemm(El::TRANSPOSE,El::TRANSPOSE,1.0,*R_tilde,*Q_tilde,0.0,RQ_tilde);

	int m3 = Q->Height();
	int n3 = RQ_tilde.Width();
	std::cout << "m3: " << m3 << std::endl;
	std::cout << "n3: " << n3 << std::endl;
	El::DistMatrix<double> RQR(*grid);
	El::Zeros(RQR,m3,n3);
	El::Gemm(El::NORMAL,El::NORMAL,1.0,*Q,RQ_tilde,0.0,RQR);




	std::vector<Vec> u1_vec;
	Vec cv;
	VecCreateMPI(comm,m,PETSC_DETERMINE,&cv);
	for(int j=0;j<M ;j++){
		for(int i=0;i<M;i++){
			VecSetValue(cv,i,((j==i) ? 1 : 0),INSERT_VALUES);
		}
		{
			Vec t2_vec;
			VecCreateMPI(comm,n,PETSC_DETERMINE,&t2_vec);
			ierr = MatMult(*Atrans,cv,t2_vec); CHKERRQ(ierr);
			u1_vec.push_back(t2_vec);
		}
	}

	El::DistMatrix<double> Ut2(*grid);
	std::cout << "N: " << N << std::endl;
	std::cout << "M: " << M << std::endl;
	Ut2.Resize(N,M);
	Vecs2ElMat(u1_vec,Ut2);



	std::vector<Vec> u2_vec;
	for(int j=0;j<N ;j++){
		for(int i=0;i<N;i++){
			VecSetValue(coeffs_vec,i,((j==i) ? 1 : 0),INSERT_VALUES);
		}
		{
			Vec t2_vec;
			VecCreateMPI(comm,m,PETSC_DETERMINE,&t2_vec);
			ierr = MatMult(*A,coeffs_vec,t2_vec); CHKERRQ(ierr);
			u2_vec.push_back(t2_vec);
		}
	}
	int n4 = u2_vec.size();
	std::cout << "n4: " << n4 << std::endl;
	std::cout << "m1: " << m1 << std::endl;
	El::DistMatrix<double> U2(*grid);
	U2.Resize(m1,n4);
	Vecs2ElMat(u2_vec,U2);


	El::DistMatrix<double> U(*grid);
	El::Zeros(U,M,N);
	El::Transpose(Ut2,U);
	El::Display(U);
	std::cout << "====================================================" << std::endl;
	El::Display(RQR);


	El::Axpy(-1,U2,U);
	std::cout << "ndnndfndanfnsd " << El::Norm(U) << std::endl;

	std::cout << "Height: " << U2.Height() << " Width: " << U2.Width() << std::endl;
	El::Axpy(-1,U2,RQR);
	std::cout << "Relative norm of difference: " << El::Norm(RQR)/El::Norm(U2) << std::endl;
	std::cout << "Absolute norm of difference: " << El::Norm(RQR) << std::endl;


	for(int i=0;i<u1_vec.size();i++){
		VecDestroy(&u1_vec[i]);
	}
	for(int i=0;i<u2_vec.size();i++){
		VecDestroy(&u2_vec[i]);
	}
	*/

	for(int i=0;i<num_vecs;i++){
		VecDestroy(&ortho_vec[i]);
	}
	return ierr;
}

struct IncidentData{
	std::vector<double> *coeffs;
	pvfmm::BoundaryType bndry;
	const pvfmm::Kernel<double>* kernel;
	// this function pointer needs to depened in some way
	//  on the vector being used in the incident_mult function
	void (*fn)(const  double* coord, int n, double* out);
	MPI_Comm comm;
	InvMedTree<FMM_Mat_t>* mask;
};


#undef __FUNCT__
#define __FUNCT__ "incident_mult"
PetscErrorCode incident_mult(Mat M, Vec U, Vec Y){
	// u  is the vector containing the random coefficients for each of the point sources

	PetscErrorCode ierr;
	// Get context ... or maybe I won't need any context
	IncidentData *incident_data = NULL;
	MatShellGetContext(M, &incident_data);
	MPI_Comm comm = incident_data->comm;
	InvMedTree<FMM_Mat_t>* mask = incident_data->mask;
	PetscInt vec_length;
	PetscScalar val;

	ierr = VecGetSize(U,&vec_length); CHKERRQ(ierr);
	// PETSc vector to std::vector
	{
		incident_data->coeffs->clear(); // this variable unfortunately needs to be global because I can not bind data to a function
		for(int i=0;i<vec_length;i++){
			VecGetValues(U,1,&i,&val);
			incident_data->coeffs->push_back((double)val);
			std::cout << "coeffs: " << (*(incident_data->coeffs))[i]  << std::endl;
		}
	}
	// create the tree with the current values of the random vector
	InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
	t->bndry = incident_data->bndry;
	t->kernel = incident_data->kernel;
	t->fn = incident_data->fn;
	t->f_max = 4;
	t->CreateTree(false);

	// convert the tree into a vector. This vector represents the function
	// that we passed into the tree constructor (which contains the current 
	// random coefficients).
	t->Multiply(mask,1);
	tree2vec(t,Y);

	delete t;

	return 0;
}

struct IncidentTransData{
	MPI_Comm comm;
	InvMedTree<FMM_Mat_t>* temp_c;
	std::vector<double> src_coord;
	pvfmm::PtFMM_Tree* pt_tree;
	InvMedTree<FMM_Mat_t>* mask;
};

#undef __FUNCT__
#define __FUNCT__ "incident_transpose_mult"
PetscErrorCode incident_transpose_mult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	// Get context ...
	IncidentTransData *incident_trans_data = NULL;
	MatShellGetContext(M, &incident_trans_data);
	MPI_Comm comm = incident_trans_data->comm;
	InvMedTree<FMM_Mat_t>* temp_c = incident_trans_data->temp_c;
	InvMedTree<FMM_Mat_t>* mask = incident_trans_data->mask;
	std::vector<double> src_coord = incident_trans_data->src_coord;
	//pvfmm::PtFMM_Tree* pt_tree = incident_trans_data->pt_tree;

	// get the input data
	vec2tree(U,temp_c);
	temp_c->Multiply(mask,1);

	// integrate
	temp_c->ClearFMMData();
	temp_c->RunFMM();
	temp_c->Copy_FMMOutput();

	// read at srcs
	std::vector<double> src_values = temp_c->ReadVals(src_coord);

	// move the data into a petsc vector
	PetscInt low, high, l_size;
	double *arr;
	ierr = VecGetOwnershipRange(Y,&low,&high); CHKERRQ(ierr);
	ierr = VecGetLocalSize(Y,&l_size); CHKERRQ(ierr);
	ierr = VecGetArray(Y,&arr); CHKERRQ(ierr);
	for(int i=low;i<high;i++){
		arr[i-low] = src_values[i];
	}
	ierr = VecRestoreArray(Y,&arr); CHKERRQ(ierr);
	
	//pt_tree->ClearFMMData();
	//std::vector<double> trg_value;
	//pvfmm::PtFMM_Evaluate(pt_tree, trg_value, 0, &src_values);


	//for(int i=0;i<trg_value.size();i++){
	//	ierr = VecSetValue(Y,i,trg_value[i],INSERT_VALUES);CHKERRQ(ierr);
	//}
	//ierr = VecAssemblyBegin(Y); CHKERRQ(ierr);
	//ierr = VecAssemblyEnd(Y); CHKERRQ(ierr);

	// Insert the values back in
	//temp->Trg2Tree(trg_value);
	//tree2vec(temp,Y);

	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "G_mult"
PetscErrorCode G_mult(Mat M, Vec U, Vec Y){
	PetscErrorCode ierr;
	InvMedData* g_data = NULL;

	// This function simply computes the convolution of G with an input U
	// and then gets only the output at the detector locations
	ierr = MatShellGetContext(M, &g_data);CHKERRQ(ierr);
	InvMedTree<FMM_Mat_t>* temp = g_data->temp;
	InvMedTree<FMM_Mat_t>* mask = g_data->phi_0;
	std::vector<double> detector_coord = g_data->src_coord;
	std::vector<double> coeff_scaling = {1};
	bool filter = g_data->filter;

	vec2tree(U,temp);
	if(filter){
		std::cout << "filtering" << std::endl;
		temp->FilterChebTree(coeff_scaling);
	}
	temp->Multiply(mask,1);
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();
	std::vector<double> detector_values = temp->ReadVals(detector_coord);
	PetscInt low, high, l_size;
	double *arr;
	ierr = VecGetOwnershipRange(Y,&low,&high); CHKERRQ(ierr);
	ierr = VecGetLocalSize(Y,&l_size); CHKERRQ(ierr);
	ierr = VecGetArray(Y,&arr); CHKERRQ(ierr);
	for(int i=low;i<high;i++){
		arr[i-low] = detector_values[i];
	}
	ierr = VecRestoreArray(Y,&arr); CHKERRQ(ierr);
	//for(int i=0;i<detector_values.size();i++){
	//  ierr = VecSetValue(Y,i,detector_values[i],INSERT_VALUES);CHKERRQ(ierr);
	//}
	//ierr = VecAssemblyBegin(Y); CHKERRQ(ierr);
	//ierr = VecAssemblyEnd(Y); CHKERRQ(ierr);

	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "Gt_mult"
PetscErrorCode Gt_mult(Mat M, Vec U, Vec Y){
	PetscErrorCode ierr;
	InvMedData* g_data = NULL;

	ierr = MatShellGetContext(M, &g_data);CHKERRQ(ierr);
	std::vector<double> detector_coord = g_data->src_coord;
	pvfmm::PtFMM_Tree* Gt_tree = g_data->pt_tree;
	InvMedTree<FMM_Mat_t>* temp = g_data->temp;
	InvMedTree<FMM_Mat_t>* mask = g_data->phi_0;

	PetscInt l_size;
	ierr = VecGetLocalSize(U,&l_size);CHKERRQ(ierr);
	
	std::vector<double> detector_values(l_size);
	const double* dv = detector_values.data();

	ierr = VecGetArrayRead(U,&dv);CHKERRQ(ierr);
	for(int i=0;i<l_size;i++){
		detector_values[i] = dv[i];
	}
	

	Gt_tree->ClearFMMData();
	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(Gt_tree, trg_value, 0, &detector_values);

	ierr = VecRestoreArrayRead(U,&dv);CHKERRQ(ierr);

	// Insert the values back in
	temp->Trg2Tree(trg_value);
	temp->Multiply(mask,1);
	tree2vec(temp, Y);

	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "LR_mult"
PetscErrorCode LR_mult(Mat M, Vec U, Vec Y){

	PetscErrorCode ierr;
	MPI_Comm comm;

	Mat* A_mat = NULL;
	ierr = MatShellGetContext(M, &A_mat);CHKERRQ(ierr);
	PetscObjectGetComm((PetscObject)(*A_mat),&comm);
	PetscInt m,n;
	MatGetSize(*A_mat,&m,&n);
	Vec v;
	VecCreateMPI(comm, PETSC_DECIDE,n,&v);

	ierr = MatMultTranspose(*A_mat, U,v); CHKERRQ(ierr);
	ierr = MatMult(*A_mat,v,Y); CHKERRQ(ierr);

	VecDestroy(&v);

	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "G_func"
int G_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, G_data *g_data){

	// This function simply computes the convolution of G with an input U
	// and then gets only the output at the detector locations
	InvMedTree<FMM_Mat_t>* temp = g_data->temp;
	InvMedTree<FMM_Mat_t>* mask = g_data->mask;
	std::vector<double> detector_coord = g_data->src_coord;
	bool filter = g_data->filter;
	MPI_Comm comm = g_data->comm;
	int rank;
	MPI_Comm_rank(comm,&rank);
	if(!rank) std::cout << "G function " << std::endl; 

	//std::cout << "Norm in" << El::OneNorm(x) << std::endl;
	
	elemental2tree(x,temp);
	//MPI_Barrier(comm);
	temp->Write2File("../results/g",6);
	temp->Multiply(mask,1);
	//MPI_Barrier(comm);
	temp->ClearFMMData();
	//temp->Write2File("../results/x_after_clear",6);
	MPI_Barrier(comm);
	temp->RunFMM();
	temp->Copy_FMMOutput();

	temp->Write2File("../results/gf_y",4);
	std::vector<double> detector_values = temp->ReadVals(detector_coord);

	vec2elemental(detector_values,y);
	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Gt_func"
int Gt_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, G_data *g_data){

	std::vector<double> detector_coord = g_data->src_coord;
	pvfmm::PtFMM_Tree* Gt_tree = g_data->pt_tree;
	InvMedTree<FMM_Mat_t>* temp = g_data->temp;
	InvMedTree<FMM_Mat_t>* mask = g_data->mask;
	MPI_Comm comm = g_data->comm;
	int rank;
	MPI_Comm_rank(comm,&rank);

	if(!rank) std::cout << "G* function " << std::endl; 

	//int n = y.Height();
	int n = (temp->ChebPoints()).size()/3;
	int nd = detector_coord.size()*2/3;
	std::vector<double> detector_values(nd);

	elemental2vec(y,detector_values);

	Gt_tree->ClearFMMData();
	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(Gt_tree, trg_value, n, &detector_values);

	// Insert the values back in
	temp->Trg2Tree(trg_value);
	temp->Write2File("../results/gt",6);
	temp->Multiply(mask,1);
	temp->Write2File("../results/gtm",6);
	tree2elemental(temp, x);

	return 0;
}


#undef __FUNCT__
#define __FUNCT__ "U_func"
int U_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, U_data *u_data){
	// u  is the vector containing the random coefficients for each of the point sources

	if((x.ColAlign() != y.ColAlign()) or (x.RowAlign() != y.RowAlign())){
		El::LogicError("x and y must have the same distribution");
	}
	//El::Grid& g = x.Grid();
	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> x_star = x;
	
	InvMedTree<FMM_Mat_t>* mask = u_data->mask;
	MPI_Comm comm = u_data->comm;

	int rank;
	MPI_Comm_rank(comm, &rank);

	elstar2vec(x_star,*(u_data->coeffs));
	/*
	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << (*(u_data.coeffs))[0] << std::endl;
		std::cout << (*(u_data.coeffs))[1] << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 1){
		std::cout << (*(u_data.coeffs))[0] << std::endl;
		std::cout << (*(u_data.coeffs))[1] << std::endl;
	}
	MPI_Barrier(comm);
	*/

	// create the tree with the current values of the random vector
	InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
	t->bndry = u_data->bndry;
	t->kernel = u_data->kernel;
	t->fn = u_data->fn;
	t->f_max = 4;
	t->CreateTree(false);

	// convert the tree into a vector. This vector represents the function
	// that we passed into the tree constructor (which contains the current 
	// random coefficients).
	t->Multiply(mask,1);
	t->Write2File("../results/ufunc",0);
	tree2elemental(t,y);

	MPI_Barrier(comm);
	delete t;

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Ut_func"
int Ut_func(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, U_data *u_data){


	if((x.ColAlign() != y.ColAlign()) or (x.RowAlign() != y.RowAlign())){
		El::LogicError("x and y must have the same distribution");
	}

	InvMedTree<FMM_Mat_t>* temp_c = u_data->temp_c;
	InvMedTree<FMM_Mat_t>* mask = u_data->mask;
	std::vector<double> src_coord = u_data->src_coord;
	int rank;
	MPI_Comm_rank(u_data->comm,&rank);
	if(!rank) std::cout << "U* function" << std::endl;

	// get the input data
	elemental2tree(y,temp_c);
	temp_c->Multiply(mask,1);

	// integrate
	temp_c->ClearFMMData();
	temp_c->RunFMM();
	temp_c->Copy_FMMOutput();

	// read at srcs
	std::vector<double> src_values = temp_c->ReadVals(src_coord);
	//if(!rank){
	//	std::cout << src_values[0] << std::endl;
	//	std::cout << src_values[1] << std::endl;
	//}

	vec2elemental(src_values,x);	

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "U_func2"
int U_func2(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, U_data *u_data){
	// u  is the vector containing the random coefficients for each of the point sources

	InvMedTree<FMM_Mat_t>* mask = u_data->mask;
	InvMedTree<FMM_Mat_t>* temp = u_data->temp;
	MPI_Comm comm               = u_data->comm;
	pvfmm::PtFMM_Tree* pt_tree  = u_data->pt_tree;
	int trg_coord_size          = u_data->trg_coord_size;

	int rank;
	MPI_Comm_rank(comm, &rank);

	if(!rank) std::cout << "U function" << std::endl;

	int n_local_pt_srcs = u_data->n_local_pt_srcs;
	std::vector<double> coeffs(n_local_pt_srcs*2);
	elemental2vec(x,coeffs);

	//std::vector<double> trg_coord = temp->ChebPoints();


	std::vector<double> trg_value;
	pt_tree->ClearFMMData();
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value, trg_coord_size,&coeffs);

	temp->Trg2Tree(trg_value);

	// convert the tree into a vector. This vector represents the function
	// that we passed into the tree constructor (which contains the current 
	// random coefficients).
	//temp->Write2File("../results/ufunc_b",8);
	temp->Multiply(mask,1);
	//temp->Write2File("../results/ufunc",8);
	tree2elemental(temp,y);

	MPI_Barrier(comm);

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "rsvd_test_func"
int rsvd_test_func(El::DistMatrix<El::Complex<double>> &x, El::DistMatrix<El::Complex<double>> &y, El::DistMatrix<El::Complex<double>> *A){

	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);
	int N = A->Height();
	El::Zeros(y,N,1);

	El::Gemm(El::NORMAL,El::NORMAL,alpha,*A,x,beta,y);

}

#undef __FUNCT__
#define __FUNCT__ "rsvd_test_t_func"
int rsvd_test_t_func(El::DistMatrix<El::Complex<double>> &x, El::DistMatrix<El::Complex<double>> &y, El::DistMatrix<El::Complex<double>> *At){

	
	El::Complex<double> alpha;
	El::SetRealPart(alpha,1.0);
	El::SetImagPart(alpha,0.0);

	El::Complex<double> beta;
	El::SetRealPart(beta,0.0);
	El::SetImagPart(beta,0.0);

	int N = At->Height();
	El::Zeros(y,N,1);

	El::Gemm(El::NORMAL,El::NORMAL,alpha,*At,x,beta,y);

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "B_func"
int B_func(
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *S_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *V_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *W,
		InvMedTree<FMM_Mat_t>* temp,
		InvMedTree<FMM_Mat_t>* temp_c
		)
{
	int R_s = W->Width();
	int R_d = V_G->Width();
	int N_disc = W->Height();

	const El::Grid& g = x.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;
	int rank;
	MPI_Comm_rank(comm,&rank);
	if(!rank) std::cout << "B function" << std::endl;

	elemental2tree(x,temp_c);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Wx(g);
	El::Zeros(Wx,N_disc,R_s);

	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> W_i = El::View(*W, 0, i, N_disc, 1);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Wx_i = El::View(Wx, 0, i, N_disc, 1);
		elemental2tree(W_i,temp);
		temp->Multiply(temp_c,1);
		tree2elemental(temp,Wx_i);
	}

	El::Complex<double> alpha = El::Complex<double>(1.0);
	El::Complex<double> beta  = El::Complex<double>(0.0);


	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VtWx(g);
	El::Zeros(VtWx,R_d,R_s);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,*V_G,Wx,beta,VtWx);

	El::DiagonalScale(El::LEFT,El::NORMAL,*S_G,VtWx);

	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> SVtWx2(g);
	SVtWx2 = VtWx;
	SVtWx2.Resize(R_s*R_d,1);
	VtWx = SVtWx2;
	y = VtWx;

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Bt_func"
int Bt_func(
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *S_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *V_G,
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> *W,
		InvMedTree<FMM_Mat_t>* temp,
		InvMedTree<FMM_Mat_t>* temp_c,
		InvMedTree<FMM_Mat_t>* temp2
		)
{

	const El::Grid& g = x.Grid();
	El::mpi::Comm elcomm = g.Comm();
	MPI_Comm comm = elcomm.comm;

	El::Complex<double> alpha = El::Complex<double>(1.0);
	El::Complex<double> beta  = El::Complex<double>(0.0);

	int rank;
	MPI_Comm_rank(comm,&rank);
	if(!rank) std::cout << "B* function" << std::endl;

	int R_s = W->Width();
	int R_d = V_G->Width();
	int N_disc = W->Height();

	//El::Display(x,"Bt vec");
	El::DistMatrix<El::Complex<double>,El::STAR,El::STAR> x2(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x_copy(g);
	El::Copy(x,x_copy);
	x2 = x_copy;
	x2.Resize(R_d,R_s);
	x_copy = x2;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Sx(g);
	El::Zeros(Sx,R_d,R_s);
	El::DiagonalScale(El::LEFT,El::ADJOINT,*S_G,x_copy);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VSx(g);
	El::Zeros(VSx,N_disc,R_s);
	El::Gemm(El::NORMAL,El::NORMAL,alpha,*V_G,x_copy,beta,VSx);


	/*
	InvMedTree<FMM_Mat_t>* t = new InvMedTree<FMM_Mat_t>(comm);
	t->bndry = temp->bndry;
	t->kernel = temp->kernel;
	t->fn = zero_fn;
	t->f_max = 4;
	t->CreateTree(false);
	*/
	temp2->Zero();


	for(int i=0;i<R_s;i++){
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VSx_i = El::View(VSx, 0, i, N_disc, 1);
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> W_i = El::View(*W, 0, i, N_disc, 1);
		elemental2tree(W_i,temp_c);
		elemental2tree(VSx_i,temp);
		temp->ConjMultiply(temp_c,1);
		temp2->Add(temp,1);
	}
	tree2elemental(temp2,y);

	//delete t;

	return 0;
}
