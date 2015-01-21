#include <iostream>
#include <profile.hpp>
//#include <kernel.hpp>
#include <pvfmm.hpp>
#include <pvfmm_common.hpp>
#include <mpi.h>

void one_fn(const double* coord, int n, double* out){ 
	//int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
     const double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=1;
      out[i*dof+1]=0;
    }
  }
}


void pt_sources_fn(const double* coord, int n, double* out){ 
  int dof=2;
//	int COORD_DIM = 3;
  double L=500;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
			double temp;
      double r_20=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.9)*(c[1]-0.9)+(c[2]-0.5)*(c[2]-0.5);
      double r_21=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.9)*(c[2]-0.9);
      double r_22=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.1)*(c[2]-0.1);
      double r_23=(c[0]-0.1)*(c[0]-0.1)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      double r_24=(c[0]-0.9)*(c[0]-0.9)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      double r_25=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.1)*(c[1]-0.1)+(c[2]-0.5)*(c[2]-0.5);
      //double rho_val=1.0;
      //rho(c, 1, &rho_val);
			temp = exp(-L*r_20)+exp(-L*r_21) + exp(-L*r_22)+ exp(-L*r_23)+ exp(-L*r_24)+ exp(-L*r_25);
      if(dof>1) out[i*dof+0]= sqrt(L/M_PI)*temp;
      if(dof>1) out[i*dof+1]=0;
    }
  }
}

void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k);
void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

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

const pvfmm::Kernel<double> helm_kernel=pvfmm::BuildKernel<double, helm_kernel_fn>("helm_kernel", 3, std::pair<int,int>(2,2));


int main(int argc, char* argv[]){

	MPI_Init(&argc,&argv);
  const pvfmm::Kernel<double>* kernel=&helm_kernel;
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
	typename FMMNode_t::NodeData tree_data;
	typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
	//Various parameters.
	int myrank, np;

  MPI_Comm comm=MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
	// Set original source coordinates on a regular grid 
	// at minepth with one point per octree node.
	// This gets refined when FMM_Init is called with adaptivity on.

	int cheb_deg = 7;
	int mult_order = 8;
	double tol = 1e-6;
	int mindepth = 3;
	int maxdepth = 5;
	bool adap = true;
	int dim = 3;
	int data_dof = 2;


	tree_data.max_pts=1; // Points per octant.
	tree_data.dim=       dim;
	tree_data.max_depth= maxdepth;
	tree_data.cheb_deg=  cheb_deg;
	tree_data.data_dof=  data_dof;

	std::vector<double> pt_coord;
	{ 
		size_t NN=ceil(pow((double)np,1.0/3.0));
		NN=std::max<size_t>(NN,pow(2.0,mindepth));
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

	pvfmm::FMM_Tree<FMM_Mat_t> *tree= new pvfmm::FMM_Tree<FMM_Mat_t>(comm);	

	tree_data.input_fn=pt_sources_fn;
	tree_data.tol=tol;

	//Create Tree and initialize with input data.
	tree->Initialize(&tree_data);
	tree->InitFMM_Tree(adap,bndry);
	std::cout << (tree->GetNodeList()).size() << std::endl;

  FMM_Mat_t *fmm_mat=new FMM_Mat_t;
	fmm_mat->Initialize(mult_order,cheb_deg,comm,kernel);

  tree->Write2File("results/before_fmm",0);
	tree->SetupFMM(fmm_mat);
	tree->RunFMM();
	tree->Copy_FMMOutput();
  tree->Write2File("results/after_fmm",0);

	delete tree;

	MPI_Finalize();


	return 0;

}

