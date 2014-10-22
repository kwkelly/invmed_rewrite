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

#undef __FUNCT__
#define __FUNCT__ "mult"
int mult(Mat M, Vec U, Vec Y);

#undef __FUNCT__
#define __FUNCT__ "fullmult"
int fullmult(Mat M, Vec U, Vec Y);

#undef __FUNCT__
#define __FUNCT__ "tree2vec"
template <class FMM_Mat_t>
int tree2vec(InvMedTree<FMM_Mat_t> *tree, Vec& Y);

#undef __FUNCT__
#define __FUNCT__ "vec2tree"
template <class FMM_Mat_t>
int vec2tree(Vec& Y, InvMedTree<FMM_Mat_t> *tree);

void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k);
void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);
void nonsingular_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

std::vector<double> randsph(int n_points, double rad);
std::vector<double> randunif(int n_points);


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

	std::srand(std::time(NULL));
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

	std::srand(std::time(NULL));
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

std::vector<double> test_pts(){
	std::vector<double> pts;
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
	pts.push_back( 0.3000);
	pts.push_back( 0.5000);
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
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.7000);
	pts.push_back( 0.5000);
	pts.push_back( 0.7000);
	pts.push_back( 0.5000);
	pts.push_back( 0.7000);
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	pts.push_back( 0.7000);
	pts.push_back( 0.5000);
	pts.push_back( 0.5000);
	return pts;
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
	int cheb_deg = InvMedTree<FMM_Mat_t>::cheb_deg;
	//int omp_p=omp_get_max_threads();

	
	vec2tree(U,temp);

	temp->Multiply(phi_0,1);

	// Run FMM ( Compute: G[ \eta * u ] )
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();

	// Sample at the points in src_coord, then apply the transpose
	// operator.
	std::vector<double> src_values = temp->ReadVals(src_coord);
//	std::cout << "vals: " << src_values[0] << ", " << src_values[1] << std::endl;
//	std::cout << "force them to be correct" << std::endl;

//	src_values[0] = -1;
//	src_values[1] = 0;

//	std::cout << "src_values[i]" << std::endl;
//	for(int i=0;i<src_values.size();i++){
//		std::cout << src_values[i] << std::endl;
//	}
	//InvMedTree<FMM_Mat_t>::SetSrcValues(src_coord,src_values,pt_tree);

	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value, 0, &src_values);

	// Insert the values back in
	temp->Trg2Tree(trg_value);
	
	// Ptwise multiply by the conjugate of phi_0
	temp->ConjMultiply(phi_0,1);

	temp->Write2File("results/aftermult",0);


	tree2vec(temp,Y);

	// Regularize
	ierr = VecAXPY(Y,alpha,U);CHKERRQ(ierr);

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
