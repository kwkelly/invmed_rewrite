#include "El.hpp"
#include <vector>
#include <omp.h>
#include "invmed_tree.hpp"
#include "par_scan/gen_scan.hpp"
#include "convert_elemental.hpp"

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;

/*
 * Convert a pvfmm tree to en elemental vector whose entries are the Chebyshev coefficients
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */

template <class FMM_Mat_t, typename T>
int tree2elemental(InvMedTree<FMM_Mat_t> *tree, El::DistMatrix<T,El::VC,El::STAR> &Y){

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


/*
 * Convert an Elemental vector to a pvfmm tree. The opposite of the previous function
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */

template <class FMM_Mat_t, typename T>
int elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, InvMedTree<FMM_Mat_t> *tree){
	
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


/*
 * Convert a std::vector to an elemental vector of the given distribution. As before, thise well reorder the data
 */
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

/*
 * Convert an elemental vector to a std::vector of the given distribution. As before, thise well reorder the data
 */
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

/*
 * Convert an elemental STAR, STAR distributed vector toa  std::vector
 */
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

/*
 * Convert a std::vector to an elemental STAR, STAR distributed vector
 */
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

/*
 * the sum op for a scan operation used in the below function.
 */
template <typename T>
void op(T& v1, const T& v2){
	v1+=v2;
}

/*
 * Due to the different sizes and distributions of the data between pvfmm and elemental, we need to calculate how much data and which data is going to be sent to each plave
 */
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

///////////////////////////////////////
// Now let's instantiate them templates
///////////////////////////////////////

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

template
int tree2elemental(InvMedTree<FMM_Mat_t> *tree, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y);

template
int elemental2tree(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, InvMedTree<FMM_Mat_t> *tree);


// the functions that I borrowed from sameer that are currently nonfunctional
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
