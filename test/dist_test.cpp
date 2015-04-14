#include <vector>
#include <iostream>
#include <mpi.h>
#include "omp.h"
#include <random>
#include <cstdlib>
#include <ctime>

#include "par_scan/gen_scan.hpp"

int comp_alltoall_sizes(const std::vector<int> &input_sizes, const std::vector<int> &output_sizes, std::vector<int> &sendcnts, std::vector<int> &sdispls, std::vector<int> &recvcnts, std::vector<int> &rdispls, MPI_Comm comm);

template<typename T>
void print(std::vector<T> output);

template <typename T>
void op(T& v1, const T& v2);

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

	return 0;
}

template<typename T>
void print(std::vector<T> output){
	for(auto i : output){
		std::cout << i << std::endl;
	}
	return;
}

int main(int argc, char *argv[]){
 	MPI_Init( &argc, &argv );

	MPI_Comm comm = MPI_COMM_WORLD;

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	//std::cout << "Size: " << size << std::endl;

	//std::vector<int> input_sizes = {5, 10, 24, 15};
	//std::vector<int> output_sizes = {1, 12, 33, 8};
	//std::vector<int> input_sizes = {5, 10, 10, 10};
	//std::vector<int> output_sizes = {10, 15, 5, 5};

	
	std::vector<int> input_sizes(size);
	std::vector<int> output_sizes(size);
	for(int i=0;i<size;i++){
		input_sizes[i] = 10 + i;
		output_sizes[size - i - 1] = 10 + i;
	}
	
	//if(!rank){
	//	print(input_sizes);
	//	print(output_sizes);
	//}
	std::vector<int> sendcnts;
	std::vector<int> sdispls;
	std::vector<int> recvcnts;
	std::vector<int> rdispls;

	std::vector<int> indata(input_sizes[rank]);
	std::vector<int> indata2(input_sizes[rank]);
	std::vector<int> outdata(output_sizes[rank]);

	std::srand(std::time(0));
	for(int i=0;i<input_sizes[rank];i++){
		indata[i] = std::rand();
	}

	double t_start = omp_get_wtime();
	comp_alltoall_sizes(input_sizes, output_sizes, sendcnts, sdispls, recvcnts, rdispls, comm);
	double t_comp1 = omp_get_wtime() - t_start;


	t_start = omp_get_wtime();
	// send data to new sizes
	MPI_Alltoallv(&indata[0],&sendcnts[0],
			&sdispls[0], MPI_INT, &outdata[0],
			&recvcnts[0],&rdispls[0] , MPI_INT,
			comm);
	double t_send1 = omp_get_wtime() - t_start;

	// send back!
	t_start = omp_get_wtime();
	comp_alltoall_sizes(output_sizes, input_sizes, sendcnts, sdispls, recvcnts, rdispls, comm);
	double t_comp2 = omp_get_wtime() - t_start;

	t_start = omp_get_wtime();
	MPI_Alltoallv(&outdata[0],&sendcnts[0],
			&sdispls[0], MPI_INT, &indata2[0],
			&recvcnts[0],&rdispls[0] , MPI_INT,
			comm);
	double t_send2 = omp_get_wtime() - t_start;

	// compute differences
	for(int i=0;i<input_sizes[rank];i++){
		indata2[i] -= indata[i];
	}

	// compute total error
	double l_sum=0;
	for(int i=0;i<input_sizes[rank];i++){
		l_sum += indata2[i];
	}
	double global_sum=0;
	double t_comp1_max;
	double t_comp2_max;
	double t_send1_max;
	double t_send2_max;
	MPI_Reduce(&l_sum,&global_sum,1,MPI_DOUBLE,MPI_SUM,0,comm);

	MPI_Reduce(&t_comp1,&t_comp1_max,1,MPI_DOUBLE,MPI_MAX,0,comm);
	MPI_Reduce(&t_comp2,&t_comp2_max,1,MPI_DOUBLE,MPI_MAX,0,comm);
	MPI_Reduce(&t_send1,&t_send1_max,1,MPI_DOUBLE,MPI_MAX,0,comm);
	MPI_Reduce(&t_send2,&t_send2_max,1,MPI_DOUBLE,MPI_MAX,0,comm);
	if(!rank){
		std::cout << "t_comp1: " << t_comp1_max << std::endl;
		std::cout << "t_comp2: " << t_comp2_max << std::endl;
		std::cout << "t_send1: " << t_send1_max << std::endl;
		std::cout << "t_send2: " << t_send2_max << std::endl;
		std::cout << "Total error: " << global_sum << std::endl;
	}

	MPI_Finalize();

	return 0;
}
