#include <vector>
#include <iostream>
#include <mpi.h>
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
				if(!(size_diff[j] >= 0)){
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
				if(size_diff[j] < 0){
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

	std::vector<int> input_sizes = {10, 6, 4, 10};
	std::vector<int> output_sizes = {6, 8, 8, 8};
	std::vector<int> sendcnts;
	std::vector<int> sdispls;
	std::vector<int> recvcnts;
	std::vector<int> rdispls;

	comp_alltoall_sizes(input_sizes, output_sizes, sendcnts, sdispls, recvcnts, rdispls, comm);

	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << "=======================" << std::endl;
		print(sendcnts);
		std::cout << "======" << std::endl;
		print(recvcnts);
		std::cout << "======" << std::endl;
		print(sdispls);
		std::cout << "======" << std::endl;
		print(rdispls);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 1){
		print(sendcnts);
		std::cout << "======" << std::endl;
		print(recvcnts);
		std::cout << "======" << std::endl;
		print(sdispls);
		std::cout << "======" << std::endl;
		print(rdispls);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 2){
		print(sendcnts);
		std::cout << "======" << std::endl;
		print(recvcnts);
		std::cout << "======" << std::endl;
		print(sdispls);
		std::cout << "======" << std::endl;
		print(rdispls);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 3){
		print(sendcnts);
		std::cout << "======" << std::endl;
		print(recvcnts);
		std::cout << "======" << std::endl;
		print(sdispls);
		std::cout << "======" << std::endl;
		print(rdispls);
		std::cout << "=======================" << std::endl;
	}

	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << "====================================================================" << std::endl;
	}
	MPI_Barrier(comm);

	std::srand(std::time(0));

	std::vector<int> indata(input_sizes[rank]);
	std::vector<int> indata2(input_sizes[rank]);
	std::vector<int> outdata(output_sizes[rank]);

	for(int i=0;i<input_sizes[rank];i++){
		indata[i] = std::rand();
	}

	MPI_Barrier(comm);
	if(rank == 0){
		print(indata);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 1){
		print(indata);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 2){
		print(indata);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 3){
		print(indata);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);


	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << "====================================================================" << std::endl;
	}
	MPI_Barrier(comm);

	MPI_Alltoallv(&indata[0],&sendcnts[0],
			&sdispls[0], MPI_INT, &outdata[0],
			&recvcnts[0],&rdispls[0] , MPI_INT,
			comm);


	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << "====================================================================" << std::endl;
	}
	MPI_Barrier(comm);

	comp_alltoall_sizes(output_sizes, input_sizes, sendcnts, sdispls, recvcnts, rdispls, comm);

	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << "====================================================================" << std::endl;
	}
	MPI_Barrier(comm);

	MPI_Alltoallv(&outdata[0],&sendcnts[0],
			&sdispls[0], MPI_INT, &indata2[0],
			&recvcnts[0],&rdispls[0] , MPI_INT,
			comm);


	MPI_Barrier(comm);
	if(rank == 0){
		std::cout << "====================================================================" << std::endl;
	}
	MPI_Barrier(comm);

	MPI_Barrier(comm);
	if(rank == 0){
		print(indata);
		std::cout << "=======================" << std::endl;
	}

	MPI_Barrier(comm);
	for(int i=0;i<input_sizes[rank];i++){
		indata2[i] -= indata[i];
	}

	MPI_Barrier(comm);
	if(rank == 0){
		print(indata2);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 1){
		print(indata2);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 2){
		print(indata2);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);
	if(rank == 3){
		print(indata2);
		std::cout << "=======================" << std::endl;
	}
	MPI_Barrier(comm);




	MPI_Finalize();

	return 0;
}
