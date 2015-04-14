#include <iostream>
#include <omp.h> 
#include <stdlib.h>
#include "gen_scan.hpp"
#include <valarray>

// define the operator 
//overwrite the first value with the new one
template <typename T>
void op(T& v1, const T& v2){
	v1+=v2;
}

// if we want to do everything with vectors, we can overload that operator.
template <typename T>
void operator+=(std::vector<T> &v1, const std::vector<T> &v2){
	size_t n = v1.size();
	//#pragma omp parallel for schedule(static)
	for(int i=0; i<n;i++) v1[i] += v2[i];
}


// helper functions for the initialization of a vector valued vector
template<typename T>
std::vector<std::valarray<T> > init_vec(int n, int m){
	std::vector<std::valarray<T> > x(n);
	for(int i=0;i<n;i++){
		x[i] = std::valarray<T> (i%2,m);
	}
	return x;
}

template<typename T>
std::vector<T> init(size_t n){
	std::vector<T> x(n);
	for(int i=0;i<n;i++) x[i] = i%2;
	return x;
}

template<typename T>
void print(std::vector<T> output){
	for(auto i : output){
		std::cout << i << std::endl;
	}
	return;
}

int main( int argc, char **argv){	
	omp_set_num_threads(12);
	
	// problem setup
	size_t n_threads = 4;
	int n=10;
	if(argc>1) n=atoi(argv[1]);
	if(argc>2) n_threads = atoi(argv[2]);
	omp_set_num_threads(n_threads);

	//initialize vector
	{
		int m = 4;
		auto x = init_vec<int>(n,m);

		/*
		 * sequential
		 */
		// scan
		double t1 = omp_get_wtime();
		seq_scan(x);
		double t2 = omp_get_wtime();
		std::cout << "seq time: " << t2 - t1 << std::endl;

		/*
		 * parallel
		 */
		// reset
		x = init_vec<int>(n,m);
		t1 = omp_get_wtime();
		it_scan(x);
		t2 = omp_get_wtime();
		std::cout << "par time: " << t2 - t1 << std::endl;

		//parallel 2 - just a slightly different implementation
		// reset
		x = init_vec<int>(n,m);
		t1 = omp_get_wtime();
		it_scan2(x);
		t2 = omp_get_wtime();
		std::cout << "par2 time: " << t2 - t1 << std::endl;

		// simpler parallel
		// reset
		x = init_vec<int>(n,m);
		t1 = omp_get_wtime();
		simple_scan(x);
		t2 = omp_get_wtime();
		std::cout << "simple time: " << t2 - t1 << std::endl;

		// simpler parallel done differently
		// reset
		x = init_vec<int>(n,m);
		t1 = omp_get_wtime();
		simple_scan2(x);
		t2 = omp_get_wtime();
		std::cout << "simple2 time: " << t2 - t1 << std::endl;
	}

	//initialize vector
	{
		auto x = init<int>(n);

		/*
		 * sequential
		 */
		// scan
		double t1 = omp_get_wtime();
		seq_scan(x);
		double t2 = omp_get_wtime();
		std::cout << "seq time: " << t2 - t1 << std::endl;

		/*
		 * parallel
		 */
		// reset
		x = init<int>(n);
		t1 = omp_get_wtime();
		it_scan(x);
		t2 = omp_get_wtime();
		std::cout << "par time: " << t2 - t1 << std::endl;

		//parallel 2 - just a slightly different implementation
		// reset
		x = init<int>(n);
		t1 = omp_get_wtime();
		it_scan2(x);
		t2 = omp_get_wtime();
		std::cout << "par2 time: " << t2 - t1 << std::endl;

		// simpler parallel
		// reset
		x = init<int>(n);
		t1 = omp_get_wtime();
		simple_scan(x);
		t2 = omp_get_wtime();
		std::cout << "simple time: " << t2 - t1 << std::endl;

		// simpler parallel done differently
		// reset
		x = init<int>(n);
		t1 = omp_get_wtime();
		simple_scan2(x);
		t2 = omp_get_wtime();
		std::cout << "simple2 time: " << t2 - t1 << std::endl;
		print(x);
	}
	
	return 0;
}


