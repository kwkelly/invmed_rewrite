#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include <valarray>

#pragma once

#define MY_PRINT(x) std::cout << #x"=" << x << std::endl

// =====================================================================
// Declarations
// =====================================================================

template <typename T>
void op(T& v1, const T& v2);

template <typename T>
void rec_scan(std::vector<T> &a);

template <typename T>
void it_scan(std::vector<T> &a);

template <typename T>
void it_scan2(std::vector<T> &a);

template <typename T>
void seq_scan(std::vector<T> &a);

template <typename T>
void simple_scan(std::vector<T> &a);

template <typename T>
void simple_scan2(std::vector<T> &a);

template <typename T>
void ex_scan(std::vector<T> &a);

template<typename T>
std::ostream &operator<<(std::ostream &stream, std::valarray<T> ob)
{
	for(int i=0;i<ob.size();i++){
	  stream << ob[i] << '\n';
	}
	return stream;
}


// =====================================================================
// Definitions
// =====================================================================

/* 
 * rec_scan - recursive scan
 * does not use the generic operator op
 */
template <typename T>
void rec_scan(std::vector<T> &a){
	size_t n = a.size();
	if(n == 1){
	}
	else{
		std::vector<T> b(n/2);
		if(n%2 == 1){
			b.push_back(a[n-1]);
		}
		//#pragma omp parallel for
		for(int i=0;i<n/2; i++){
			b[i] = a[2*i] + a[2*i+1];	
		}
		rec_scan(b);

		//#pragma omp parallel for
		for(int i=1;i<n;i++){
			if(i%2 == 1) a[i] = b[i/2];
			else         a[i] = b[i/2-1] + a[i];
		}
	}
	return;
}

/* 
 * it_scan - iterative scan
 */ 
template <typename T>
void it_scan(std::vector<T> &a){
	size_t n = a.size();
	// Go down the tree
	for(size_t i=0;i<floor(log2(n)); i++){
		size_t step = std::pow(2,i);
		size_t start_idx = step - 1;
		#pragma omp parallel for
		for(size_t j=0;j<n/(2*step)-1;j++){
			op(a[start_idx + (2*j+1)*step], a[start_idx + 2*step*j]);
		}
		size_t j = n/(2*step)-1;
		if(start_idx + (2*j+1)*step < n){	
			op(a[start_idx + (2*j+1)*step], a[start_idx + 2*step*j]);
		}
	}

	// Go back up the tree
	size_t step = std::pow(2,ceil(log2(n)))/2;
	size_t start_idx = std::pow(2,ceil(log2(n)));
	for(size_t i=0;i<floor(log2(n)); i++){
		start_idx /= 2;
		step /= 2;
		size_t k = n/(2*step) - (n%(2*step) == 0) ;
		#pragma omp parallel for
		for(size_t j=0;j<k-1;j++){
			//std::cout << start_idx + (2*j+1)*step-1 << std::endl;
			op(a[start_idx + (2*j+1)*step-1], a[start_idx + 2*j*step-1]);
		}
		if(start_idx + (2*k-1)*step -1 < n){	
			op(a[start_idx + (2*k-1)*step-1], a[start_idx + (2*k-2)*step-1]);
		}
	}
	return;
}

/* 
 * it_scan2 - iterative scan
 * just a more readable version of it_scan ... I think so anyway. Be wary of bitshifts
 */ 
template <typename T>
void it_scan2(std::vector<T> &a){
	size_t n = a.size();
	size_t depth = floor(log2(n));
	size_t idx;
	size_t step;
	size_t start;
	size_t n_pairs;

	// Go down the tree
	for(size_t l=0;l<depth; l++){
		step = 1 << l;
		start = step - 1;
		n_pairs = n/(2*step)-1;
		#pragma omp parallel for private(idx)
		for(size_t j=0;j<n_pairs;j++){
			idx = start + 2*j*step;
			op(a[idx + step], a[idx]);
		}
		idx = start + 2*n_pairs*step;
		if(idx+step < n){	
			op(a[idx + step], a[idx]);
		}
	}

	// Go back up the tree
	step = (1<<int(ceil(log2(n))-1));
	start = step << 1;
	for(size_t l=0;l<depth; l++){
		start = start >> 1;
		step = step >> 1;
		n_pairs = n/(2*step) - (n%(2*step) == 0) ;
		#pragma omp parallel for private(idx)
		for(size_t j=0;j<n_pairs-1;j++){
			idx = start + 2*j*step-1; 
			op(a[idx+step], a[idx]);
		}
		idx = start + (2*n_pairs - 2)*step -1;
		if(idx+step < n){	
			op(a[idx+step], a[idx]);
		}
	}
	return;
}


/* 
 * seq_scan - sequential scan
 */
template <typename T>
void seq_scan(std::vector<T> &a){
	size_t n = a.size();
	for(size_t i=0;i<n-1;i++){
		op(a[i+1], a[i]);
	}
	return;
}

/* 
 * simple_scan - simple scan
 */ 
template <typename T>
void simple_scan(std::vector<T> &a){
	size_t n = a.size();
	size_t n_threads;
	std::vector<T> partial_sums;

	#pragma omp parallel
	{
			n_threads = omp_get_num_threads();
			size_t t_id= omp_get_thread_num();
			if(t_id == 0){
				partial_sums.resize(n_threads+1);
			}
	}

	// compute partial sums
	T sum_private = a[0]; //this guarantees that sum private has the same dim as a[0]
	sum_private = 0; // but then reset it to 0
	partial_sums[0] = sum_private;
	#pragma omp parallel for schedule(static) firstprivate(sum_private)
	for(int i=0;i<n;i++){
		size_t t_id= omp_get_thread_num();
		op(sum_private,a[i]);
		a[i] = sum_private;
		partial_sums[t_id+1] = sum_private;
	}
	
	// now we will compute a prefix sum over the partial sums
	for(int i=0;i<n_threads;i++){
		op(partial_sums[i+1],partial_sums[i]);
	}

	#pragma omp parallel for schedule(static)
	for(int i=0;i<n;i++){
		size_t t_id= omp_get_thread_num();
		op(a[i], partial_sums[t_id]);
	}
	return;
}


/* 
 * simple_scan2 - simple scan
 */ 
template <typename T>
void simple_scan2(std::vector<T> &a){
	size_t n = a.size();
	std::vector<T> partial_sums;
	#pragma omp parallel
	{
		int t_id = omp_get_thread_num();
		size_t n_threads = omp_get_num_threads();
		#pragma omp single
		{
			partial_sums.resize(n_threads+1);
			partial_sums[0] = a[0];
			partial_sums[0] = 0;
		}

		T sum = a[0];
		sum = 0;
		#pragma omp for schedule(static)
		for (size_t i=0; i<n; i++) {
			op(sum,a[i]);
			a[i] = sum;
		}
		partial_sums[t_id+1] = sum;
		#pragma omp barrier
		T offset = a[0];
		offset = 0;
		for(size_t i=0; i<(t_id+1); i++) {
			op(offset,partial_sums[i]);
		}
		#pragma omp for schedule(static)
		for (size_t i=0; i<n; i++) {
			op(a[i],offset);
		}
	}
	return;
}


/* 
 * ex_scan - simple exclusive scan
 */ 
template <typename T>
void ex_scan(std::vector<T> &a){
	size_t n = a.size();
	std::vector<T> partial_sums;
	#pragma omp parallel
	{
		int t_id = omp_get_thread_num();
		size_t n_threads = omp_get_num_threads();
		#pragma omp single
		{
			partial_sums.resize(n_threads+1);
			partial_sums[0] = a[0];
			partial_sums[0] = 0;
		}

		T sum = a[0];
		sum = 0;
		T temp = a[0];
		temp = 0;
		#pragma omp for schedule(static)
		for (size_t i=0; i<n; i++) {
			temp = a[i];
			a[i] = sum;
			op(sum,temp);
			//a[i] = sum;
		}
		partial_sums[t_id+1] = sum;
		#pragma omp barrier
		T offset = a[0];
		offset = 0;
		for(size_t i=0; i<(t_id+1); i++) {
			op(offset,partial_sums[i]);
		}
		#pragma omp for schedule(static)
		for (size_t i=0; i<n; i++) {
			op(a[i],offset);
		}
	}
	return;
}

