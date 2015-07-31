#include <pvfmm.hpp>

#pragma once

/*
 * Function for the helm kernel that accepts a variable which is the wavenumber k
 */
void helm_kernel_fn_var(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr, double k);

/*
 * Helmholtz kernel with the wavenumber k=1
 */
void helm_kernel_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

/*
 * Kernel associated to the adjoiunt operator which just has wavenumber k=-1
 */
void helm_kernel_conj_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

/*
 * Kernels with slighty higher wavenumbers = 3 and -3
 */
void helm_kernel3_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

void helm_kernel_conj3_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

/*
 * Low wave number kernels with wavenumber =+/- 0.1
 */
void helm_kernel_low_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

void helm_kernel_conj_low_fn(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, pvfmm::mem::MemoryManager* mem_mgr);

const pvfmm::Kernel<double> helm_kernel=pvfmm::BuildKernel<double, helm_kernel_fn>("helm_kernel", 3, std::pair<int,int>(2,2));

const pvfmm::Kernel<double> helm_kernel_conj=pvfmm::BuildKernel<double, helm_kernel_conj_fn>("helm_kernel_conj", 3, std::pair<int,int>(2,2));

const pvfmm::Kernel<double> helm_kernel3=pvfmm::BuildKernel<double, helm_kernel3_fn>("helm_kernel3", 3, std::pair<int,int>(2,2));

const pvfmm::Kernel<double> helm_kernel_conj3=pvfmm::BuildKernel<double, helm_kernel_conj3_fn>("helm_kernel_conj3", 3, std::pair<int,int>(2,2));

const pvfmm::Kernel<double> helm_kernel_low=pvfmm::BuildKernel<double, helm_kernel_low_fn>("helm_kernel_low", 3, std::pair<int,int>(2,2));

const pvfmm::Kernel<double> helm_kernel_conj_low=pvfmm::BuildKernel<double, helm_kernel_conj_low_fn>("helm_kernel_conj_low", 3, std::pair<int,int>(2,2));

#include "helm_kernels.txx"
