#include <cmath>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <cassert>
#include "point_distribs.hpp"

template <class Real_t>
std::vector<Real_t> unif_point_distrib(size_t N, Real_t lmin, Real_t lmax, MPI_Comm comm){

	/*
	 * Generate equidistributed points in the unit cube between lmin and lmax in each dimension
	 */

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

	/*
	 * Generate uniformly random points points in the unit cube between lmin and lmax in each dimension
	 */

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

	/*
	 * Generate unformly random points in the the sphere of radius rad centered at (0.5,0.5,0.5)
	 */

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


template <class Real_t>
std::vector<Real_t> unif_plane(size_t N, int plane, Real_t pos, MPI_Comm comm){

	/*
	 * Generate uniformly spaced points on a plane at distance pos from the given axis
	 * plane == 0 = yz plane
	 * plane == 1 = xz plane
	 * plane == 2 = xy plane
	 */

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
					coord.push_back( pos                              );
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


std::vector<double> test_pts(){

	/*
	 * Generate the test points listed below. Note that all points are create on all procs
	 */

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

	/*
	 * Generate n_points equidistributed points on the
	 * surface of a sphere of radius rad
	 * http://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
	 * Note that this function is not rwritten to  be parallel friendly, i.e. all points
	 * will be created on all procs
	 */

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

/*
 * Instantiate tempaltes
 */

template 
std::vector<double> unif_point_distrib(size_t N, double lmin, double lmax, MPI_Comm comm);

template
std::vector<double> rand_unif_point_distrib(size_t N, double lmin, double lmax, MPI_Comm comm);

template
std::vector<double> rand_sph_point_distrib(size_t N, double rad, MPI_Comm comm);

template
std::vector<double> unif_plane(size_t N, int plane, double pos, MPI_Comm comm);

