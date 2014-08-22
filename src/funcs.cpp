#include "funcs.hpp"
#include <cmath>
#include <math.h>
void eta_fn(double* coord, int n, double* out){ 
	double eta_ = 1;
  int dof=2;
	int COORD_DIM = 3;
  for(int i=0;i<n;i++){
    double* c=&coord[i*COORD_DIM];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			r_2 = sqrt(r_2);
      out[i*dof]=eta_*(r_2<0.1?0.01:0.0);//*exp(-L*r_2);
			//std::cout << out[i] << std::endl;
			out[i*dof+1] = 0; //complex part
    }
  }
}


void pt_sources_fn(double* coord, int n, double* out){ 
  int dof=2;
	int COORD_DIM = 3;
  double L=500;
  for(int i=0;i<n;i++){
    double* c=&coord[i*COORD_DIM];
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


void zero_fn(double* coord, int n, double* out){ 
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=0;
      out[i*dof+1]=0;
    }
  }
}


void one_fn(double* coord, int n, double* out){ 
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=1;
      out[i*dof+1]=0;
    }
  }
}

void eye_fn(double* coord, int n, double* out){ 
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=0;
      out[i*dof+1]=1;
    }
  }
}
