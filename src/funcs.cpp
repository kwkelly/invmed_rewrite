#include "funcs.hpp"
#include <cmath>
#include <math.h>

// 0 outside of a sphere of radius 0.1, 0.01 inside that.
void eta_fn(const  double* coord, int n, double* out){ 
	double eta_ = 1;
  int dof=2;
	int COORD_DIM = 3;
  for(int i=0;i<n;i++){
     const double* c=&coord[i*COORD_DIM];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			r_2 = sqrt(r_2);
      out[i*dof]=eta_*(r_2<0.1?0.01:0.0);//*exp(-L*r_2);
			//std::cout << out[i] << std::endl;
			out[i*dof+1] = 0; //complex part
    }
  }
}

// Using gaussians to approximate pt sources located neat the 
// edge of the computational domain
void pt_sources_fn(const double* coord, int n, double* out){ 
  int dof=2;
	int COORD_DIM = 3;
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

// 0
void zero_fn(const  double* coord, int n, double* out){ 
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
     const double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=0;
      out[i*dof+1]=0;
    }
  }
}

// 1
void one_fn( const double* coord, int n, double* out){ 
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
     const double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=1;
      out[i*dof+1]=0;
    }
  }
}

// i
void eye_fn( const double* coord, int n, double* out){ 
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
     const double* c=&coord[i*COORD_DIM];
    {
      out[i*dof]=0;
      out[i*dof+1]=1;
    }
  }
}

//Solution of integrating the Helmholtz green function against
//a point source at the center of the computational domain.
void ctr_pt_sol_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= 1/(4*M_PI*r)*cos(1*r);
      if(dof>1) out[i*dof+1]= 1/(4*M_PI*r)*sin(1*r);
    }
  }

}

void ctr_pt_sol_neg_conj_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= -1/(4*M_PI*r)*cos(1*r);
      if(dof>1) out[i*dof+1]= 1/(4*M_PI*r)*sin(1*r);
    }
  }

}

// The above function multiplied by i
void ctr_pt_sol_i_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= -1/(4*M_PI*r)*sin(1*r);
      if(dof>1) out[i*dof+1]= 1/(4*M_PI*r)*cos(1*r);
    }
  }

}

// conjugate of the function two above
void ctr_pt_sol_conj_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= 1/(4*M_PI*r)*cos(1*r);
      if(dof>1) out[i*dof+1]= -1/(4*M_PI*r)*sin(1*r);
    }
  }

}

// product of ctr_pt_sol_fn and ctr_pt_sol_i_fn
void ctr_pt_sol_prod_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= 1/(16*M_PI*M_PI*r*r)*cos(2*r);
      if(dof>1) out[i*dof+1]= 1/(16*M_PI*M_PI*r*r)*cos(2*r);
    }
  }

}

// product of ctr_pt_sol_fn and ctr_pt_sol_conj_fn
void ctr_pt_sol_conj_prod_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= 1/(16*M_PI*M_PI*r*r);
      if(dof>1) out[i*dof+1]= 0;
    }
  }

}

// cos + isin
void cs_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r=sqrt((c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5));
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= cos(r);
      if(dof>1) out[i*dof+1]= sin(r);
    }
  }

}
// sin + icos
void sc_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r = sqrt(r_2);
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= sin(r_2);
      if(dof>1) out[i*dof+1]= cos(r_2);
    }
  }

}

//sin - icos
void scc_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      double r=sqrt(r_2);
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= sin(r_2);
      if(dof>1) out[i*dof+1]= -cos(r_2);
    }
  }

}


//2sin
void twosin_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
  int dof=2;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r= sqrt(r_2);
			// Assumes that k = 1;
      if(dof>1) out[i*dof+0]= 2*sin(r_2);
      if(dof>1) out[i*dof+1]= 0;
    }
  }

}


// point source at the center
void ctr_pt_source_fn(const double* coord, int n, double* out){ 
  int dof=2;
	int COORD_DIM = 3;
  double L=500;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
			double temp;
      double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
      if(dof>1) out[i*dof+0]= sqrt(L/M_PI)*exp(-L*r_2);
      if(dof>1) out[i*dof+1]=0;
    }
  }
}


void int_test_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	double mu=1;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			out[i*dof+0]=((2*a*r_2+3)*2*a*exp(a*r_2)+mu*mu*exp(a*r_2));
			out[i*dof+1]=0;
		}
	}
}

void int_test_sol_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			out[i*dof+0]=-exp(a*r_2);
			out[i*dof+1]=0;
		}
	}
}


void poly_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r = sqrt(r_2);
			out[i*dof+0]= 1 + r_2 + r_2*r_2;
			//out[i*dof+0]= 1;
			out[i*dof+1]=0;
		}
	}
}


void poly_prod_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r = sqrt(r_2);
			out[i*dof+0]= 1 + 2*r_2 + 3*r_2*r_2 + 2*r_2*r_2*r_2 + r_2*r_2*r_2*r_2;
		//	out[i*dof+0]= 1 ;
			out[i*dof+1]=0;
		}
	}
}


void linear_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof+0]= c[0] + c[1] + c[2];
			out[i*dof+1]=0;
		}
	}
}


void linear_prod_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof+0]= (c[0] + c[1] + c[2])*(c[0] + c[1] + c[2]);
			out[i*dof+1]=0;
		}
	}
}
