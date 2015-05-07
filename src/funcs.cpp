#include "funcs.hpp"
#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>

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

// smooth exponential
void eta_smooth_fn(const  double* coord, int n, double* out){
	double eta_ = 1;
	int dof=2;
	int COORD_DIM = 3;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r_8 = r_2*r_2*r_2*r_2*12*12*12*12*12*12*12*12;
			out[i*dof]= 0.01*exp(-r_8);
			out[i*dof+1] = 0; //complex part
		}
	}
}

// cos2 pi x
void cos2pix_fn(const  double* coord, int n, double* out){
	int dof=2;
	int COORD_DIM = 3;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof]= sin(2*M_PI*c[0]);
			out[i*dof+1] = 0; //complex part
		}
	}
}

// sin2 pi x
void sin2pix_fn(const  double* coord, int n, double* out){
	int dof=2;
	int COORD_DIM = 3;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof]= cos(2*M_PI*c[0]);
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

void two_pt_sol_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
	int dof=2;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r1=sqrt((c[0]-0.33)*(c[0]-0.33)+(c[1]-0.33)*(c[1]-0.33)+(c[2]-0.33)*(c[2]-0.33));
			double r2=sqrt((c[0]-0.66)*(c[0]-0.66)+(c[1]-0.66)*(c[1]-0.66)+(c[2]-0.66)*(c[2]-0.66));
			// Assumes that k = 1;
			if(dof>1) out[i*dof+0]= 1/(4*M_PI*r1)*cos(1*r1) + 1/(4*M_PI*r2)*cos(1*r2);
			if(dof>1) out[i*dof+1]= 1/(4*M_PI*r1)*sin(1*r1) + 1/(4*M_PI*r2)*sin(1*r2);
		}
	}

}


void eight_pt_sol_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
	int dof=2;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r1=sqrt((c[0]-0.375)*(c[0]-0.375)+(c[1]-0.375)*(c[1]-0.375)+(c[2]-0.375)*(c[2]-0.375));
			double r2=sqrt((c[0]-0.625)*(c[0]-0.625)+(c[1]-0.375)*(c[1]-0.375)+(c[2]-0.375)*(c[2]-0.375));
			double r3=sqrt((c[0]-0.375)*(c[0]-0.375)+(c[1]-0.625)*(c[1]-0.625)+(c[2]-0.375)*(c[2]-0.375));
			double r4=sqrt((c[0]-0.625)*(c[0]-0.625)+(c[1]-0.625)*(c[1]-0.625)+(c[2]-0.375)*(c[2]-0.375));
			double r5=sqrt((c[0]-0.375)*(c[0]-0.375)+(c[1]-0.375)*(c[1]-0.375)+(c[2]-0.625)*(c[2]-0.625));
			double r6=sqrt((c[0]-0.625)*(c[0]-0.625)+(c[1]-0.375)*(c[1]-0.375)+(c[2]-0.625)*(c[2]-0.625));
			double r7=sqrt((c[0]-0.375)*(c[0]-0.375)+(c[1]-0.625)*(c[1]-0.625)+(c[2]-0.625)*(c[2]-0.625));
			double r8=sqrt((c[0]-0.625)*(c[0]-0.625)+(c[1]-0.625)*(c[1]-0.625)+(c[2]-0.625)*(c[2]-0.625));
			// Assumes that k = 1;
			if(dof>1) out[i*dof+0]= 1/(4*M_PI*r1)*cos(1*r1) + 1/(4*M_PI*r2)*cos(1*r2) +  1/(4*M_PI*r3)*cos(1*r3) +  1/(4*M_PI*r4)*cos(1*r4)
				+  1/(4*M_PI*r5)*cos(1*r5) +  1/(4*M_PI*r6)*cos(1*r6) +  1/(4*M_PI*r7)*cos(1*r7) +  1/(4*M_PI*r8)*cos(1*r8);
			if(dof>1) out[i*dof+1]= 1/(4*M_PI*r1)*sin(1*r1) + 1/(4*M_PI*r2)*sin(1*r2) +  1/(4*M_PI*r3)*sin(1*r3) +  1/(4*M_PI*r4)*sin(1*r4)
				+  1/(4*M_PI*r5)*sin(1*r5) +  1/(4*M_PI*r6)*sin(1*r6) +  1/(4*M_PI*r7)*sin(1*r7) +  1/(4*M_PI*r8)*sin(1*r8);
			//if(dof>1) out[i*dof+1]= 1/(4*M_PI*r1)*sin(1*r1) + 1/(4*M_PI*r2)*sin(1*r2);
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
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r = sqrt(r_2);
			// Assumes that k = 1;
			if(dof>1) out[i*dof+0]= cos(10*r);
			if(dof>1) out[i*dof+1]= sin(10*r);
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
			if(dof>1) out[i*dof+0]= sin(r);
			if(dof>1) out[i*dof+1]= cos(r);
		}
	}

}


// sin + icos
void sc_osc_fn(const double* coord, int n, double* out){
	int COORD_DIM = 3;
	int dof=2;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r = sqrt(r_2);
			// Assumes that k = 1;
			if(dof>1) out[i*dof+0]= sin(10*r);
			if(dof>1) out[i*dof+1]= cos(10*r);
		}
	}

}

//sin - icos
void scc_fn(const double* coord, int n, double* out){
	int coord_dim = 3;
	int dof=2;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*coord_dim];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r=sqrt(r_2);
			// assumes that k = 1;
			if(dof>1) out[i*dof+0]= sin(r);
			if(dof>1) out[i*dof+1]= -cos(r);
		}
	}

}

//(sin + icos)^2
void sc2_fn(const double* coord, int n, double* out){
	int coord_dim = 3;
	int dof=2;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*coord_dim];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			double r=sqrt(r_2);
			// assumes that k = 1;
			if(dof>1) out[i*dof+0]= sin(r)*sin(r) - cos(r)*cos(r);
			if(dof>1) out[i*dof+1]= 2*sin(r)*cos(r);
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
			if(dof>1) out[i*dof+0]= 2*sin(r);
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

void prod_fn(const double* coord, int n, double* out){
	int COORD_DIM=3;
	int dof=2;
	double a=-160;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof+0]=c[0]*c[1]*c[2];
			out[i*dof+1]=c[0]*c[1]*c[2];
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

// Create incident field generate by a point function. This just
// the Green's function for the Helmholtz problem with wavenumber k
// evaluated at the difference of the current point and the source point
void pt_src_sol_fn(const double* coord, int n, double* out, const double* src_coord){
	int COORD_DIM = 3;
	int dof=2;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		const double* s=&src_coord[0]; // only have one pt source for this thing
		{
			double r=sqrt((c[0]-s[0])*(c[0]-s[0])+(c[1]-s[1])*(c[1]-s[1])+(c[2]-s[2])*(c[2]-s[2]));
			// Assumes that k = 1;
			if(dof>1) out[i*dof+0]= 1/(4*M_PI*r)*cos(1*r);
			if(dof>1) out[i*dof+1]= 1/(4*M_PI*r)*sin(1*r);
		}
	}
}

void linear_comb_of_pt_src(const double* coord, int n, double* out, const std::vector<double> &coeffs, const std::vector<double> &src_coords){
	int COORD_DIM = 3;
	int dof=2;
	//#pragma omp parallel for
	for(int i=0;i<n;i++){
		if(dof>0) out[i*dof+0]= 0;
		if(dof>1) out[i*dof+1]= 0;
	}
	//#pragma omp parallel for
	for(int j=0; j<coeffs.size()/dof;j++){
		for(int i=0;i<n;i++){
			const double* c=&coord[i*COORD_DIM];
			const double* s=&src_coords[j*COORD_DIM];
			double r=sqrt((c[0]-s[0])*(c[0]-s[0])+(c[1]-s[1])*(c[1]-s[1])+(c[2]-s[2])*(c[2]-s[2]));
			// Assumes that k = 1;
			if(dof>0) out[i*dof+0] += (coeffs[j*dof]*1/(4*M_PI*r)*cos(1*r) - coeffs[j*dof+ 1]*1/(4*M_PI*r)*sin(1*r));
			if(dof>1) out[i*dof+1] += (coeffs[j*dof]*1/(4*M_PI*r)*sin(1*r) + coeffs[j*dof+1]*1/(4*M_PI*r)*cos(1*r));
		}
	}
}

// Smooth varying background function
void k2_fn(const  double* coord, int n, double* out){
	int dof=2;
	int COORD_DIM = 3;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			r_2 = sqrt(r_2);
			out[i*dof]=.5*cos(3*r_2);
			out[i*dof+1] = .5*sin(3*r_2); //complex part
		}
	}
}

void eta_plus_k2_fn(const  double* coord, int n, double* out){
	int dof=2;
	int COORD_DIM = 3;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			double r_2=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.5)*(c[1]-0.5)+(c[2]-0.5)*(c[2]-0.5);
			r_2 = sqrt(r_2);
			out[i*dof]=.1*cos(3*r_2) + (r_2<0.1?0.01:0.0);
			out[i*dof+1] = .1*sin(3*r_2); //complex part
		}
	}
}


void mask_fn(const  double* coord, int n, double* out){
	int dof=2;
	int COORD_DIM = 3;
	double min = .25;
	double max = .75;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof+0] = ((min < c[0] and c[0] < max) and (min < c[1] and c[1] < max) and (min < c[2] and c[2] < max)) ? 1 : 0;
			out[i*dof+1] = 0;
		}
	}
}

void cmask_fn(const  double* coord, int n, double* out){
	int dof=2;
	int COORD_DIM = 3;
	double min = .25;
	double max = .75;
	for(int i=0;i<n;i++){
		const double* c=&coord[i*COORD_DIM];
		{
			out[i*dof+0] = ((min < c[0] and c[0] < max) and (min < c[1] and c[1] < max) and (min < c[2] and c[2] < max)) ? 0 : 1;
			out[i*dof+1] = 0;
		}
	}
}

void eta2_fn(const  double* coord, int n, double* out){
  double eta_ = 1;
  int dof=2;
  int COORD_DIM = 3;
  for(int i=0;i<n;i++){
    const double* c=&coord[i*COORD_DIM];
    {
      double sz = 0.01;
      double r_21=(c[0]-0.5)*(c[0]-0.5)+(c[1]-0.375)*(c[1]-0.375)+(c[2]-0.5)*(c[2]-0.5);
      r_21 = sqrt(r_21);
      double r_22=(c[0]-0.375)*(c[0]-0.375)+(c[1]-0.625)*(c[1]-0.625)+(c[2]-0.5)*(c[2]-0.5);
      r_22 = sqrt(r_22);
      double r_23=(c[0]-0.625)*(c[0]-0.625)+(c[1]-0.625)*(c[1]-0.625)+(c[2]-0.5)*(c[2]-0.5);
      r_23 = sqrt(r_23);
      out[i*dof]=eta_*(r_21<0.11?sz:0.0) + eta_*(r_22<0.075?sz:0.0) + eta_*(r_23<0.1?sz:0.0);
      //std::cout << out[i] << std::endl;
      out[i*dof+1] = 0; //complex part
    }
  }
}
