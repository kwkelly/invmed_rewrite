#include <profile.hpp>
#include <pvfmm_common.hpp>
#include <mpi.h>

int main(int argc, char* argv[]){
	MPI_Comm comm=MPI_COMM_WORLD;

	std::cout << "__PROFILE__: " << __PROFILE__ << std::endl;
	#ifdef __VERBOSE__
	std::cout << "__VERBOSE__: on" << std::endl;
	#endif

	pvfmm::Profile::Tic("ProfileTest",&comm,true);
	{
		std::cout << "Between tic and toc" << std::endl;
	}
	pvfmm::Profile::Toc();

	return 0;
}
