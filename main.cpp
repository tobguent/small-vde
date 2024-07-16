#include "field.hpp"
#include "io.hpp"
#include "vde.hpp"

#include <iostream>

int main(int argc, char* argv[])
{
	// is test data provided?
	if (argc < 2) {
		std::cout << "pass in the full path to the test data 'cyl2d.am'" << std::endl;
		return -1;
	}

	// load the vector field from file
	uvf2d vv;
	io::load_amira(argv[1], vv);

	// estimate the partial derivatives
	uvf2d vv_dx, vv_dy, vv_dt;
	partial::estimate(vv, vv_dx, vv_dy, vv_dt);

	// precompute spatial integrals
	usf2d g_bar;
	uvf2d u_bar;
	utf2d M_bar;
	vde::precompute(vv, vv_dx, vv_dy, vv_dt,
		5,		// neighborhood size R
		0,		// regularization epsilon
		g_bar, u_bar, M_bar);

	// compute vortex deviation error
	sf2d result;
	vde::compute(
		vv, vv_dx, vv_dy, vv_dt, g_bar, u_bar, M_bar,
		0,		// time slice to compute VDE for
		0.01,	// integration step size dt
		100,	// Number of integration steps 
		result);

	// plot the VDE field as bmp file
	io::write_bmp("vde.bmp", result,
		0.2);	// exponent for transfer function

	return 0;
}
