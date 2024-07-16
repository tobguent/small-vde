#pragma once

#include "field.hpp"

/**
 * @brief Implementation of Vortex Deviation Error (VDE).
*/
class vde
{
public:
	/**
	 * @brief Performs spatial integration of g_bar, u_bar, and M_bar using Eqs. (53)-(58).
	 * @param _vv Velocity field.
	 * @param _vv_dx x-partial of velocity field.
	 * @param _vv_dy y-partial of velocity field.
	 * @param _vv_dt t-partial of velocity field.
	 * @param _R Size of neighborhood region (2R+1 x 2R+1).
	 * @param _epsilon Regularization weight (default = 0).
	 * @param _g_bar Output field for g_bar.
	 * @param _u_bar Output field for u_bar.
	 * @param _M_bar Output field for M_bar.
	*/
	static void precompute(const uvf2d& _vv, const uvf2d& _vv_dx, const uvf2d& _vv_dy, const uvf2d& _vv_dt, int _R, double _epsilon, usf2d& _g_bar, uvf2d& _u_bar, utf2d& _M_bar)
	{
		// allocate output fields
		_g_bar.resolution = _vv.resolution;
		_u_bar.resolution = _vv.resolution;
		_M_bar.resolution = _vv.resolution;
		_g_bar.domain = _vv.domain;
		_u_bar.domain = _vv.domain;
		_M_bar.domain = _vv.domain;
		_g_bar.data.resize(_g_bar.resolution.prod());
		_u_bar.data.resize(_u_bar.resolution.prod());
		_M_bar.data.resize(_M_bar.resolution.prod());

		// iterate all voxels of the pp grid
		Eigen::Vector3i resolution = _vv.resolution;
		int numElements = resolution.prod();

#ifdef NDEBUG
#pragma omp parallel for
#endif
		for (int ivertex_pp = 0; ivertex_pp < numElements; ++ivertex_pp)
		{
			// sample pathline geometry
			Eigen::Vector3i grid_index_pp = _vv.grid_index(ivertex_pp);
			auto pp_ = _vv.coordinate(grid_index_pp); // get the physical coordinate in the domain
			Eigen::Vector2d pp(pp_.x(), pp_.y());
			Eigen::Vector2d ppdot = _vv.value(grid_index_pp);
			Eigen::Vector2d ppddot = _vv_dx.value(grid_index_pp) * ppdot.x() + _vv_dy.value(grid_index_pp) * ppdot.y() + _vv_dt.value(grid_index_pp);

			// prepare variables to accumulate in
			double g_bar = 0;
			Eigen::Vector2d u_bar = Eigen::Vector2d::Zero();
			Eigen::Matrix2d M_bar = Eigen::Matrix2d::Zero();
			int count = 0;

			// iterate neighborhood
			for (int grid_index_xx_i = std::max(0, grid_index_pp.x() - _R); grid_index_xx_i <= std::min(grid_index_pp.x() + _R, resolution.x() - 1); ++grid_index_xx_i)
			{
				for (int grid_index_xx_j = std::max(0, grid_index_pp.y() - _R); grid_index_xx_j <= std::min(grid_index_pp.y() + _R, resolution.y() - 1); ++grid_index_xx_j)
				{
					// sample information at point xx in neighborhood
					Eigen::Vector3i grid_index_xx(grid_index_xx_i, grid_index_xx_j, grid_index_pp.z());
					Eigen::Vector3d xx_ = _vv.coordinate(grid_index_xx);
					Eigen::Vector2d xx(xx_.x(), xx_.y());
					const Eigen::Vector2d& vv = _vv.value(grid_index_xx);
					const Eigen::Vector2d& vv_x = _vv_dx.value(grid_index_xx);
					const Eigen::Vector2d& vv_y = _vv_dy.value(grid_index_xx);
					const Eigen::Vector2d& vv_t = _vv_dt.value(grid_index_xx);

					Eigen::Vector2d vdotp = vdot_p(
						ppdot, ppddot,
						vv_x, vv_y, vv_t);

					Eigen::Matrix2d Mp = M_p(
						pp, ppdot, ppddot,
						xx, vv, vv_x, vv_y, vv_t);

					// sum up the variables
					g_bar += vdotp.dot(vdotp);
					u_bar += Mp.transpose() * vdotp;
					M_bar += Mp.transpose() * Mp;

					// add regularization
					if (_epsilon > 0)
					{
						Eigen::Vector2d vdotrp = vdot_rp(ppdot, vv);
						Eigen::Matrix2d Mrp = M_rp(pp, xx);

						g_bar += _epsilon * vdotrp.dot(vdotrp);
						u_bar += _epsilon * Mrp.transpose() * vdotrp;
						M_bar += _epsilon * Mrp.transpose() * Mrp;
					}
					count += 1;
				}
			}

			// compute the averages
			double avg_g_bar = count ? g_bar / count : 0;
			Eigen::Vector2d avg_u_bar = count ? (Eigen::Vector2d)(u_bar / count) : Eigen::Vector2d::Zero();
			Eigen::Matrix2d avg_M_bar = count ? (Eigen::Matrix2d)(M_bar / count) : Eigen::Matrix2d::Zero();

			// store information on grid
			_g_bar.set_value(grid_index_pp, Eigen::Vector<double, 1>(avg_g_bar));
			_u_bar.set_value(grid_index_pp, avg_u_bar);
			_M_bar.set_value(grid_index_pp, avg_M_bar);
		}
	}

	/**
	 * @brief Calculates VDE for a given time slice.
	 * @param _vv Velocity field.
	 * @param _vv_dx x-partial of velocity field.
	 * @param _vv_dy y-partial of velocity field.
	 * @param _vv_dt t-partial of velocity field.
	 * @param _g_bar Precomputed field g_bar.
	 * @param _u_bar Precomputed field u_bar.
	 * @param _M_bar Precomputed field M_bar.
	 * @param _time_slice time slice to compute VDE for.
	 * @param _dt Integration step size for the pathline tracing.
	 * @param _N Number of pathline integration steps (line will have length N+1).
	 * @param _vde Output field for VDE.
	*/
	static void compute(const uvf2d& _vv, const uvf2d& _vv_dx, const uvf2d& _vv_dy, const uvf2d& _vv_dt, const usf2d& _g_bar, const uvf2d& _u_bar, const utf2d& _M_bar, int _time_slice, double _dt, int _N, sf2d& _vde)
	{
		// allocate output field
		_vde.resolution = Eigen::Vector2i(_vv.resolution.x(), _vv.resolution.y());
		_vde.domain = Eigen::AlignedBox2d(Eigen::Vector2d(_vv.domain.min().x(), _vv.domain.min().y()), Eigen::Vector2d(_vv.domain.max().x(), _vv.domain.max().y()));
		_vde.data.resize(_vde.resolution.prod());

		auto resolution = _vv.resolution;
		int numElements = resolution.x() * resolution.y();

#ifdef NDEBUG
#pragma omp parallel for
#endif
		for (int ivertex = 0; ivertex < numElements; ++ivertex)
		{
			int vertex_offset = ivertex + _time_slice * resolution.x() * resolution.y();
			auto grid_index = _vv.grid_index(vertex_offset);
			auto position = _vv.coordinate(grid_index); // get the physical coordinate in the domain

			if (_vv.domain.contains(position)) // if the sample location is inside the domain of the input field
			{
				// compute VDE
				std::pair<double, bool> result = compute_vde(_vv, _vv_dx, _vv_dy, _vv_dt, _g_bar, _u_bar, _M_bar,
					position, _dt, _N);

				// assign to output voxel
				if (result.second)
					_vde.set_value(Eigen::Vector2i(grid_index.x(), grid_index.y()), Eigen::Vector<double, 1>(result.first));
				else _vde.set_value(Eigen::Vector2i(grid_index.x(), grid_index.y()), Eigen::Vector<double, 1>(0.));
			}
		}
	}

private:
	/**
	 * @brief Evaluates vdot_p in Eq. (44).
	 * @param _ppdot Pathline tangent at time t.
	 * @param _ppddot Pathline acceleration at time t.
	 * @param _vv_x x-partial of velocity at (xx, t).
	 * @param _vv_y y-partial of velocity at (xx, t).
	 * @param _vv_t t-partial of velocity at (xx, t).
	 * @return vdot_p.
	*/
	static Eigen::Vector2d vdot_p(
		const Eigen::Vector2d& _ppdot,
		const Eigen::Vector2d& _ppddot,
		const Eigen::Vector2d& _vv_x,
		const Eigen::Vector2d& _vv_y,
		const Eigen::Vector2d& _vv_t)
	{
		// compute matrices
		Eigen::Matrix2d J;
		J << _vv_x, _vv_y;

		// compute vector
		Eigen::Vector2d vdotp = J * _ppdot + _vv_t - _ppddot;
		return vdotp;
	}

	/**
	 * @brief Evaluates vdot_rp in Eq. (44).
	 * @param _ppdot Pathline tangent at time t.
	 * @param _vv Velocity at (xx, t).
	 * @return vdot_rp.
	*/
	static Eigen::Vector2d vdot_rp(const Eigen::Vector2d& _ppdot, const Eigen::Vector2d& _vv)
	{
		return _vv - _ppdot;
	}

	/**
	 * @brief Evaluates M_p in Eq. (45).
	 * @param _pp Pathline vertex at time t.
	 * @param _ppdot Pathline tangent at time t.
	 * @param _ppddot Pathline acceleration at time t.
	 * @param _xx Position xx.
	 * @param _vv Velocity at (xx, t).
	 * @param _vv_x x-partial of velocity at (xx, t).
	 * @param _vv_y y-partial of velocity at (xx, t).
	 * @param _vv_t t-partial of velocity at (xx, t).
	 * @return M_p.
	*/
	static Eigen::Matrix2d M_p(
		const Eigen::Vector2d& _pp,
		const Eigen::Vector2d& _ppdot,
		const Eigen::Vector2d& _ppddot,
		const Eigen::Vector2d& _xx,
		const Eigen::Vector2d& _vv,
		const Eigen::Vector2d& _vv_x,
		const Eigen::Vector2d& _vv_y,
		const Eigen::Vector2d& _vv_t)
	{
		// compute deltas
		Eigen::Vector2d delta_xx = _xx - _pp;
		Eigen::Vector2d delta_vv = _vv - _ppdot;

		// compute matrices
		Eigen::Matrix2d J;
		J << _vv_x, _vv_y;
		Eigen::Matrix2d Q;
		Q << 0, 1, -1, 0;

		// compute matrix
		Eigen::Matrix2d Mp;
		Mp << J * Q * delta_xx - Q * delta_vv, -Q * delta_xx;
		return Mp;
	}

	/**
	 * @brief Evaluates M_rp in Eq. (46).
	 * @param _pp Pathline vertex at time t.
	 * @param _xx Position xx.
	 * @return M_rp.
	*/
	static Eigen::Matrix2d M_rp(const Eigen::Vector2d& _pp, const Eigen::Vector2d& _xx)
	{
		Eigen::Matrix2d Q;
		Q << 0, 1, -1, 0;
		Eigen::Vector2d temp = -Q * (_xx - _pp);
		Eigen::Matrix2d Mrp;
		Mrp << temp.x(), 0,
			temp.y(), 0;
		return Mrp;
	}

	/**
	 * @brief Generates the scalar g in Eq. (67).
	 * @param _g_bars List of gbar sampled along the pathline.
	 * @return g.
	*/
	static double generate_g(const std::vector<double>& _g_bars)
	{
		size_t N = _g_bars.size();
		double result = 0;
		for (int i = 0; i < N; ++i)
			result += _g_bars[i];
		return result / N; // Note that our "N" in code is "N+1" in the paper;
	}

	/**
	 * @brief Generates the vector u in Eq. (68)
	 * @param _u_bars List of ubar sampled along the pathline.
	 * @param _L Derivative operator.
	 * @return u.
	*/
	static Eigen::VectorXd generate_u(std::vector<Eigen::Vector2d> _u_bars, const Eigen::SparseMatrix<double>& _L)
	{
		size_t N = _u_bars.size();
		Eigen::VectorXd uu1(N), uu2(N);
		for (size_t i = 0; i < N; ++i)
		{
			uu1(i) = _u_bars[i].x();
			uu2(i) = _u_bars[i].y();
		}
		return (uu1 + _L.transpose() * uu2) / N;// Note that our "N" in code is "N+1" in the paper;
	}

	/**
	 * @brief Generates the matrix M in Eq. (68).
	 * @param _M_bars List of Mbar sampled along the pathline.
	 * @param _L Derivative operator.
	 * @return M.
	*/
	static Eigen::SparseMatrix<double> generate_M(std::vector<Eigen::Matrix2d> _M_bars, const Eigen::SparseMatrix<double>& _L)
	{
		size_t N = _M_bars.size();
		Eigen::SparseMatrix<double> M11(N, N), M12(N, N), M22(N, N);
		for (size_t i = 0; i < N; ++i)
		{
			M11.coeffRef(i, i) = _M_bars[i](0);
			M12.coeffRef(i, i) = _M_bars[i](1);
			M22.coeffRef(i, i) = _M_bars[i](3);
		}
		return (M11 + _L.transpose() * M22 * _L + _L.transpose() * M12 + M12 * _L) / N;// Note that our "N" in code is "N+1" in the paper;
	}

	/**
	 * @brief Generates the derivative operator L in Eq. (64).
	 * @param _num_vertices Number of pathline vertices.
	 * @param _dt Integration step size for the pathline tracing.
	 * @return L.
	*/
	static Eigen::SparseMatrix<double> generate_L(Eigen::Index _num_vertices, double _dt)
	{
		Eigen::SparseMatrix<double> result(_num_vertices, _num_vertices);
		if (_num_vertices < 2)
			return result;

		for (int i = 0; i < _num_vertices; ++i)
		{
			if (i == 0)
			{
				result.coeffRef(0, 0) = -3;
				result.coeffRef(0, 1) = 4;
				result.coeffRef(0, 2) = -1;
			}
			else if (i == _num_vertices - 1)
			{
				result.coeffRef(_num_vertices - 1, _num_vertices - 3) = 1;
				result.coeffRef(_num_vertices - 1, _num_vertices - 2) = -4;
				result.coeffRef(_num_vertices - 1, _num_vertices - 1) = 3;
			}
			else
			{
				result.coeffRef(i, i - 1) = -1;
				result.coeffRef(i, i) = 0;
				result.coeffRef(i, i + 1) = 1;
			}
		}
		return result / (2 * _dt);
	}

	/**
	 * @brief Solves for c_opt using Eq. (68).
	 * @param _M System matrix M.
	 * @param _u Right-hand side u.
	 * @param _success Writes true if the solver was successful.
	 * @return c_opt.
	*/
	static Eigen::VectorXd solve_c_opt(const Eigen::SparseMatrix<double>& _M, const Eigen::VectorXd& _u, bool& _success)
	{
		// create a Sparse Cholesky solver
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

		// compute factorization
		solver.compute(_M);
		if (solver.info() != Eigen::Success)
		{
			_success = false;
			return Eigen::VectorXd::Zero(_M.rows());
		}

		// solver the linear system
		Eigen::VectorXd copt = solver.solve(_u);
		if (solver.info() != Eigen::Success)
		{
			_success = false;
			return Eigen::VectorXd::Zero(_M.rows());
		}

		// return result
		_success = true;
		return copt;
	}

	/**
	 * @brief Calculates VDE at a given space-time position by releasing a pathline, sampling the precomputed fields, setting up the system and solving it.
	 * @param _vv Velocity field.
	 * @param _vv_x x-partial of velocity field.
	 * @param _vv_y y-partial of velocity field.
	 * @param _vv_t t-partial of velocity field.
	 * @param _g_bar Precomputed field g_bar.
	 * @param _u_bar Precomputed field u_bar.
	 * @param _M_bar Precomputed field M_bar.
	 * @param _pos Space-time position where to evaluate VDE.
	 * @param _dt Integration step size for the pathline tracing.
	 * @param _N Number of pathline integration steps (line will have length N+1).
	 * @return Pair containing the VDE value and a flag that determines whether the computation was successful.
	*/
	static std::pair<double, bool> compute_vde(
		const uvf2d& _vv, const uvf2d& _vv_dx, const uvf2d& _vv_dy, const uvf2d& _vv_dt, const usf2d& _g_bar, const uvf2d& _u_bar, const utf2d& _M_bar,
		Eigen::Vector3d _pos, double _dt, int _N)
	{
		// trace pathline forward in time using RK4
		std::vector<Eigen::Vector3d> curve;
		curve.reserve(_N + 1);
		curve.push_back(_pos);
		for (int istep = 0; istep < _N; ++istep)
		{
			Eigen::Vector2d k1 = _vv.value(_pos);
			Eigen::Vector3d k1_(k1.x(), k1.y(), 1.);
			Eigen::Vector2d k2 = _vv.value((_pos + k1_ * _dt * 0.5).eval());
			Eigen::Vector3d k2_(k2.x(), k2.y(), 1.);
			Eigen::Vector2d k3 = _vv.value((_pos + k2_ * _dt * 0.5).eval());
			Eigen::Vector3d k3_(k3.x(), k3.y(), 1.);
			Eigen::Vector2d k4 = _vv.value((_pos + k3_ * _dt).eval());
			Eigen::Vector3d k4_(k4.x(), k4.y(), 1.);
			_pos += _dt * (k1_ + 2 * k2_ + 2 * k3_ + k4_) / 6;
			if (_vv.domain.contains(_pos))
				curve.push_back(_pos);
			else break;
		}

		int N = curve.size();
		if (N > 2)
		{
			// sample precomputed values along pathline
			std::vector<double> g_bars(N);
			std::vector<Eigen::Vector2d> u_bars(N);
			std::vector<Eigen::Matrix2d> M_bars(N);
			for (int itimestep = 0; itimestep < N; ++itimestep)
			{
				Eigen::Vector3d xx_ = curve[itimestep];
				g_bars[itimestep] = _g_bar.value(xx_).x();
				u_bars[itimestep] = _u_bar.value(xx_);
				M_bars[itimestep] = _M_bar.value(xx_);
			}

			// setup the linear system
			Eigen::SparseMatrix<double> L = generate_L(N, _dt);
			Eigen::SparseMatrix<double> M = generate_M(M_bars, L);
			Eigen::VectorXd u = generate_u(u_bars, L);

			// solve the linear system
			bool success = false;
			Eigen::VectorXd c_opt = solve_c_opt(M, u, success);
			if (!success)
			{
				return std::make_pair(0., false);
			}

			// compute the final measure
			double g = generate_g(g_bars);
			double VDE = g - u.dot(c_opt);

			return std::make_pair(VDE, true);
		}
		else
		{
			return std::make_pair(0, false);
		}
	}
};
