#pragma once

#include <Eigen/Eigen>

#include <vector>

/**
 * @brief Class that stores a field on a regular grid.
 * @tparam _Tvalue Value type that is stored on the grid. Must be an Eigen type.
 * @tparam _Tdimensions Number of dimensions of the grid.
*/
template<typename _Tvalue, int _Tdimensions>
class field
{
public:
	/**
	 * @brief Type of values stored on the grid.
	*/
	using value_type = _Tvalue;

	/**
	 * @brief Type of the linear index.
	*/
	using linear_index_type = Eigen::Index;

	/**
	 * @brief Type of the grid index.
	*/
	using grid_index_type = Eigen::Vector<int32_t, _Tdimensions>;

	/**
	 * @brief Type of the grid resolution.
	*/
	using resolution_type = Eigen::Vector<int32_t, _Tdimensions>;

	/**
	 * @brief Type of a coordinate in the domain.
	*/
	using coordinate_type = Eigen::Vector<double, _Tdimensions>;

	/**
	 * @brief Type of the domain.
	*/
	using domain_type = Eigen::AlignedBox<double, _Tdimensions>;

	/**
	 * @brief Number of dimensions.
	*/
	static const int dimensions = _Tdimensions;

	/**
	 * @brief Constructor. Does not initialize the field.
	*/
	explicit field()
		: resolution(resolution_type::Zero())
		, domain(domain_type())
	{
	}

	/**
	 * @brief Constructor. Allocates a field for the given size.
	 * @param _resolution Resolution of the grid.
	 * @param _domain Spatial extent of the domain.
	*/
	field(resolution_type _resolution, domain_type _domain)
		: resolution(_resolution)
		, domain(_domain)
	{
		this->data.resize(this->resolution.prod());
	}

	/**
	 * @brief Converts a linear index to a grid index.
	 * @param _linear_index Linear index to convert.
	 * @return Corresponding grid index.
	*/
	grid_index_type grid_index(linear_index_type _linear_index) const
	{
		grid_index_type result;
		linear_index_type stride = 1;
		for (int d = 0; d < dimensions - 1; ++d)
			stride *= this->resolution[d];

		linear_index_type t = _linear_index;
		for (int d = dimensions - 1; d >= 0; --d) {
			result[d] = (int)(t / stride);
			t = t % stride;
			if (d > 0)
				stride /= this->resolution[d - 1];
		}
		return result;
	}

	/**
	 * @brief Gets the linear array index based on a grid index.
	 * @param gridIndex Grid index to compute linear index for.
	 * @return Zero-based linear index.
	*/
	linear_index_type linear_index(const grid_index_type& _grid_index) const
	{
		linear_index_type stride = 1;
		linear_index_type linear_index = _grid_index[0];
		for (int64_t d = 1; d < dimensions; ++d) {
			stride *= this->resolution[d - 1];
			linear_index += _grid_index[d] * stride;
		}
		return linear_index;
	}

	/**
	 * @brief Gets the spatial coordinate of a grid index.
	 * @param _grid_index Grid index to get domain coordinate for.
	 * @return Domain coordinate of the grid index.
	*/
	coordinate_type coordinate(const grid_index_type& _grid_index) const
	{
		coordinate_type s;
		for (int i = 0; i < dimensions; ++i) {
			s[i] = this->resolution[i] < 2 ? 0.5 : _grid_index[i] / (this->resolution[i] - 1.);
		}
		return this->domain.min() + (this->domain.max() - this->domain.min()).cwiseProduct(s);
	}

	/**
	 * @brief Trilinearly interpolates a data value from the grid at a given domain coordinate.
	 * @param _coordinate Coordinate where to interpolate from. Assumes that the coordinate is contained in the domain.
	 * @return Trilinearly interpolated value.
	*/
	value_type value(const coordinate_type& _coordinate) const
	{
		coordinate_type vf_tex = (_coordinate - this->domain.min()).cwiseQuotient(this->domain.max() - this->domain.min());
		coordinate_type vf_sample = vf_tex.cwiseProduct(this->resolution.template cast<double>() - coordinate_type::Ones());

		grid_index_type vi_sample_base0, vi_sample_base1;
		for (int i = 0; i < dimensions; ++i) {
			vi_sample_base0[i] = std::min(std::max(0, (int)vf_sample[i]), this->resolution[i] - 1);
			vi_sample_base1[i] = std::min(std::max(0, vi_sample_base0[i] + 1), this->resolution[i] - 1);
		}
		coordinate_type vf_sample_interpol = vf_sample - vi_sample_base0.template cast<double>();

		int num_corners = (int)std::pow(2, dimensions);
		value_type result = value_type::Zero();
		for (int i = 0; i < num_corners; ++i) {
			double weight = 1;
			grid_index_type grid_index = grid_index_type::Zero();
			for (int d = 0; d < dimensions; ++d) {
				if (i & (int(1) << (dimensions - 1 - d))) {
					grid_index[d] = vi_sample_base1[d];
					weight *= vf_sample_interpol[d];
				}
				else {
					grid_index[d] = vi_sample_base0[d];
					weight *= 1 - vf_sample_interpol[d];
				}
			}
			result += this->value(grid_index) * weight;
		}
		return result;
	}

	/**
	 * @brief Gets the origin of the domain.
	 * @return Origin of domain.
	*/
	coordinate_type origin() const { return this->domain.min(); }

	/**
	 * @brief Gets the spacing between grid coordinates in the domain.
	 * @return Spacing of regular grid.
	*/
	coordinate_type spacing() const {
		coordinate_type div;
		for (int i = 0; i < dimensions; ++i) {
			div[i] = this->resolution[i] < 2 ? 0 : (this->resolution[i] - 1.);
		}
		return (this->domain.max() - this->domain.min()).cwiseQuotient(div);
	}

	/**
	 * @brief Gets the value at a certain grid index.
	 * @param _grid_index Grid index where to read the value from.
	 * @return Data value stored at grid index.
	*/
	const value_type& value(const grid_index_type& _grid_index) const {
		return this->data[this->linear_index(_grid_index)];
	}

	/**
	 * @brief Sets the value at a certain grid index.
	 * @param _grid_index Grid index where to write the value to.
	 * @param _value Data value to store at grid index.
	*/
	void set_value(const grid_index_type& _grid_index, const value_type& _value) {
		this->data[this->linear_index(_grid_index)] = _value;
	}

	/**
	 * @brief Resolution of the grid.
	*/
	resolution_type resolution;

	/**
	 * @brief Domain extent of the grid.
	*/
	domain_type domain;

	/**
	 * @brief Linear array of data values.
	*/
	std::vector<value_type> data;
};

/**
 * @brief Unsteady two-dimensional scalar field.
*/
using usf2d = field<Eigen::Vector<double, 1>, 3>;

/**
 * @brief Unsteady two-dimensional vector field.
*/
using uvf2d = field<Eigen::Vector2d, 3>;

/**
 * @brief Unsteady two-dimensional tensor field.
*/
using utf2d = field<Eigen::Matrix2d, 3>;

/**
 * @brief Steady two-dimensional scalar field.
*/
using sf2d = field<Eigen::Vector<double, 1>, 2>;

/**
 * @brief Helper class for the estimation of partial derivatives.
*/
class partial
{
public:
	/**
	 * @brief Estimates first partial derivatives using second-order central difference in interior and first-order forward/backward difference on boundary.
	 * @param _vv Velocity field to compute partials for.
	 * @param _vv_dx Field into which the x-partial is stored.
	 * @param _vv_dy Field into which the y-partial is stored.
	 * @param _vv_dt Field into which the t-partial is stored.
	*/
	static void estimate(const uvf2d& _vv, uvf2d& _vv_dx, uvf2d& _vv_dy, uvf2d& _vv_dt)
	{
		// allocate output fields
		_vv_dx.resolution = _vv.resolution;
		_vv_dy.resolution = _vv.resolution;
		_vv_dt.resolution = _vv.resolution;
		_vv_dx.domain = _vv.domain;
		_vv_dy.domain = _vv.domain;
		_vv_dt.domain = _vv.domain;
		_vv_dx.data.resize(_vv_dx.resolution.prod());
		_vv_dy.data.resize(_vv_dy.resolution.prod());
		_vv_dt.data.resize(_vv_dt.resolution.prod());

		// get resolution and spacing
		Eigen::Vector3i resolution = _vv.resolution;
		Eigen::Vector3d spacing = _vv.spacing();

#ifdef NDEBUG
#pragma omp parallel for
#endif
		for (int iz = 0; iz < resolution.z(); ++iz)
		{
			int iz0 = std::max(iz - 1, 0);
			int iz1 = std::min(iz + 1, resolution.z() - 1);
			for (int iy = 0; iy < resolution.y(); ++iy)
			{
				int iy0 = std::max(iy - 1, 0);
				int iy1 = std::min(iy + 1, resolution.y() - 1);
				for (int ix = 0; ix < resolution.x(); ++ix)
				{
					int ix0 = std::max(ix - 1, 0);
					int ix1 = std::min(ix + 1, resolution.x() - 1);
					_vv_dx.set_value(Eigen::Vector3i(ix, iy, iz),
						(_vv.value(Eigen::Vector3i(ix1, iy, iz)) - _vv.value(Eigen::Vector3i(ix0, iy, iz))) / ((ix1 - ix0) * spacing.x()));
					_vv_dy.set_value(Eigen::Vector3i(ix, iy, iz),
						(_vv.value(Eigen::Vector3i(ix, iy1, iz)) - _vv.value(Eigen::Vector3i(ix, iy0, iz))) / ((iy1 - iy0) * spacing.y()));
					_vv_dt.set_value(Eigen::Vector3i(ix, iy, iz),
						(_vv.value(Eigen::Vector3i(ix, iy, iz1)) - _vv.value(Eigen::Vector3i(ix, iy, iz0))) / ((iz1 - iz0) * spacing.z()));
				}
			}
		}
	}
};
