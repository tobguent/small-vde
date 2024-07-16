#pragma once

#include "field.hpp"

#include <fstream>

class io
{
public:
	/**
	 * @brief Loads an amira file from file. Note that the file must store internally float values for an unsteady 2D vector field.
	 * @param path Path to the file to load.
	 * @param _vv Vector field that is written into.
	*/
	static void load_amira(const char* path, uvf2d& _vv)
	{
		// open the file
		FILE* fp = fopen(path, "rb");
		if (!fp)
		{
			throw std::runtime_error("File not found!");
		}

		// read header.
		char buffer[2048];
		fread(buffer, sizeof(char), 2047, fp);
		buffer[2047] = '\0';

		// correct format?
		if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
		{
			fclose(fp);
			throw std::runtime_error("Wrong file format.");
		}

		// read resolution
		int sscanf_result = sscanf(find_and_jump(buffer, "define Lattice"), "%d %d %d", &_vv.resolution.x(), &_vv.resolution.y(), &_vv.resolution.z());
		if (sscanf_result != 3)
		{
			fclose(fp);
			throw std::runtime_error("Wrong number of arguments read!");
		}
		_vv.data.resize(_vv.resolution.prod());

		// read domain
		Eigen::Vector3d min_corner, max_corner;
		sscanf_result = sscanf(find_and_jump(buffer, "BoundingBox"), "%lf %lf %lf %lf %lf %lf", &min_corner.x(), &max_corner.x(), &min_corner.y(), &max_corner.y(), &min_corner.z(), &max_corner.z());
		if (sscanf_result != 6)
		{
			fclose(fp);
			throw std::runtime_error("Wrong number of arguments read!");
		}
		_vv.domain = Eigen::AlignedBox3d(min_corner, max_corner);

		// read number of components
		int num_components = 0;
		if (sscanf(find_and_jump(buffer, "Lattice { float["), "%d", &num_components) != 1)
		{
			fclose(fp);
			throw std::runtime_error("Internal format not as expected");
		}

		// read data
		std::vector<float> raw_data(num_components * _vv.resolution.prod());
		const long idxStartData = (long)(strstr(buffer, "# Data section follows") - buffer);
		if (idxStartData > 0)
		{
			// set the file pointer to the beginning of "# Data section follows"
			fseek(fp, idxStartData, SEEK_SET);
			// consume this line, which is "# Data section follows"
			fgets(buffer, 2047, fp);
			// consume the next line, which is "@1"
			fgets(buffer, 2047, fp);

			size_t num_to_read = num_components * _vv.resolution.prod();
			const size_t actual_read = fread(raw_data.data(), sizeof(float), num_to_read, fp);
			if (num_to_read != actual_read) {
				fclose(fp);
				throw std::runtime_error("Premature end of file.");
			}
		}
		else
		{
			fclose(fp);
			throw std::runtime_error("Data section not found.");
		}
		fclose(fp);

		// copy the data into the output array.
		for (size_t id = 0; id < _vv.data.size(); ++id)
			_vv.data[id] = Eigen::Vector2d(raw_data[id * num_components + 0], raw_data[id * num_components + 1]);
	}

	/**
	 * @brief Writes a scalar field out as a bitmap file (linear gray scale).
	 * @param _path Path to the destination file.
	 * @param _field Scalar field to write out.
	 * @param _exponent Raises the VDE value to the power of the exponent before mapping to gray value.
	*/
	static void write_bmp(const char* _path, const sf2d& _field, double _exponent)
	{
		unsigned char file[14] = {
			'B', 'M',			// magic
			0, 0, 0, 0,			// size in bytes
			0, 0,				// app data
			0, 0,				// app data
			40 + 14, 0, 0, 0	// start of data offset
		};
		unsigned char info[40] = {
			40, 0, 0, 0,		// info hd size
			0, 0, 0, 0,			// width
			0, 0, 0, 0,			// heigth
			1, 0,				// number color planes
			24, 0,				// bits per pixel
			0, 0, 0, 0,			// compression is none
			0, 0, 0, 0,			// image bits size
			0x13, 0x0B, 0, 0,	// horz resoluition in pixel / m
			0x13, 0x0B, 0, 0,	// vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
			0, 0, 0, 0,			// #colors in pallete
			0, 0, 0, 0,			// #important colors
		};

		int w = _field.resolution.x();
		int h = _field.resolution.y();

		int padSize = (4 - (w * 3) % 4) % 4;
		int sizeData = w * h * 3 + h * padSize;
		int sizeAll = sizeData + sizeof(file) + sizeof(info);

		file[2] = (unsigned char)(sizeAll);
		file[3] = (unsigned char)(sizeAll >> 8);
		file[4] = (unsigned char)(sizeAll >> 16);
		file[5] = (unsigned char)(sizeAll >> 24);

		info[4] = (unsigned char)(w);
		info[5] = (unsigned char)(w >> 8);
		info[6] = (unsigned char)(w >> 16);
		info[7] = (unsigned char)(w >> 24);

		info[8] = (unsigned char)(h);
		info[9] = (unsigned char)(h >> 8);
		info[10] = (unsigned char)(h >> 16);
		info[11] = (unsigned char)(h >> 24);

		info[20] = (unsigned char)(sizeData);
		info[21] = (unsigned char)(sizeData >> 8);
		info[22] = (unsigned char)(sizeData >> 16);
		info[23] = (unsigned char)(sizeData >> 24);

		std::ofstream stream(_path, std::ios::binary | std::ios::out);
		stream.write((char*)file, sizeof(file));
		stream.write((char*)info, sizeof(info));

		unsigned char pad[3] = { 0, 0, 0 };

		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				double gray = _field.value(Eigen::Vector2i(x, y)).x();

				gray = std::pow(gray, _exponent);	// apply transfer function

				unsigned char pixel[3] = { 0 };
				pixel[0] = (unsigned char)(std::min(std::max(0., gray), 1.) * 255);
				pixel[1] = (unsigned char)(std::min(std::max(0., gray), 1.) * 255);
				pixel[2] = (unsigned char)(std::min(std::max(0., gray), 1.) * 255);

				stream.write((char*)pixel, 3);
			}
			stream.write((char*)pad, padSize);
		}
	}

private:
	/**
	 * @brief Helper function that finds a seach string inside a buffer
	 * @param buffer Buffer to search substring in.
	 * @param search_string String to search.
	 * @return Offset to the found location.
	*/
	static const char* find_and_jump(const char* _buffer, const char* _search_string) {
		const char* found_loc = strstr(_buffer, _search_string);
		if (found_loc) return found_loc + strlen(_search_string);
		return _buffer;
	}
};
