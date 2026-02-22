#include "io/image_parser.h"

#include <exception>
#include <fstream>
#include <iostream>

BitmapImage ImageParser::read_bitmap(const std::filesystem::path& file_path) {
	if (!std::filesystem::exists(file_path)) {
		throw std::exception{};
	}

	if (!std::filesystem::is_regular_file(file_path)) {
		throw std::exception{};
	}

	if (file_path.extension() != ".bmp") {
		throw std::exception{};
	}

	auto file_reader = std::ifstream{ file_path , std::ios::binary | std::ios::in };

	const auto read_from_file = [&file_reader](auto val) {
		file_reader.read(reinterpret_cast<char*>(&val), sizeof(val));
		if (!file_reader) {
			throw std::exception{};
		}

		return val;
		};

	auto bfType = std::uint16_t{};
	auto bfSize = std::uint32_t{};
	auto bfReserved = std::uint32_t{};
	auto bfOffBits = std::uint32_t{};

	bfType = read_from_file(bfType);
	bfSize = read_from_file(bfSize);
	bfReserved = read_from_file(bfReserved);
	bfOffBits = read_from_file(bfOffBits);

	auto biSize = std::uint32_t{};
	auto biWidth = std::int32_t{};
	auto biHeight = std::int32_t{};
	auto biPlanes = std::uint16_t{};
	auto biBitCount = std::uint16_t{};
	auto biCompression = std::uint32_t{};
	auto biSizeImage = std::uint32_t{};
	auto biXPelsPerMeter = std::int32_t{};
	auto biYPelsPerMeter = std::int32_t{};
	auto biClrUsed = std::uint32_t{};
	auto biClrImportant = std::uint32_t{};

	biSize = read_from_file(biSize);
	biWidth = read_from_file(biWidth);
	biHeight = read_from_file(biHeight);
	biPlanes = read_from_file(biPlanes);
	biBitCount = read_from_file(biBitCount);
	biCompression = read_from_file(biCompression);
	biSizeImage = read_from_file(biSizeImage);
	biXPelsPerMeter = read_from_file(biXPelsPerMeter);
	biYPelsPerMeter = read_from_file(biYPelsPerMeter);
	biClrUsed = read_from_file(biClrUsed);
	biClrImportant = read_from_file(biClrImportant);

	auto bitmap_buffer = std::vector<char>{};
	bitmap_buffer.resize(biSizeImage);

	file_reader.clear();
	file_reader.seekg(bfOffBits);
	file_reader.read(bitmap_buffer.data(), biSizeImage);

	const auto bitmap_height = static_cast<std::uint32_t>(std::abs(biHeight));
	const auto bitmap_width = static_cast<std::uint32_t>(std::abs(biWidth));

	auto bitmap = BitmapImage{ bitmap_height, bitmap_width };

	for (auto y = std::uint32_t(0); y < bitmap_height; y++) {
		for (auto x = std::uint32_t(0); x < bitmap_width; x++) {
			const auto index = y * bitmap_width * 3 + x * 3;

			const auto blue = static_cast<BitmapImage::BitmapPixel::value_type>(bitmap_buffer[index]);
			const auto green = static_cast<BitmapImage::BitmapPixel::value_type>(bitmap_buffer[index + 1]);
			const auto red = static_cast<BitmapImage::BitmapPixel::value_type>(bitmap_buffer[index + 2]);

			const auto pixel = BitmapImage::BitmapPixel{ red , green, blue };

			bitmap.set_pixel(y, x, pixel);
		}
	}

	return bitmap;
}

void ImageParser::write_bitmap(const std::filesystem::path& file_path, const BitmapImage& bitmap) {
	auto file_writer = std::ofstream{ file_path , std::ios::out | std::ios::binary };

	const auto write_to_file = [&file_writer](const auto value) {
		file_writer.write(reinterpret_cast<const char*>(&value), sizeof(value));
		};

	auto bfType = std::uint16_t{ 19778 };
	auto bfSize = std::uint32_t{ 3 * bitmap.get_height() * bitmap.get_width() };
	auto bfReserved = std::uint32_t{ 0 };
	auto bfOffBits = std::uint32_t{ 54 };

	write_to_file(bfType);
	write_to_file(bfSize);
	write_to_file(bfReserved);
	write_to_file(bfOffBits);

	auto biSize = std::uint32_t{ 40 };
	auto biWidth = std::int32_t{ static_cast<std::int32_t>(bitmap.get_width()) };
	auto biHeight = std::int32_t{ static_cast<std::int32_t>(bitmap.get_height()) };
	auto biPlanes = std::uint16_t{ 1 };
	auto biBitCount = std::uint16_t{ 24 };
	auto biCompression = std::uint32_t{ 0 };
	auto biSizeImage = std::uint32_t{ 3 * bitmap.get_height() * bitmap.get_width() };
	auto biXPelsPerMeter = std::int32_t{ 0 };
	auto biYPelsPerMeter = std::int32_t{ 0 };
	auto biClrUsed = std::uint32_t{ 0 };
	auto biClrImportant = std::uint32_t{ 0 };

	write_to_file(biSize);
	write_to_file(biWidth);
	write_to_file(biHeight);
	write_to_file(biPlanes);
	write_to_file(biBitCount);
	write_to_file(biCompression);
	write_to_file(biSizeImage);
	write_to_file(biXPelsPerMeter);
	write_to_file(biYPelsPerMeter);
	write_to_file(biClrUsed);
	write_to_file(biClrImportant);

	for (auto y = std::uint32_t(0); y < bitmap.get_height(); y++) {
		for (auto x = std::uint32_t(0); x < bitmap.get_width(); x++) {
			const auto pixel = bitmap.get_pixel(y, x);

			const auto red = pixel.get_red_channel();
			const auto green = pixel.get_green_channel();
			const auto blue = pixel.get_blue_channel();

			write_to_file(blue);
			write_to_file(green);
			write_to_file(red);
		}
	}
}

void ImageParser::write_grayscale_image(const std::filesystem::path& file_path, const GrayscaleImage& grayscale){
	auto file_writer = std::ofstream{ file_path , std::ios::out | std::ios::binary };

	const auto write_to_file = [&file_writer](const auto value) {
		file_writer.write(reinterpret_cast<const char*>(&value), sizeof(value));
		};

	// check for necessary padding
	if((3*grayscale.width % 4) != 0){
		// padding necessary
		// determine width of necessary padding
		std::uint32_t padding_pixels;
		for(padding_pixels = (std::uint32_t) 1; padding_pixels < (std::uint32_t) 4; ++padding_pixels){
			if(((3*(grayscale.width+padding_pixels))%4) == 0){
				break;
			}
		}
		std::cout << "ImageParser::write_grayscale_image: adding padding of width: " << padding_pixels << std::endl;

		// create a padded copy of grayscale due to the constant nature
		GrayscaleImage padded_grayscale;
		padded_grayscale.height = grayscale.height;
		padded_grayscale.width = grayscale.width + padding_pixels;
		padded_grayscale.pixels.resize(grayscale.height * (grayscale.width + padding_pixels));
		for(std::uint32_t i = 0; i < grayscale.height; ++i){
			for(std::uint32_t j = 0; j < grayscale.width; ++j){
				padded_grayscale.pixels[i*padded_grayscale.width + j] = grayscale.pixels[i*grayscale.width+j];
			}
			// add padding
			for(std::uint32_t x = 1; x <= padding_pixels; ++x){
				padded_grayscale.pixels[i*padded_grayscale.width + grayscale.width - 1 + x] = 0;
			}
			
		}
		
		auto bfType = std::uint16_t{ 19778 };
		auto bfSize = std::uint32_t{ 3 * padded_grayscale.height * padded_grayscale.width };
		auto bfReserved = std::uint32_t{ 0 };
		auto bfOffBits = std::uint32_t{ 54 };

		write_to_file(bfType);
		write_to_file(bfSize);
		write_to_file(bfReserved);
		write_to_file(bfOffBits);

		auto biSize = std::uint32_t{ 40 };
		auto biWidth = std::int32_t{ static_cast<std::int32_t>(padded_grayscale.width) };
		auto biHeight = std::int32_t{ static_cast<std::int32_t>(padded_grayscale.height) };
		auto biPlanes = std::uint16_t{ 1 };
		auto biBitCount = std::uint16_t{ 24 };
		auto biCompression = std::uint32_t{ 0 };
		auto biSizeImage = std::uint32_t{ 3* padded_grayscale.height * padded_grayscale.width };
		auto biXPelsPerMeter = std::int32_t{ 0 };
		auto biYPelsPerMeter = std::int32_t{ 0 };
		auto biClrUsed = std::uint32_t{ 0 };
		auto biClrImportant = std::uint32_t{ 0 };

		write_to_file(biSize);
		write_to_file(biWidth);
		write_to_file(biHeight);
		write_to_file(biPlanes);
		write_to_file(biBitCount);
		write_to_file(biCompression);
		write_to_file(biSizeImage);
		write_to_file(biXPelsPerMeter);
		write_to_file(biYPelsPerMeter);
		write_to_file(biClrUsed);
		write_to_file(biClrImportant);

		for (auto y = std::uint32_t(0); y < padded_grayscale.height; y++) {
			for (auto x = std::uint32_t(0); x < padded_grayscale.width; x++) {
				const auto grayscale_value = padded_grayscale.pixels[y*padded_grayscale.width + x];

				const auto red = grayscale_value;
				const auto green = grayscale_value;
				const auto blue = grayscale_value;

				write_to_file(blue);
				write_to_file(green);
				write_to_file(red);
			}
		}

	}	// padding necessary
	else{
		// no padding necessary
		auto bfType = std::uint16_t{ 19778 };
		auto bfSize = std::uint32_t{ 3 * grayscale.height * grayscale.width };
		auto bfReserved = std::uint32_t{ 0 };
		auto bfOffBits = std::uint32_t{ 54 };

		write_to_file(bfType);
		write_to_file(bfSize);
		write_to_file(bfReserved);
		write_to_file(bfOffBits);

		auto biSize = std::uint32_t{ 40 };
		auto biWidth = std::int32_t{ static_cast<std::int32_t>(grayscale.width) };
		auto biHeight = std::int32_t{ static_cast<std::int32_t>(grayscale.height) };
		auto biPlanes = std::uint16_t{ 1 };
		auto biBitCount = std::uint16_t{ 24 };
		auto biCompression = std::uint32_t{ 0 };
		auto biSizeImage = std::uint32_t{ 3* grayscale.height * grayscale.width };
		auto biXPelsPerMeter = std::int32_t{ 0 };
		auto biYPelsPerMeter = std::int32_t{ 0 };
		auto biClrUsed = std::uint32_t{ 0 };
		auto biClrImportant = std::uint32_t{ 0 };

		write_to_file(biSize);
		write_to_file(biWidth);
		write_to_file(biHeight);
		write_to_file(biPlanes);
		write_to_file(biBitCount);
		write_to_file(biCompression);
		write_to_file(biSizeImage);
		write_to_file(biXPelsPerMeter);
		write_to_file(biYPelsPerMeter);
		write_to_file(biClrUsed);
		write_to_file(biClrImportant);

		for (auto y = std::uint32_t(0); y < grayscale.height; y++) {
			for (auto x = std::uint32_t(0); x < grayscale.width; x++) {
				const auto grayscale_value = grayscale.pixels[y*grayscale.width + x];

				const auto red = grayscale_value;
				const auto green = grayscale_value;
				const auto blue = grayscale_value;

				write_to_file(blue);
				write_to_file(green);
				write_to_file(red);
			}
		}
	}  // no padding necessary
}