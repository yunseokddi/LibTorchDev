#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define IMAGE_SIZE 224
#define CHANNELS 3

using namespace std;

bool LoadImage(std::string file_name, cv::Mat& image)
{
	image = cv::imread(file_name);

	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// scale image to fit
	cv::Size scale(IMAGE_SIZE, IMAGE_SIZE);
	cv::resize(image, image, scale);

	// convert [unsigned int] to [float]
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

	return true;
}

bool LoadImageNetLabel(std::string file_name, std::vector<std::string>& labels)
{
	std::ifstream ifs(file_name);
	if (!ifs)
	{
		return false;
	}
	std::string line;
	while (std::getline(ifs, line))
	{
		labels.push_back(line);
	}
	return true;
}

int main() {
	// change your weight, label, img paths
	const string weight_path = "../traced_resnet18.pt";
	const string label_path = "../label.txt";
	const string img_path = "../dog.jpg";

	// load the pretrained model
	torch::jit::script::Module module = torch::jit::load(weight_path);

	std::vector<std::string> labels;
	
	// mapping from label.txt to idx
	LoadImageNetLabel(label_path, labels);

	cv::Mat image;

	if (LoadImage(img_path, image))
	{
		// Mat to tensor
		auto input_tensor = torch::from_blob(image.data, { 1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

		// model forward
		torch::Tensor out_tensor = module.forward({ input_tensor }).toTensor();

		auto results = out_tensor.sort(-1, true);
		auto softmaxs = std::get<0>(results)[0].softmax(0);
		auto indexs = std::get<1>(results)[0];


		auto idx = indexs[0].item<int>();
		std::cout << "Label:  " << labels[idx] << std::endl;
		std::cout << "Probability:  " << softmaxs[0].item<float>() * 100.0f << "%" << std::endl;



	}
	return 0;
}