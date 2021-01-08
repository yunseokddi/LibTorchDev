#include <torch/script.h>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

#define INPUTSIZE 224

using namespace std;

int main()
{	
	// change your weight path
	const string pretrained_model_path = "C:/Users/ø¿¿±ºÆ/Desktop/weight/traced_resnet152_model.pt";

	torch::jit::script::Module resnet152;

	try
	{
		resnet152 = torch::jit::load(pretrained_model_path);
	}

	catch (const torch::Error& error)
	{
		cerr << "check your weight path" << endl;

		return -1;
	}



	cv::Mat img;
	img = cv::imread("C:/Users/ø¿¿±ºÆ/Desktop/data/sample_image/cat.jpg");

	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	cv::Mat img_float;
	img.convertTo(img_float, 5, 1.0 / 255.0);
	cv::resize(img_float, img_float, cv::Size(INPUTSIZE, INPUTSIZE));

	torch::Tensor img_tensor = torch::from_blob(img_float.data, { 1, INPUTSIZE, INPUTSIZE, 3 }, torch::ScalarType::Float);
	//img_tensor = img_tensor.to(torch::kFloat32);
	img_tensor = img_tensor.permute({ 0,3,1,2 });

	vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_tensor);

	auto output = resnet152.forward(inputs).toTensor().clone().squeeze(0);

	torch::Tensor predict = torch::argmax(output);
	cout << torch::max(output) << endl;
	cout << predict.item() << endl;


	return 0;
}