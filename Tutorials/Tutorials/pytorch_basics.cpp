#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>

using namespace std;

void print_script_module(const torch::jit::script::Module& module, size_t spaces = 0);

int main()
{
	cout << "Pytorch Basics \n\n";

	//// ================================================================ //
	////                     BASIC AUTOGRAD EXAMPLE 1                     //
	//// ================================================================ //

	//tensor 생성
	//torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
	//torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
	//torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

	//auto y = w * x + b;

	////gradient 계산
	//y.backward();

	////gradient 출력
	//cout << x.grad() << endl;
	//cout << w.grad() << endl;
	//cout << b.grad() << endl;

	// ================================================================ //
	//                     BASIC AUTOGRAD EXAMPLE 2                     //
	// ================================================================ //

	//x = torch::randn({ 10,3 });
	//y = torch::randn({ 10,2 });

	////fc 계산
	//torch::nn::Linear linear(3, 2);
	//cout << "w:\n" << linear->weight << endl;
	//cout << "b:\n" << linear->bias << endl;

	////loss function, optimizer 생성
	//torch::nn::MSELoss criterion;
	//torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));

	//auto pred = linear->forward(x);

	//auto loss = criterion(pred, y);
	//cout << "Loss: " << loss.item<double>() << endl;

	//loss.backward();

	//std::cout << "dL/dw:\n" << linear->weight.grad() << '\n';
	//std::cout << "dL/db:\n" << linear->bias.grad() << '\n';

	//optimizer.step();

	//pred = linear->forward(x);
	//loss = criterion(pred, y);
	//std::cout << "Loss after 1 optimization step: " << loss.item<double>() << "\n\n";


	// =============================================================== //
	//               CREATING TENSORS FROM EXISTING DATA               //
	// =============================================================== //

	////array to tesor 변환
	//float data_array[] = { 1,2,3,4 };
	//torch::Tensor t1 = torch::from_blob(data_array, { 2,2 });
	//cout << "Tensor array:\n" << t1 << endl;

	//TORCH_CHECK(data_array == t1.data_ptr<float>());

	////vector to tensor 변환
	//vector<float> data_vector = { 1,2,3,4 };
	//torch::Tensor t2 = torch::from_blob(data_vector.data(), { 2,2 });
	//std::cout << "Tensor from vector:\n" << t2 << "\n\n";

	//TORCH_CHECK(data_vector.data() == t2.data_ptr<float>());

	//// =============================================================== //
	////             SLICING AND EXTRACTING PARTS FROM TENSORS           //
	//// =============================================================== //
	//vector<int64_t> test_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//torch::Tensor s = torch::from_blob(test_data.data(), { 3, 3 }, torch::kInt64);
	//cout << "s:\n" << s << endl;

	//using torch::indexing::Slice;
	//using torch::indexing::None;
	//using torch::indexing::Ellipsis;

	//// single element 추출
	//cout << "\"s[0,2]\" as tensor:\n" << s.index({ 0, 2 }) << '\n';
	//cout << "\"s[0,2]\" as value:\n" << s.index({ 0, 2 }).item<int64_t>() << '\n';

	//// slicing
	//// == s[:, 2]
	//std::cout << "\"s[:,2]\":\n" << s.index({ Slice(), 2 }) << endl;

	//// 해당 dimension 제외하고 slicing 출력
	//// == s[:2,:]
	//cout << "\"s[:2,:]\":\n" << s.index({ Slice(None, 2), Slice() }) << endl;

	//// == s[:,1:]
	//cout << "\"s[:,1:]\":\n" << s.index({ Slice(), Slice(1, None) }) << endl;

	//// == s[:,::2]
	//cout << "\"s[:,::2]\":\n" << s.index({ Slice(), Slice(None, None, 2) }) << endl;

	//// Combination
	//// == s[:2,1]
	//cout << "\"s[:2,1]\":\n" << s.index({ Slice(None, 2), 1 }) << endl;

	////Ellipsis
	//cout << "\"s[..., :2]\":\n" << s.index({ Ellipsis, Slice(None, 2) }) << endl;


	// =============================================================== //
	//                         INPUT PIPELINE                          //
	// =============================================================== //

	//const string MNIST_data_path = "C:/Users/오윤석/Desktop/data/train-images.idx3-ubyte";

	//auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
	//	.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	//	.map(torch::data::transforms::Stack<>());

	//auto example = dataset.get_batch(0);
	//cout << "Sample data size: ";
	//cout << example.data.sizes() << endl;
	//cout << "Sample target: " << example.target.item<int>() << endl;

	//auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, 64);

	//auto example_batch = *dataloader->begin();

	//std::cout << "Sample batch - data size: ";
	//std::cout << example_batch.data.sizes() << "\n";
	//std::cout << "Sample batch - target size: ";
	//std::cout << example_batch.target.sizes() << "\n\n";


	// =============================================================== //
	//               INPUT PIPELINE FOR CUSTOM DATASET                 //
	// =============================================================== //

	// See Deep Residual Network tutorial files cifar10.h and cifar10.cpp
	// for an example of a custom dataset implementation.

	// =============================================================== //
	//                        PRETRAINED MODEL                         //
	// =============================================================== //

	//const string pretrained_model_path = "C:/Users/오윤석/Desktop/traced_resnet151_model.pt";

	//torch::jit::script::Module resnet;

	//try
	//{
	//	resnet = torch::jit::load(pretrained_model_path);
	//}

	//catch (const torch::Error& error)
	//{
	//	std::cerr << "Could not load scriptmodule from file " << pretrained_model_path << ".\n"
	//		<< "You can create this file using the provided Python script 'create_resnet18_scriptmodule.py' "
	//		"in tutorials/basics/pytorch-basics/model/.\n";
	//	return -1;
	//}
	//
	//cout << "Resnet151 model: " << endl;

	//print_script_module(resnet, 2);

	//cout << endl;

	//const auto fc_weight = resnet.attr("fc").toModule().attr("weight").toTensor();

	//auto in_features = fc_weight.size(1);
	//auto out_features = fc_weight.size(0);

	//std::cout << "Fully connected layer: in_features=" << in_features << ", out_features=" << out_features << endl;

	//auto sample_input = torch::randn({ 1,3,224,224 });
	//vector<torch::jit::IValue> inputs{ sample_input };

	//cout << "Input size: ";
	//cout << sample_input.sizes() << endl;
	//auto output = resnet.forward(inputs).toTensor();
	//cout << "Output size: ";
	//cout << output.sizes() << endl << endl;



	// =============================================================== //
	//                      SAVE AND LOAD A MODEL                      //
	// =============================================================== //

	torch::nn::Sequential model
	{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1,16,3).stride(2).padding(1)),
		torch::nn::ReLU()
	};

	const string model_save_path = "C:/Users/오윤석/Desktop/weight/model.pt";

	torch::save(model, model_save_path);

	cout << "Saved model:\n" << model << endl;

	torch::load(model, model_save_path);

	cout << "Loaded model:\n" << model;

	return 0;

}

void print_script_module(const torch::jit::script::Module& module, size_t spaces)
{
	for (const auto& sub_module : module.named_children())
	{
		if (!sub_module.name.empty())
		{
			cout << string(spaces, ' ') << sub_module.value.type()->name().value().name() << " " << sub_module.name << endl;
		}

		print_script_module(sub_module.value, spaces + 2);
	}
}