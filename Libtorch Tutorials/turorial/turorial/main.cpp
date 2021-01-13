#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <string>

#include "convert.h"

using namespace std;

//void print_script_module(const torch::jit::script::Module& module, size_t spaces = 0);

int main()
{
	//// ================================================================ //
	////                     BASIC AUTOGRAD EXAMPLE 1                     //
	//// ================================================================ //

	//tensor »ý¼º
	//torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
	//torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
	//torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

	//auto y = w * x + b;

	////gradient °è»ê
	//y.backward();

	////gradient Ãâ·Â
	//cout << x.grad() << endl;
	//cout << w.grad() << endl;
	//cout << b.grad() << endl;

	// ================================================================ //
	//                     BASIC AUTOGRAD EXAMPLE 2                     //
	// ================================================================ //

	//x = torch::randn({ 10,3 });
	//y = torch::randn({ 10,2 });

	////fc °è»ê
	//torch::nn::Linear linear(3, 2);
	//cout << "w:\n" << linear->weight << endl;
	//cout << "b:\n" << linear->bias << endl;

	////loss function, optimizer »ý¼º
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

	////array to tesor º¯È¯
	//float data_array[] = { 1,2,3,4 };
	//torch::Tensor t1 = torch::from_blob(data_array, { 2,2 });
	//cout << "Tensor array:\n" << t1 << endl;

	//TORCH_CHECK(data_array == t1.data_ptr<float>());

	////vector to tensor º¯È¯
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

	//// single element ÃßÃâ
	//cout << "\"s[0,2]\" as tensor:\n" << s.index({ 0, 2 }) << '\n';
	//cout << "\"s[0,2]\" as value:\n" << s.index({ 0, 2 }).item<int64_t>() << '\n';

	//// slicing
	//// == s[:, 2]
	//std::cout << "\"s[:,2]\":\n" << s.index({ Slice(), 2 }) << endl;

	//// ÇØ´ç dimension Á¦¿ÜÇÏ°í slicing Ãâ·Â
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

	//const string MNIST_data_path = "C:/Users/¿ÀÀ±¼®/Desktop/data/train-images.idx3-ubyte";

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

	//const string pretrained_model_path = "C:/Users/¿ÀÀ±¼®/Desktop/traced_resnet151_model.pt";

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

	//torch::nn::Sequential model
	//{
	//	torch::nn::Conv2d(torch::nn::Conv2dOptions(1,16,3).stride(2).padding(1)),
	//	torch::nn::ReLU()
	//};

	//const string model_save_path = "C:/Users/¿ÀÀ±¼®/Desktop/weight/model.pt";

	//torch::save(model, model_save_path);

	//cout << "Saved model:\n" << model << endl;

	//torch::load(model, model_save_path);

	//cout << "Loaded model:\n" << model;

	// =============================================================== //
	//                      Linear Regression	                       //
	// =============================================================== //


	//cout << "Linear Regression\n\n";

	//const int64_t input_size = 1;
	//const int64_t output_size = 1;
	//const size_t  num_epochs = 60;
	//const double learning_rate = 0.001;

	//auto x_train = torch::randint(0, 10, { 15,1 });
	//auto y_train = torch::randint(0, 10, { 15,1 });

	//torch::nn::Linear model(input_size, output_size);

	//torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

	//cout << fixed << setprecision(4);

	//cout << "Training..." << endl;

	//for (size_t epoch = 0; epoch != num_epochs; ++epoch)
	//{
	//	auto output = model(x_train);
	//	auto loss = torch::nn::functional::mse_loss(output, y_train);

	//	optimizer.zero_grad();
	//	loss.backward();
	//	optimizer.step();

	//	if ((epoch + 1) % 5 == 0)
	//	{
	//		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs <<
	//			"], Loss: " << loss.item<double>() << "\n";

	//	}
	//}

	//cout << "Training Finished!" << endl;


	// =============================================================== //
	//					 Logistic Regression					       //
	// =============================================================== //

	//std::cout << "Logistic Regression\n\n";

	//// Device
	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	//// Hyper parameters
	//const int64_t input_size = 784;
	//const int64_t num_classes = 10;
	//const int64_t batch_size = 100;
	//const size_t num_epochs = 5;
	//const double learning_rate = 0.001;

	//// *****change mnist file's name!!!!!*****
	//const std::string MNIST_data_path = "./mnist";

	//// MNIST Dataset (images and labels)
	//auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
	//	.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	//	.map(torch::data::transforms::Stack<>());

	//// Number of samples in the training set
	//auto num_train_samples = train_dataset.size().value();

	//auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
	//	.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	//	.map(torch::data::transforms::Stack<>());

	//// Number of samples in the testset
	//auto num_test_samples = test_dataset.size().value();

	//// Data loaders
	//auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	//	std::move(train_dataset), batch_size);

	//auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	//	std::move(test_dataset), batch_size);

	//// Logistic regression model
	//torch::nn::Linear model(input_size, num_classes);

	//model->to(device);

	//// Loss and optimizer
	//torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

	//// Set floating point output precision
	//std::cout << std::fixed << std::setprecision(4);

	//std::cout << "Training...\n";

	//for (size_t epoch = 0; epoch != num_epochs; ++epoch)
	//{
	//	double running_loss = 0.0;
	//	size_t num_correct = 0;

	//	for (auto& batch : *train_loader)
	//	{
	//		auto data = batch.data.view({ batch_size, -1 }).to(device);
	//		auto target = batch.target.to(device);

	//		auto output = model->forward(data);

	//		auto loss = torch::nn::functional::cross_entropy(output, target);

	//		running_loss += loss.item<double>() * data.size(0);

	//		auto prediction = output.argmax(1);

	//		num_correct += prediction.eq(target).sum().item<int64_t>();

	//		optimizer.zero_grad();
	//		loss.backward();
	//		optimizer.step();
	//	}

	//	auto sample_mean_loss = running_loss / num_train_samples;
	//	auto accuracy = static_cast<double>(num_correct) / num_train_samples;

	//	std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
	//		<< sample_mean_loss << ", Accuracy: " << accuracy << '\n';

	//	std::cout << "Training finished!\n\n";
	//	std::cout << "Testing...\n";

	//}
	//model->eval();
	//torch::NoGradGuard no_grad;

	//double running_loss = 0.0;
	//size_t num_correct = 0;

	//for (const auto& batch : *test_loader)
	//{
	//	auto data = batch.data.view({ batch_size, -1 }).to(device);
	//	auto target = batch.target.to(device);

	//	auto output = model->forward(data);

	//	auto loss = torch::nn::functional::cross_entropy(output, target);

	//	running_loss += loss.item<double>() * data.size(0);

	//	auto prediction = output.argmax(1);

	//	num_correct += prediction.eq(target).sum().item<int64_t>();
	//}

	//cout << "Testing Finished!" << endl;

	//auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
	//auto test_sample_mean_loss = running_loss / num_test_samples;

	//cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

	// =============================================================== //
	//                 convolution neural network					   //
	// =============================================================== //

	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

	cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	const int64_t num_classes = 10;
	const int64_t batch_size = 100;
	const size_t num_epochs = 5;
	const double learning_rate = 0.001;

	const string MNIST_data_path = "./mnist";

	auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto num_train_samples = train_dataset.size().value();

	auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	auto num_test_samples = test_dataset.size().value();

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(train_dataset), batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(move(test_dataset), batch_size);

	ConvNet model(num_classes);
	model->to(device);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

	cout << fixed << setprecision(4);

	cout << "Training..." << endl;

	for (size_t epoch = 0; epoch != num_epochs; ++epoch)
	{
		double running_loss = 0.0;
		size_t num_correct = 0;

		for (auto& batch : *train_loader)
		{
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			auto output = model->forward(data);

			auto loss = torch::nn::functional::cross_entropy(output, target);

			running_loss += loss.item<double>() * data.size(0);

			auto prediction = output.argmax(1);

			num_correct += prediction.eq(target).sum().item<int64_t>();

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		auto sample_mean_loss = running_loss / num_train_samples;
		auto accuracy = static_cast<double>(num_correct) / num_train_samples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << '\n';
	}

	std::cout << "Training finished!\n\n";
	std::cout << "Testing...\n";

	// Test the model
	model->eval();
	torch::NoGradGuard no_grad;

	double running_loss = 0.0;
	size_t num_correct = 0;

	for (const auto& batch : *test_loader) {
		auto data = batch.data.to(device);
		auto target = batch.target.to(device);

		auto output = model->forward(data);

		auto loss = torch::nn::functional::cross_entropy(output, target);
		running_loss += loss.item<double>() * data.size(0);

		auto prediction = output.argmax(1);
		num_correct += prediction.eq(target).sum().item<int64_t>();
	}

	std::cout << "Testing finished!\n";

	auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
	auto test_sample_mean_loss = running_loss / num_test_samples;

	std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

	return 0;

}

//void print_script_module(const torch::jit::script::Module& module, size_t spaces)
//{
//	for (const auto& sub_module : module.named_children())
//	{
//		if (!sub_module.name.empty())
//		{
//			cout << string(spaces, ' ') << sub_module.value.type()->name().value().name() << " " << sub_module.name << endl;
//		}
//
//		print_script_module(sub_module.value, spaces + 2);
//	}
//}
