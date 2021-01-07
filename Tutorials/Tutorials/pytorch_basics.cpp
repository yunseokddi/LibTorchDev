#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>

using namespace std;

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

	//array to tesor 변환
	float data_array[] = { 1,2,3,4 };
	torch::Tensor t1 = torch::from_blob(data_array, { 2,2 });
	cout << "Tensor array:\n" << t1 << endl;

	TORCH_CHECK(data_array == t1.data_ptr<float>());

	//vector to tensor 변환
	vector<float> data_vector = { 1,2,3,4 };
	torch::Tensor t2 = torch::from_blob(data_vector.data(), { 2,2 });
	std::cout << "Tensor from vector:\n" << t2 << "\n\n";

	TORCH_CHECK(data_vector.data() == t2.data_ptr<float>());

	// =============================================================== //
	//             SLICING AND EXTRACTING PARTS FROM TENSORS           //
	// =============================================================== //
	vector<int64_t> test_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	torch::Tensor s = torch::from_blob(test_data.data(), { 3, 3 }, torch::kInt64);
	cout << "s:\n" << s << endl;

	using torch::indexing::Slice;
	using torch::indexing::None;
	using torch::indexing::Ellipsis;

	//single element 추출
	std::cout << "\"s[0,2]\" as tensor:\n" << s.index({ 0, 2 }) << '\n';
	std::cout << "\"s[0,2]\" as value:\n" << s.index({ 0, 2 }).item<int64_t>() << '\n';



	return 0;

}