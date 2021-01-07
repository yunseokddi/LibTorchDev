#include <torch/script.h>
#include <iostream>
#include <memory>

using namespace std;
using namespace torch;

int main()
{
	string WeightPath = "C:/Users/ø¿¿±ºÆ/Desktop/traced_resnet_model.pt";
	torch::jit::script::Module module;

	try
	{
		module = torch::jit::load(WeightPath);
	}

	catch (const c10::Error& e)
	{
		cerr << "error loading the model" << endl;
		return -1;
	}

	cout << "ok" << endl;

	vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({ 1,3,224,224 }));

	at::Tensor output = module.forward(inputs).toTensor();
	cout << output.slice(1, 0, 5) << endl;

	return 0;
}
