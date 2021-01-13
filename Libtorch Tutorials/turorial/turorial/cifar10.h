#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <string>

using namespace std;

class CIFAR10 :public torch::data::datasets::Dataset<CIFAR10>
{
public:
	enum Mode {kTrain, kTest};

	explicit CIFAR10(const string& root, Mode mode = Mode::kTrain);

	torch::data::Example<> get(size_t index) override;

	torch::optional<size_t> size() const override;

	bool is_train() const noexcept;

	const torch::Tensor& images() const;

	const torch::Tensor& targets() const;

private:
	torch::Tensor images_;
	torch::Tensor targets_;
	Mode mode_;

};