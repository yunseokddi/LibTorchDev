import torch
import torchvision
import argparse


def trace_weight(path):
    # declare ResNet152 pretrained model
    model = torchvision.models.resnet152(pretrained=True)
    save_traced_weight_path = './traced_resnet152_model.pt'

    # create the traced weight
    example = torch.rand(1, 3, 224, 224)

    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(save_traced_weight_path)
    print('create traced weight')
    print('path: {}'.format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_traced_weight_path', type=str,
                        help='enter save traced weight path including weight name',
                        default='./traced_resnet152_model.pt')
    args = parser.parse_args()

    trace_weight(args.save_traced_weight_path)
