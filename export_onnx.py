import onnx
import torch
from effnetv2 import effnetv2_s
from onnxsim import simplify
import argparse
import torch.nn as nn


def main(model_path, output_path, input_shape=(224, 224), batch_size=1):
    model = effnetv2_s(num_classes=2)
    checkpoint = torch.load(model_path)['state_dict']
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()} 
    model.load_state_dict(checkpoint)
    dummy_input = torch.autograd.Variable(torch.randn(batch_size, 3, input_shape[0], input_shape[1]))
    torch.onnx.export(model, dummy_input, output_path, verbose=True, keep_initializers_as_inputs=True, opset_version=12)
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, default='./model_best.pth.tar', help='weights file path')
    parser.add_argument('--output_file', type=str, default='./effnetv2.onnx', help='onnx file path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    main(opt.weights_file, opt.output_file, input_shape=opt.img_size, batch_size=opt.batch_size)
