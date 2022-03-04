import argparse
import tensorrt as trt
import numpy as np

def main(args):
    logger = trt.Logger(trt.Logger.VERBOSE)
    with open(args.path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    print(f'path to the model: {args.path}')
    print(f'implicit batch dims: {engine.has_implicit_batch_dimension}')
    print(f'num opt profiles: {engine.num_optimization_profiles}')
    for binding in range(engine.num_bindings):
        name = engine.get_binding_name(binding)
        dims = engine.get_binding_shape(binding)
        dtype = engine.get_binding_dtype(binding)
        print(f'binding name: {name}, modified dims: {dims}, dtype: {dtype}, volume: {trt.volume(dims)}')
        print(f'binding desc: {engine.get_binding_format_desc(binding)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='path to TensorRT engine file')
    args = parser.parse_args()
    main(args)

