import argparse
import torch
import torch.nn as nn
from train_mlp import get_data_loaders, test, MLP

def main():
    parser = argparse.ArgumentParser(description="Apply dynamic PTQ to MLP and evaluate.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to FP32 model state_dict')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for eval')
    parser.add_argument('--save-path',  type=str,   default='../../saved_models/quant_mnist_model_int8.pth', help='Path to save the trained model')
    parser.add_argument('--data-dir',   type=str, default='../../data', help='MNIST data directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model FP32
    input_size= 28 * 28
    output_size = 10
    model_fp32 = MLP(input_size, output_size).to(device)
    state = torch.load(args.model_path, map_location=device)
    model_fp32.load_state_dict(state)

    # evaluate FP32 accuracy
    _ ,_ , test_loader = get_data_loaders(batch_size = args.batch_size, data_dir = args.data_dir)
    acc_fp32 = test(model_fp32, test_loader, device)
    print(f"FP32 Test Accuracy: {acc_fp32:.4f}")

    # quantized model
    model_quant = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    model_quant.to(device)

    # save quantized model
    torch.save(model_quant.state_dict(), args.save_path)

    # evaluate quantized model
    acc_quant = test(model_quant, test_loader, device)
    print(f"Dynamic Quantized Test Accuracy: {acc_quant:.4f}")


if __name__=='__main__':
    main()