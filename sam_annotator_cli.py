import argparse
from sam_annotator import test_images_segment_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "sam_annotator_cli.py")
    subs = parser.add_subparsers(dest = "option")

    onnx_image_segment_test = subs.add_parser("onnx_image_segment_test", help = "Function to test the segmentation of the sam vit_h onnx model on an image")
    onnx_image_segment_test.add_argument("-image",  dest = "image", help = "Input image path", type = str, required = True)
    onnx_image_segment_test.add_argument("-output", dest = "output",  help = "Output image path", type = str, required = True)
    onnx_image_segment_test.add_argument("-model_type",  dest = "model_type", help = "Type of model to use (vit_h, vit_l, vit_b)", type = str, required = False, default="vit_h")
    onnx_image_segment_test.add_argument("-pth_file", dest = "pth_file",  help = "Path of the .pth file corresponding to the type of model chosen",  type = str, required = False, default="sam_vit_h_4b8939.pth")
    onnx_image_segment_test.add_argument("-onnx_file", dest = "onnx_file",  help = "Path of the .onnx file corresponding to the type of model chosen", type = str, required = False, default="sam_onnx_example.onnx")
    onnx_image_segment_test.add_argument("-device", dest = "device", help = "Device to run the model ('cpu' or 'cuda')", type = str, required = False, default="cuda")
    onnx_image_segment_test.add_argument("--export_binary_mask",  dest = "export_binary_mask", help = "Flag if you want to export the generated mask as binary", action = "store_true")

    args = parser.parse_args()
    option = str(args.option)
    params = vars(args)
    del params["option"]
    eval(f"test_images_segment_onnx.{option}(**params)")