import argparse
from sam_annotator import test_image_segment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "sam_annotator_cli.py")
    subs = parser.add_subparsers(dest = "option")

    image_segment_test = subs.add_parser("image_segment_test", help = "Function to test the segmentation of the sam model on an image")
    image_segment_test.add_argument("-image",  dest = "image", help = "Input image path", type = str, required = True)
    image_segment_test.add_argument("-output", dest = "output",  help = "Output image path", type = str, required = True)
    image_segment_test.add_argument("-model_type",  dest = "model_type", help = "Type of model to use (vit_h, vit_l, vit_b)", type = str, required = False, default = "vit_h")
    image_segment_test.add_argument("-pth_file", dest = "pth_file",  help = "Path of the .pth file corresponding to the type of model chosen",  type = str, required = False, default = "sam_vit_h_4b8939.pth")
    image_segment_test.add_argument("-onnx_file", dest = "onnx_file",  help = "Path of the .onnx file corresponding to the type of model chosen", type = str, required = False, default = None)
    image_segment_test.add_argument("-device", dest = "device", help = "Device to run the model ('cpu' or 'cuda')", type = str, required = False, default = "cuda")
    image_segment_test.add_argument("-scale_percent", dest = "scale_percent",  help = "Image downscaling percentage", type = int, required = False, default = None)
    image_segment_test.add_argument("--export_binary_mask",  dest = "export_binary_mask", help = "Flag if you want to export the generated mask as binary", action = "store_true")
    image_segment_test.add_argument("-save_embedding_path",  dest = "save_embedding_path", help = "Path to save the embedding", type = str, required = False, default = None)
    image_segment_test.add_argument("-input_embedding_path", dest = "input_embedding_path",  help = "File path with embedding", type = str, required = False, default = None)

    args = parser.parse_args()
    option = str(args.option)
    params = vars(args)
    del params["option"]
    eval(f"test_image_segment.{option}(**params)")