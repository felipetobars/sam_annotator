import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime

def draw_points(input_points,input_labels,img_masked,img):
    for i,point in enumerate(input_points):
        x,y=point
        if input_labels[i] == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if img_masked is not None:
            cv2.circle(img_masked, (x, y), 5, color, -1)
        else:
            cv2.circle(img, (x, y), 5, color, -1)
    if img_masked is not None:
        cv2.imshow('Image', img_masked)
    else:
        cv2.imshow('Image', img)

def create_image_masked(mask,  img, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        color = np.array([3, 44, 155])
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1)).astype(np.uint8)
    #print(mask_image.shape, type(mask_image))
    #print(img.shape, type(img))
    new_mask_image = cv2.cvtColor(mask_image,cv2.COLOR_RGB2BGR)
    new_image=cv2.addWeighted(src1=img,alpha=1,src2=new_mask_image,beta=0.8,gamma=0)
    return new_image, new_mask_image

def predict_mask(ort_inputs):
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    return masks

def mouse_callback(event, x, y, flags, param):
    global mask_image
    global img_masked
    global input_points
    global input_labels
    global onnx_mask_input
    global onnx_has_mask_input
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        onnx_points_input = np.concatenate([np.array(input_points), np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_points_input = predictor.transform.apply_coords(onnx_points_input, img.shape[:2]).astype(np.float32)
        #print("Coords:", x, y)
        if flags == cv2.EVENT_FLAG_LBUTTON:
            input_labels.append(1)
        else:
            input_labels.append(0)
        onnx_labels_input = np.concatenate([np.array(input_labels), np.array([-1])], axis=0)[None, :].astype(np.float32)
        ort_inputs = {"image_embeddings": image_embedding,"point_coords": onnx_points_input,"point_labels": onnx_labels_input,"mask_input": onnx_mask_input,"has_mask_input": onnx_has_mask_input,"orig_im_size": np.array(img.shape[:2], dtype=np.float32)}
        masks = predict_mask(ort_inputs)
        img_masked, mask_image = create_image_masked(masks, img)
        draw_points(input_points,input_labels,img_masked,img)
    elif event == cv2.EVENT_MOUSEMOVE:
        new_input_points = input_points.copy()
        new_input_labels = input_labels.copy()
        new_input_points.append([x,y])
        onnx_points_input = np.concatenate([np.array(new_input_points), np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_points_input = predictor.transform.apply_coords(onnx_points_input, img.shape[:2]).astype(np.float32)
        if flags == (cv2.EVENT_MOUSEMOVE + cv2.EVENT_FLAG_CTRLKEY):
            new_input_labels.append(0)
        else:
            new_input_labels.append(1)
        onnx_labels_input = np.concatenate([np.array(new_input_labels), np.array([-1])], axis=0)[None, :].astype(np.float32)
        ort_inputs = {"image_embeddings": image_embedding,"point_coords": onnx_points_input,"point_labels": onnx_labels_input,"mask_input": onnx_mask_input,"has_mask_input": onnx_has_mask_input,"orig_im_size": np.array(img.shape[:2], dtype=np.float32)}
        masks = predict_mask(ort_inputs)
        img_masked_prev, mask_image_prev = create_image_masked(masks, img)
        draw_points(new_input_points,new_input_labels,img_masked_prev,img)

def combine_mask(complete_class_mask, mask_image):
    x1,y1 = np.where(np.all(complete_class_mask == [155, 44, 3], axis=-1))
    x2,y2 = np.where(np.all(mask_image == [155, 44, 3], axis=-1))
    combine_mask = np.zeros_like(mask_image)
    combine_mask[x1,y1] =[155, 44, 3]
    combine_mask[x2,y2] = [155, 44, 3]
    return combine_mask

def onnx_image_segment_test(image, output, model_type = "vit_h", pth_file = "sam_vit_h_4b8939.pth", onnx_file = "sam_onnx_example.onnx", device = "cuda", export_binary_mask = False):
    """
    Function to test the segmentation of the sam vit_h onnx model on an image
    Parameters:
        image (str): Input image path
        output (str): Output image path
        model_type (str): Type of model to use (vit_h, vit_l, vit_b)
        pth_file (str): Path of the .pth file corresponding to the type of model chosen
        onnx_file (str): Path of the .onnx file corresponding to the type of model chosen
        device (str): Device to run the model ('cpu' or 'cuda')
        export_binary_mask (bool): Flag if you want to export the generated mask as binary
    """
    #Global variables
    global img
    global mask_image
    global complete_class_mask
    global img_masked
    global input_points
    global input_labels
    global ort_session
    global predictor
    global onnx_mask_input
    global onnx_has_mask_input
   
    #SAM configuration
    sam_checkpoint = pth_file
    model_type = model_type
    torch.device(device)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    onnx_model_path = onnx_file 
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # Read image and create window
    img = cv2.imread(image)
    height, width, _ = img.shape
    scale_percent = 95 # Porcentaje de reducción
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.namedWindow('Image')
    cv2.imshow('Image', img)
    mask_image = None
    complete_class_mask = np.zeros_like(img)
    img_masked = None
    input_points = []
    input_labels = []

    #Keyboard shortcuts
    print('KEYBOARD SHORTCUTS:','\n','ESC: Close window','\n','Left click: Add positive point','\n','CTRL + Left click: Add negative point','\n','SPACE: Add mask section','\n', 'ENTER: Confirm mask','\n','Backspace: Remove last point')
      
    # Assign the function to the window mouse event
    cv2.setMouseCallback('Image', mouse_callback)
    predictor.set_image(img)
    global image_embedding
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(type(image_embedding))
    # Wait for the window to close
    while True:
        # Wait for a key press for 1 millisecond
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ASCII code for "ESC"
            break
        elif key == 32:# ASCII code for "SPACE"
            if mask_image is None:
                print('No mask added','\n')
                cv2.imshow('Image', img)
            else:
                print('Mask added','\n')
                complete_class_mask = combine_mask(complete_class_mask, mask_image)
                cv2.imshow('Image', complete_class_mask)
            mask_image = None
            img_masked = None
            input_points = []
            input_labels = []
        elif key == 13: # ASCII code for "ENTER"
            #print(len(np.unique(complete_class_mask)))
            if len(np.unique(complete_class_mask)) == 1:
                print('No mask added','\n')
                cv2.imshow('Image', img)
            else:
                print('Mask confirmed','\n')
                if not export_binary_mask:
                    cv2.imwrite(output,complete_class_mask)
                else:
                    binary_image = np.zeros_like(complete_class_mask, dtype=np.uint8)
                    binary_image[complete_class_mask > 0] = 1
                    cv2.imwrite(output,binary_image)
                complete_class_mask=cv2.addWeighted(src1=img,alpha=1,src2=complete_class_mask,beta=0.8,gamma=0)
                cv2.imshow('Image', complete_class_mask)
            mask_image = None
            complete_class_mask = np.zeros_like(img)
            img_masked = None
            input_points = []
            input_labels = []

        elif key == 8: # ASCII code for "BACKSPACE"
            if len(input_labels) > 0 and len(input_labels)>0: 
                input_points.pop()
                input_labels.pop()
            if len(input_labels) == 0 and len(input_points) == 0:
                cv2.imshow('Image', img)
            else:
                onnx_points_input = np.concatenate([np.array(input_points), np.array([[0.0, 0.0]])], axis=0)[None, :, :]
                onnx_points_input = predictor.transform.apply_coords(onnx_points_input, img.shape[:2]).astype(np.float32)
                onnx_labels_input = np.concatenate([np.array(input_labels), np.array([-1])], axis=0)[None, :].astype(np.float32)
                ort_inputs = {"image_embeddings": image_embedding,"point_coords": onnx_points_input,"point_labels": onnx_labels_input,"mask_input": onnx_mask_input,"has_mask_input": onnx_has_mask_input,"orig_im_size": np.array(img.shape[:2], dtype=np.float32)}
                masks = predict_mask(ort_inputs)
                img_masked, mask_image = create_image_masked(masks, img)
                draw_points(input_points,input_labels,img_masked,img)
                
    cv2.destroyAllWindows()
