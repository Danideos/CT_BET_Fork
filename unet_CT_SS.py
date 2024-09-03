import os
import numpy as np
import nibabel as nb
import tensorflow as tf
import argparse
from tqdm import tqdm
from deepModels import Unet


def load_model(weights_path, image_shape, nb_classes=2):
    model = Unet(image_shape=image_shape, nb_classes=nb_classes)
    model.load_weights(weights_path)
    return model

def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU")

def load_test_data(images_path, each):
    img = nb.load(os.path.join(images_path, each))
    images = img.get_fdata()
    affine = img.affine
    if len(images.shape) == 2:
        images = images.reshape(images.shape[0], images.shape[1], 1)
    if len(images.shape) != 3:
        return None, None

    img_rows, img_cols, num_imgs = images.shape
    images = images.transpose(2, 0, 1).reshape(num_imgs, img_rows, img_cols, 1).astype(np.float32)
    return images, affine

def save_nifti_image(data, affine, save_path):
    nifti_img = nb.Nifti1Image(data, affine)
    nb.save(nifti_img, save_path)

def process_one_image(model, each, img_dir, save_path):
    try:
        print(f'Processing case: {each}')
        test_images, affine = load_test_data(img_dir, each)
        if test_images is None or test_images.shape[1] != 512 or test_images.shape[2] != 512:
            print(f"Skipping {each} due to shape mismatch: {test_images.shape if test_images is not None else 'None'}")
            return

        pred_image = model.predict(test_images, batch_size=8, verbose=1)
        pred_image = tf.argmax(pred_image, axis=-1).numpy().astype(np.uint8)
        pred_image = pred_image.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2])

        save_nifti_image(pred_image.transpose(1, 2, 0), affine, save_path)
        print(f'Processed {each} successfully.')
    except Exception as e:
        print(f"Skipping file {each}, error: {e}")

def main(img_dir, res_dir, weights_path, series=False):
    check_gpu()
    model = load_model(weights_path, image_shape=(512, 512, 1))

    if series:
        for series_dir in tqdm(os.listdir(img_dir)):
            os.makedirs(os.path.join(res_dir, series_dir))
            for each in os.listdir(os.path.join(img_dir, series_dir)): 
                save_path = os.path.join(res_dir, series_dir, each)
                load_img_dir = os.path.join(img_dir, series_dir)
                process_one_image(model, each, load_img_dir, save_path)
    
    else:
        files = [each for each in sorted(os.listdir(img_dir)) if each.endswith(".nii.gz")]
        for each in files:
            if each is None:
                continue
            save_path = os.path.join(res_dir, each)
            process_one_image(model, each, img_dir, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Brain Extraction Tool Prediction")
    parser.add_argument("--res_dir", type=str, required=True, help="Path to the directory that will contain resulting mask files")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing NIfTI images to do BET on")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the weights file for the model")
    args = parser.parse_args()

    main(args.img_dir, args.res_dir, args.weights_path)
