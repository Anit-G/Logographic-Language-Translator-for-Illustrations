import argparse
from pipelines_utils import craftbbox_utils
from crop_images import image_utils
from ocr_utils import OCR_ops
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

#CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='Models/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='Data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--results_folder',default='Results/',type=str,help='The folder which will store the results')
parser.add_argument('--csv',default='Results/data.csv',type=str,help='link to the data csv')
parser.add_argument('--crop_dir',default='Crop Words/',type=str,help='link to where the cropped words are saved')
parser.add_argument('--ocr_model', default='Models/ocr_model',type=str,help='OCR model path')
args = parser.parse_args()



if __name__ == '__main__':
    Pipeline = craftbbox_utils()
    img_obj = image_utils()
    OCR_Pipeline = OCR_ops(args.ocr_model,args.crop_dir,args.csv)

    Pipeline.craft_pipeline(args)
    img_obj.crop_bound_images(args.csv,args.results_folder,args.crop_dir)
    text = OCR_Pipeline.model_pipeline()

    # OCR_Pipeline.add_csv(text)
    print(text)
