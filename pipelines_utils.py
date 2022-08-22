import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn

import cv2
import pandas as pd

from CRAFT.craft_utils import getDetBoxes,adjustResultCoordinates
from crop_images import generate_words
from CRAFT.test import copyStateDict, test_net
from CRAFT.file_utils import get_files, saveResult
from CRAFT.imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg, loadImage
from CRAFT.craft import CRAFT


class craftbbox_utils():
    def __init__(self):
        # Class variables
        pass

    def create_dir(self,test_folder,results_folder):
        image_list, _, _ = get_files(test_folder)
        image_names = []
        image_paths = []

        # Get all the images names
        for i in range(len(image_list)):
            image_names.append(str("res_"+os.path.relpath(image_list[i], test_folder)))

        data = pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
        data['image_name'] = image_names

        # Check if the folder already exists
        if os.path.isdir(results_folder):
            shutil.rmtree(results_folder)
            os.mkdir(results_folder)
        else:
            os.mkdir(results_folder)

        if not os.path.isdir(test_folder):
            os.mkdir(test_folder)

        return image_list,data

    def load_net(self,trained_model,cuda):
        net = CRAFT()
        print('Loading weights from checkpoint (' + trained_model + ')')
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        # Evaluation mode
        net.eval()
        return net


    def LineRefiner(self,refiner_model,cuda):
        from CRAFT.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

        refine_net.eval()
        refine_net = refine_net
        poly = True
        return refine_net,poly

    def get_data(self,image_list,data,result_folder,net,refine_net,poly,text_threshold,link_threshold,low_text,cuda,args):
        t = time.time()
        for k,image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
            image = loadImage(image_path)

            bboxes, polys, score_text, det_scores = test_net(net, image, text_threshold, link_threshold,
                                                             low_text, cuda, poly, args, refine_net)

            bbox_score = {}

            for box_num in range(len(bboxes)):
                key = str(det_scores[box_num])
                item = bboxes[box_num]
                bbox_score[key] = item
            data['word_bboxes'][k] = bbox_score
            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder +"res_"+ filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

        data.to_csv(args.csv, sep=',', na_rep='Unknown')
        print("elapsed time : {}s".format(time.time() - t))

        return data


    def craft_pipeline(self,args):

        # Bounding Box the images
        image_list,data = self.create_dir(args.test_folder,args.results_folder)
        net = self.load_net(args.trained_model,args.cuda)

        refine_net = None
        if args.refine:
            refine_net,args.poly = self.LineRefiner(args.refiner_model,args.cuda)

        data = self.get_data(image_list,data,args.results_folder,net,refine_net,args.poly,
                             args.text_threshold,args.link_threshold,
                             args.low_text, args.cuda,args)
        return data
