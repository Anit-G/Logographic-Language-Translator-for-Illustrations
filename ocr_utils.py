import os

import numpy as np
import pandas as pd
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import re
import jaconv
from PIL import  Image

class OCR_ops():
    def __init__(self, model_path,data_path,csv_path):
        self._model_path = model_path
        self._data_path =data_path
        self._csv_path = csv_path
        pass

    def post_process(self,text):
        text = ''.join(text.split())
        text = text.replace('…', '...')
        text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        text = jaconv.h2z(text, ascii=True, digit=True)
        return text

    def load_net(self):
        model_path = self._model_path
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)

        return feature_extractor, tokenizer, model

    def ocr_model(self,image):
        # Load the net
        feature_extractor, tokenizer, model = self.load_net()

        #Run images through the net
        image = image.convert('L').convert('RGB')
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        ouput = model.generate(pixel_values)[0]
        text = tokenizer.decode(ouput, skip_special_tokens=True)
        text = self.post_process(text)

        return text

    def get_data(self):
        # Get the image data
        image_paths = [os.path.join(self._data_path, i) for i in os.listdir(self._data_path)]
        image_samples = [Image.open(i) for i in image_paths]

        return image_samples
    def add_csv(self,text):
        # add the recognized words to CSV
        data = pd.read_csv(self._csv_path)
        data['pred_words'] = text
        data.to_csv(self._csv_path, sep=',', na_rep='Unknown')
        pass
    def model_pipeline(self):
        # Get the data
        image_samples = self.get_data()

        # Run the models
        text = [self.ocr_model(x) for x in image_samples]
        return text







