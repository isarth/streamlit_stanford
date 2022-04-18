
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import time
import pickle
import torch

from models_arc import *

n_classes = 40

fl = open("../stanford_class_labels.pkl",'rb')
classes = pickle.load(fl)
classes = list(classes.keys())

model = ConvModel_BnResnet(n_classes)
model.eval()
model.load_state_dict(torch.load('../best_model_stanford.pt', map_location=torch.device('cpu')))

def load_image(image_file):
  img = Image.open(image_file)
  img = img.resize((224,224))
  img = img.convert('RGB')
  return img


st.title("Stanford-40 Demo App!")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if image_file is not None:
    img = load_image(image_file)
    st.image(img, use_column_width=True)
    img = np.asarray(img, dtype=np.float32)/255.0
    img = np.rollaxis(img, 2, 0)
    img_tensor = torch.tensor(img).view(1,3,224,224)
    prediction = model(img_tensor).detach().view(-1).numpy()
    probs = np.exp(prediction)
    anss = np.argsort(probs)[:5]
    ans = [classes[i] for i in anss]
    text = "Top 5 predictions in order are \n"
    for n,i in enumerate(ans):
      text += f"{n+1}. {i} " + "\n"

    st.text(f"{text}")

