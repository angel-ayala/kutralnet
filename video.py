# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
from datasets import available_datasets
from utils.training import test_model
from utils.models import models_conf

parser = argparse.ArgumentParser(description='Fire classification training')
parser.add_argument('--base_model', metavar='BM', default='kutralnet',
                    help='modelo a entrenar')
parser.add_argument('--weights_path', metavar='W', default=os.path.join('.', 'models', 'saved'),
                    help='parametros del modelo')
parser.add_argument('--dataset', metavar='D', default='fismo',
                    help='seleccion de dataset de entrenamiento')
parser.add_argument('--video_source', metavar='V', default='0',
                    help='seleccion de video')
parser.add_argument('--model_version', metavar='MODELVER', default=None,
                    help='seleccion de modelo')
args = parser.parse_args()

# constant
root_path = '.'
dataset_path = os.path.join(root_path, 'datasets')
test_root_path = os.path.join(dataset_path, 'video_test')

# choose model
base_model = args.base_model#'octfiresnet'
# video test config
# batch_size = 32
# epochs = args.epochs#100
# shuffle_dataset = True
# preload_data = bool(args.preload_data)#False # load dataset on-memory

# model pre-configuration
config = models_conf[base_model]
img_dims = config['img_dims']
model_name = config['model_name']
num_classes = 2

# model selection
if base_model in models_conf:
    module = importlib.import_module(config['module_name'])
    fire_model = getattr(module, config['class_name'])
    model = fire_model(classes=num_classes)
else:
    raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet (and lite variations)]')

# video
if args.video_source == '0':
    video_path = 0
else:
    video_path = os.path.join(test_root_path,
        'slow_motion_fire_blaze_from_the_bottom_stock_video_footage_cPYaQ-_MKt0_360p.mp4'
        # 'raw_video_shows_how_fast_texas_wildfire_spread_m67ZokFYl2A_360p.mp4'
        # 'slow_motion_with_fire_W0iMjuRXYZo_360p.mp4'
        # 'dalma_400240.mp4'
        # 'gwanak_400240.mp4'
        # 'nofire_400240.mp4'
        # 'inside_the_fire_zvPa_yEEd4E_360p.mp4'
        )#'FireSenseDataset', 'Fire', 'posVideo2.871.avi')
# video_path = 0
print('Loading video source', video_path)

# folder for save results
training_dataset = args.dataset
version = args.model_version
final_folder = training_dataset if version is None else '{}_{}'.format(training_dataset, version)
folder_name = os.path.join(base_model, final_folder)
models_root = args.weights_path
folder_path = os.path.join(models_root, folder_name)
model_path = os.path.join(folder_path, model_name)
print('Loading model', model_path, 'trained with', training_dataset)

# net = FireNet(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def init_capture(path):
    # model prep
    labels = ['NoFire', 'Fire']
    # init capture
    cap = cv2.VideoCapture(path)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    i = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #
        image_r = cv2.resize(frame, img_dims)
        # Normalize data.
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
        image_r = image_r.astype('float32') / 255
        image = np.expand_dims(image_r, axis=0)

        with torch.no_grad():
            pred = model(torch.from_numpy(image.transpose((0, 3, 1, 2))))
            pred = F.softmax(pred, dim=1)

        print('frame: {:05d} = '.format(i), 'pred', pred)
        accuracy = pred[0]
        nofire_perc = '{}: {:.2f}%'.format(labels[0], accuracy[0]*100)
        fire_perc = '{}: {:.2f}%'.format(labels[1], accuracy[1]*100)
        #print(pred, np.argmax(pred, axis=1), labels[np.argmax(pred)])

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        # Our operations on the frame come here
        if path == 0:
            frame = cv2.flip(frame, 1)
        # frame = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)

        # puts fps
        cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.50, color=(255, 0, 0), thickness=2)
        # no fire label
        cv2.putText(frame, text=nofire_perc, org=(3, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.50, color=(0, 255, 0), thickness=2)
        # fire label
        cv2.putText(frame, text=fire_perc, org=(3, 55), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.50, color=(0, 0, 255), thickness=2)
        # alert
        if accuracy[1] > .5:
            cv2.putText(frame, text='Fuego!', org=(3, 75),  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(test_root_path, 'frames', 'frame_{:05d}.png'.format(i)), frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    init_capture(video_path)
    # get_model()
