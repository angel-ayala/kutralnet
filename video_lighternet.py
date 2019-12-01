# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
from models.lighternet_v2 import LighterNet

# constant
root_path = '.'
dataset_path = os.path.join(root_path, 'datasets')
test_root_path = os.path.join(dataset_path, 'video_test')

# img redimension
img_dims = (68, 68)
num_classes = 2

# video
video_path = os.path.join(test_root_path,
    # 'slow_motion_fire_blaze_from_the_bottom_stock_video_footage_cPYaQ-_MKt0_360p.mp4'
    # 'raw_video_shows_how_fast_texas_wildfire_spread_m67ZokFYl2A_360p.mp4'
    # 'slow_motion_with_fire_W0iMjuRXYZo_360p.mp4'
    'dalma_400240.mp4'
    # 'gwanak_400240.mp4'
    # 'nofire_400240.mp4'
    # 'inside_the_fire_zvPa_yEEd4E_360p.mp4'
    )#'FireSenseDataset', 'Fire', 'posVideo2.871.avi')
# video_path = 0
print('loading', video_path)

models_root = os.path.join(root_path, 'models', 'saved')
model_path = os.path.join(models_root, 'model_lighternet_v2.pth')

net = LighterNet(num_classes)
net.load_state_dict(torch.load(model_path))
net.eval()

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
            pred = net(torch.from_numpy(image.transpose((0, 3, 1, 2))))
            pred = F.softmax(pred, dim=1)

        print('pred', pred)
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
