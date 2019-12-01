import os
import cv2
import torch
import numpy as np
import pandas as pd

class Helper:
    def __init__(self, name, path, size=(224, 224), debug=False):
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.root_path, path)
        self.name = name
        self.size = size
        self.debug = debug
        self.classes = {
            'no_fire': {
                'idx': 0,
                'label': 'no_fire',
                'name': 'NoFire'
            },
            'fire': {
                'idx': 1,
                'label': 'fire',
                'name': 'Fire'
            }
        }
    # end __init__

    def clean_string(self, string):
        return string.rstrip().replace('./', '')
    # end clean_string

    def preprocess(self, batch):
        return batch.astype('float32') / 255.

    def read_and_resize(self, img_path, color_fix=True):
        img = cv2.imread(img_path)
        if color_fix:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.size is None and isinstance(self.size, tuple):
            img = cv2.resize(img, self.size)
        else:
            raise ValueError('size must be a tuple (width, height)')
        return img
    # end read_and_resize

    def read_from_csv(self, csv_name='dataset.csv', val_split=True):
        # dataset storage
        x_train, y_train = [], []
        if val_split:
            x_test, y_test = [], []

        # csv read
        dataset_df = pd.read_csv(os.path.join(self.path, csv_name))

        print('Loading images...')
        # rows iterate and image load
        for index, row in dataset_df.iterrows():
            path = os.path.join(self.path, row['folder_path'], row['image_id'])
            if self.debug:
                print(path, end=' ')
            img = self.read_and_resize(path)
            target = self.classes[row['class']]['idx']

            if val_split and row['purpose'] == 'test':
                x_test.append(img)
                y_test.append(target)
            else:
                x_train.append(img)
                y_train.append(target)

            if self.debug:
                print('ok')

        print('Ok')
        if val_split:
            return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        else:
            return np.array(x_train), np.array(y_train)
    # read_from_csv

    def read_test_from_csv(self, csv_name='dataset.csv'):
        # dataset storage
        x_test, y_test = [], []
        # csv read
        dataset_df = pd.read_csv(os.path.join(self.path, csv_name))
        dataset_df = dataset_df.loc[dataset_df['purpose'] == 'test']

        print('Loading test images...')
        # rows iterate and image load
        for index, row in dataset_df.iterrows():
            path = os.path.join(self.path, row['folder_path'], row['image_id'])
            if self.debug:
                print(path, end=' ')
            img = self.read_and_resize(path)
            target = self.classes[row['class']]['idx']

            x_test.append(img)
            y_test.append(target)

            if self.debug:
                print('ok')

        return np.array(x_test), np.array(y_test)
    # end read_test_from_csv

    def get_frames_from_video(self, video_path, prev_imgs=None, debug=False):
        imgs = [] if prev_imgs is None else prev_imgs

        cap_qt = 0
        frame_count = 0 #numbers of frame
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # get one frame per second
        frames_skip = int(fps)
        frames_qt = int(total_frames / fps)

        print('Video Info:',
            '{}x{}'.format(width, height),
            'FPS: {:.3f}'.format(fps),
            'Total Frames: {}'.format(total_frames),
            'skips: {}'.format(frames_skip),
            'qt: {}'.format(frames_qt))

        while True:
            if cap_qt == frames_qt:
                break

            frame_count += 1 # frames add

            if (frame_count % frames_skip) == 0:
                ret, frame = cap.read()
                if ret == False:
                    break
                imgs.append(frame)
                cap_qt += 1

        cap.release()

        return imgs
    # end get_dataset_from_video

    def extract_frames_to_folder(self, videos_path, prefix, save_folder):
        videos = glob.glob(os.path.join(videos_path, '*.avi'))
        #frame storage
        imgs = []
        print('Loading video frames with 1 FPS interval...')
        print(len(videos), 'videos found at', videos_path)
        for video_path in videos:
            fire_imgs = self.get_frames_from_video(video_path,
                                                prev_imgs=imgs)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        i = 0
        for image in imgs:
            filename = '{}{:05d}.png'.format(prefix, i)
            cv2.imwrite(os.path.join(save_folder, filename), image)
            i += 1

        print('Done')
    # end extract_frames_to_folder

    def split_dataframe(self, dataframe, quantity):
        # shufle data and reset
        dataframe = dataframe.sample(frac=1.)
        dataframe.reset_index(drop=True, inplace=True)

        # purpose labeling
        dataframe['purpose'] = 'training'
        dataframe.loc[dataframe.tail(quantity).index, 'purpose'] = 'test'

        return dataframe
    # end split_dataframe

    def save_dataframe(self, dataframe, filename='dataset.csv'):
        csv_path = os.path.join(self.path, filename)
        dataframe.to_csv(csv_path, index=False)
        print('Saved at', csv_path)
    # end save_dataframe

    def load_train_val(self, val_split=True):
        # check for csv loading
        if not os.path.isfile(os.path.join(self.path, 'dataset.csv')):
            prepare_training()
        # load images
        print('Loading', self.name, 'dataset with', self.size, 'dimensions')
        return self.read_from_csv(csv_name='dataset.csv', val_split=val_split)
    # end load_train_val

    def load_test(self):
        # check for csv loading
        if not os.path.isfile(os.path.join(self.path, 'test_dataset.csv')):
            prepare_testing()
        # load images
        print('Loading {}-Test dataset with'.format(self.name), self.size, 'dimensions')
        return self.read_from_csv(csv_name='test_dataset.csv', val_split=False)
    # end load_test
