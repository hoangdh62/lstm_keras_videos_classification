from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, LSTM, BatchNormalization, Flatten, Dropout, TimeDistributed, Lambda, Input
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import categorical_accuracy
import os
import cv2
import numpy as np
from random import randint
from PIL import Image
from tqdm import tqdm


def create_model(categories, batch_size, frames, channels, pixels_x, pixels_y):
    input = Input(shape=(frames, pixels_x, pixels_y, channels))
    input_resnet = Lambda(lambda x: K.reshape(input, shape=(batch_size*frames, pixels_x, pixels_y, channels)))(input)
    out_resnet = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(pixels_x, pixels_y, 3),
        pooling='max'
    )(input_resnet)
    # out_resnet = K.reshape(out_resnet, shape=(batch_size, frames, -1))
    out_resnet = Lambda(lambda x: K.reshape(out_resnet, shape=(batch_size, frames, -1)))(out_resnet)
    out_lstm = LSTM(512, activation='relu', return_sequences=False)(out_resnet)
    middle_dense = Dense(256, activation='relu')(out_lstm)
    # middle_dense = Dropout(0.5)(middle_dense)
    out = Dense(len(categories), activation='softmax')(middle_dense)

    model = Model(input, out)
    return model


def generator(root, batch_size, num_frames, width, height, channels, is_train=True, return_image=False):
    categories = os.listdir(root)
    cat_map = {}
    for i, cat in enumerate(categories):
        cat_map[cat] = i

    if is_train:
        with open('trainlist01.txt', 'r') as file:
            lines = file.readlines()
            data_paths = [(root + path.split()[0].strip(), path.split('/')[0]) for path in lines]
    else:
        with open('testlist01.txt', 'r') as file:
            lines = file.readlines()
            data_paths = [(root + path.split()[0].strip(), path.split('/')[0]) for path in lines]

    while True:
        batch_xdata = np.zeros(shape=(batch_size, num_frames, width, height, channels))
        batch_ydata = np.zeros(shape=(batch_size, len(categories)))
        videos = []
        # chọn ngẫu nhiên 'batch_size' videos từ kho
        video_indexs = np.random.choice(list(range(len(data_paths))), size=batch_size)
        for idx, video_index in enumerate(video_indexs):
            video_path, cat = data_paths[video_index]
            cat_id = cat_map[cat]
            data = []
            frames = []
            cap = cv2.VideoCapture(video_path)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                # frame = frame/128 - 1
                # resize image to width, height
                w, h = frame.shape[:-1]
                scale = min(width/w, height/h)
                nw = int(scale*w)
                nh = int(scale*h)
                frame = cv2.resize(frame, (nh, nw))
                # paste image in zeros background
                frame_data = np.zeros(shape=(width, height, channels))
                frame_data[(width-nw)//2:(width-nw)//2+nw, (height-nh)//2:(height-nh)//2+nh, :] = frame
                # cv2.imshow('test', frame_data)
                data.append(frame_data)
            cap.release()
            # Cắt bớt frame ở đầu nếu số frame lơn hơn num_frames
            # Nếu thiếu thì padding trái
            if len(data) > num_frames:
                offset = randint(0, len(data) - num_frames)
                data = data[offset: offset+num_frames]
                frames = frames[offset: offset+num_frames]
            # paste it in batch_data
            offset = num_frames - len(data)
            batch_xdata[idx, offset:, :, :, :] = np.asarray(data)
            batch_ydata[idx, cat_id] = 1
            videos.append(frames)
        if return_image:
            yield videos, batch_xdata, batch_ydata
        else:
            yield batch_xdata, batch_ydata

def process_video(video_path, width, height, channels):
    processed_frames = []
    frames = []
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        w, h = frame.shape[:-1]
        scale = min(width / w, height / h)
        nw = int(scale * w)
        nh = int(scale * h)
        frame = cv2.resize(frame, (nh, nw))
        # paste image in zeros background
        frame_data = np.zeros(shape=(width, height, channels))
        frame_data[(width - nw) // 2:(width - nw) // 2 + nw, (height - nh) // 2:(height - nh) // 2 + nh, :] = frame
        processed_frames.append(frame_data)
        frames.append(frame)
    cap.release()
    return np.asarray(processed_frames), frames

def predict_val(model, root, num_frames, width, height, channels):
    data_paths = []
    categories = os.listdir(root)
    cat_map = {}
    for cat_id, cat in enumerate(categories):
        cat_map[cat] = cat_id
        data_paths += ['{}{}/{}'.format(root, cat, video) for video in os.listdir(root + cat)][:4]

    for data_path in tqdm(data_paths):
        video_name = data_path.split('/')[-1]
        data, frames = process_video(data_path, width, height, channels)
        batch_size = len(data)//(num_frames//2) - 1

        fheight, fwidth = frames[0].shape[:-1]
        out = cv2.VideoWriter('output/' + video_name, cv2.VideoWriter_fourcc(*'XVID'), 24, (fwidth, fheight))
        for idx in range(batch_size):
            begin = (num_frames//2)*idx
            end = begin + num_frames
            X = np.expand_dims(data[begin:end, :], axis=0)
            y_pred = model.predict(X)
            cat_id = np.argmax(y_pred[0])
            cat_pre = categories[cat_id]
            confidence = y_pred[0][cat_id]
            for i in range(begin, end, 1):
                frames[i] = cv2.putText(frames[i], '{}:{}'.format(cat_pre, confidence), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        for frame in frames:
            out.write(frame)
        out.release()

if __name__ == '__main__':
    root_data = 'data/'
    log_dir = 'logs/'
    batch_size = 4
    num_frames = 6
    width = 224
    height = 224
    channels = 3
    # gen = generator(root_data, batch_size, num_frames, width, height, channels, True)
    # for data in gen:
    #     x = data

    categories = os.listdir(root_data)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    model = create_model(categories, batch_size, num_frames, channels, width, height)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=["accuracy"])
    # model.load_weights('logs/ep008-loss0.076-val_loss0.987.h5')
    history = model.fit_generator(generator(root_data, batch_size, num_frames, width, height, channels, True),
                        steps_per_epoch=200,
                        validation_data=generator(root_data, batch_size, num_frames, width, height, channels, False),
                        validation_steps=300,
                        epochs=20,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr])
    with open('logs.txt', 'a') as file:
        file.writelines('batch: {}, num_frames: {}\n'.format(batch_size, num_frames))
        file.writelines('loss: {}\n'.format(history['loss'].join(' ')))
        file.writelines('acc: {}\n'.format(history['acc'].join(' ')))
        file.writelines('val_loss: {}\n'.format(history['val_loss'].join(' ')))
        file.writelines('val_acc: {}\n'.format(history['val_acc'].join(' ')))

    # results = model.evaluate_generator(generator(root_data, batch_size, num_frames, width, height, channels, False),
    #                                   steps=20, verbose=1)
    # print(results)

    # gen = generator(root_data, batch_size, num_frames, width, height, channels, is_train=False, return_image=True)
    # count = 0
    # for i in tqdm(range(100)):
    #     videos, X, y = gen.__next__()
    #     y_preds = model.predict(X)
    #     for i in range(batch_size):
    #         frames = videos[i]
    #         category = categories[np.argmax(y_preds[i])]
    #         if np.argmax(y[i]) == np.argmax(y_preds[i]):
    #             count += 1
    # print(count)
    # predict_val(model, root_data, num_frames, width, height, channels)