import os
import cv2
import numpy as np

def get_original_image(name: str):
    filename, _ = os.path.splitext(name)
    image_name = filename.replace('videos', 'images') + '.png'
    vc = cv2.VideoCapture(name)

    count = 0
    frame = None
    rval = vc.isOpened()
    while rval:
        rval, frame = vc.read()
        count += 1

    if not frame:
        cv2.imwrite(image_name, frame)

    vc.release()
    return image_name

def video_to_image(video: str, sample_freq: int):
    image_path, _ = os.path.splitext(video.replace('videos', 'images'))
    if not os.path.exists(image_path):
        os.mkdir(image_path, )
    video_clip = cv2.VideoCapture(video)
    fps = video_clip.get(cv2.CAP_PROP_FPS)
    print(fps)

    stride = fps // sample_freq
    count = 0
    rval = video_clip.isOpened()
    while rval:
        rval, frame = video_clip.read()
        if count % stride == 0:
            cv2.imwrite(os.path.join(image_path, '{}.png'.format(count)), frame)

        count += 1

def image_to_video(image_dir: str, sample_freq: int):
    assert not image_dir.endswith('/')
    image_list = [os.path.join(image_dir, s) for s in os.listdir(image_dir) if s.endswith('.png')]
    image_list.sort()
    video_name = image_dir.replace('images', 'videos') + '.avi'

    h, w, _ = cv2.imread(image_list[0]).shape
    print(h, w)
    fourcc = cv2.VideoWriter_fourcc(*'DVIX')
    video_writer = cv2.VideoWriter(video_name, fourcc, sample_freq, (w, h))
    for s in image_list:
        frame = cv2.imread(s)
        video_writer.write(frame)

    video_writer.release()





if __name__ == '__main__':
    # the path of videos should be '../data/videos/( original | pre | post )/[0-9]+.avi'
    # the direction of the images should be '../data/images/( original | pre | post )/[0-9]+/'

    # video = '../data/videos/pre/bag.avi'
    # video_to_image(video, 3)

    images = '../data/images/post/bag'
    image_to_video(images, 3)
