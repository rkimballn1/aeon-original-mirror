import sys


class Plugin:
    def __init__(self):
        pass

    def prepare(self):
        print('prepare not implemented')
        raise RuntimeError('Not implemented')

    def augment_image(self, image):
        print('augment image not implemented')
        raise RuntimeError('Not implemented')

    def augment_boundingbox(self, bboxes):
        print('augment boundingbox not implemented')
        raise RuntimeError('Not implemented')
