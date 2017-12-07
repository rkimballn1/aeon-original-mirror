from plugin import Plugin


class plugin(Plugin):

    def __init__(self, param_string):
        pass

    def prepare(self):
        pass

    def augment_image(self, mat):
        return mat

    def augment_boundingbox(self, boxes):
        return boxes
