import abc

class BaseWriter(object):

    def __init__(self):
        pass

    @abc.abstractmethod
    def convert(self):
        pass
        
    def save(self):
        pass

