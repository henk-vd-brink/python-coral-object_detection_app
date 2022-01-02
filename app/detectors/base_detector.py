from abc import ABCMeta, abstractmethod

class BaseDetector(metaclass=ABCMeta):

    @abstractmethod
    def detect(self, image):
        ...