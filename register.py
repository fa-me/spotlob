import inspect
from parameters import *


class Register(object):
    readers = []

    @classmethod
    def reader(cls, reader_function):
        reader = Reader.from_function(reader_function)
        cls.readers += [reader]
        return reader_function


class SpotlobProcessStep(object):
    def __init__(self, function, parameters):
        self.function = function
        self.parameters = SpotlobParameterSet(parameters)


class Reader(SpotlobProcessStep):
    def __init__(self, reader_function, parameters):
        super(Reader, self).__init__(reader_function, parameters)

    def apply(self):
        return self.function(*self.parameters.values)

    @classmethod
    def from_function(cls, reader_function):
        argspecs = inspect.getargspec(reader_function)

        try:
            assert len(argspecs.args) == 1
            assert type(argspecs.defaults[0]) == str
        except AssertionError:
            raise Exception(
                "Could not register function. Invalid signature for reader function %s" % reader_function.__name__)

        fpp = FilepathParameter(argspecs.defaults[0])
        return Reader(reader_function, [fpp])
