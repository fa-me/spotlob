import inspect


class Register(object):
    readers = []

    @classmethod
    def reader(cls, reader_function):
        reader = Reader.from_function(reader_function)
        cls.readers += [reader]
        return reader_function
