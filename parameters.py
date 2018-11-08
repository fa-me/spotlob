class SpotlobParameter(object):
    def __init__(self, name, value, type_, description=""):
        self.name = name
        self.value = value
        self.type = type_
        self.description = description

    def __repr__(self):
        return "<SpotlobParameter(%s) %s: %s>" % (self.type, self.name, self.value)


class SpotlobParameterSet(object):
    def __init__(self, parameters):
        self.parameters = parameters

    @property
    def names(self):
        return [p.name for p in self.parameters]

    @property
    def values(self):
        return [p.value for p in self.parameters]

    def __getitem__(self, identifier):
        try:
            return self.parameters[identifier]
        except TypeError:
            ind = self.names.index(identifier)
            return self.parameters[ind]


class FilepathParameter(SpotlobParameter):
    def __init__(self, path):
        super(FilepathParameter, self).__init__(
            "Filepath", path, str, "")


class FloatParameter(SpotlobParameter):
    def __init__(self, name, value, minvalue, maxvalue, description=""):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        super(FloatParameter, self).__init__(
            name, value, float, description)


class IntParameter(SpotlobParameter):
    def __init__(self, name, value, minvalue, maxvalue, description=""):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        super(IntParameter, self).__init__(
            name, value, int, description)
