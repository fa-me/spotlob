class SpotlobParameter(object):
    def __init__(self, name, value, type_, description=""):
        self.name = name
        self.value = value
        self.type = type_
        self.description = description
        super(SpotlobParameter, self).__init__()

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

    def to_dict(self):
        return dict([(p.name, p.value) for p in self.parameters])


class FilepathParameter(SpotlobParameter):
    def __init__(self, name, path):
        super(FilepathParameter, self).__init__(name, path, str, "")


class EnumParameter(SpotlobParameter):
    def __init__(self, name, value, options, description=""):
        self.options = options
        super(EnumParameter, self).__init__(
            name, value, str, description)


class NumericRangeParameter(SpotlobParameter):
    def __init__(self, name, value, minvalue, maxvalue, type_=int, step=1, description=""):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.step = step
        super(NumericRangeParameter, self).__init__(
            name, value, type_, description)
