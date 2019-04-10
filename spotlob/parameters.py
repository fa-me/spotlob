def parameter_from_spec(spec):
    """This function will create a SpotlobParameter from a specification

    PARAMETERS
    ----------
    spec : tuple(str, object)
        specification for the parameter, must be one of the following
        options:

        +---------------+--------------------------------------+
        | float range   | ("parameter_name", (float_min_value, |
        |               |                     float_max_value, |
        |               |                     float_value))    |
        +---------------+--------------------------------------+
        | integer range | ("parameter_name", (int_value,       |
        |               |                     int_min_value,   |
        |               |                     int_max_value))  |
        +---------------+--------------------------------------+
        | boolean value | ("parameter name", boolean)          |
        +---------------+--------------------------------------+
        | enumeration   | ("parameter name", ["option1",       |
        |               |                     "option2",       |
        |               |                     "option3"])      |
        +---------------+--------------------------------------+

    RETURNS
    -------
    SpotlobParameter
        An instance of a SpotlobParameter subclass: EnumParameter,
        FloatParameter, ... depending on the type of the spec

    """
    try:
        parname, val = spec
        try:
            # split into min,max,value
            minv, maxv, v = val
            if any([type(vi) == float for vi in val]):
                return NumericRangeParameter(parname,
                                             float(v),
                                             float(minv),
                                             float(maxv),
                                             float)
            elif all([type(vi) == int for vi in val]):
                return NumericRangeParameter(parname,
                                             int(v),
                                             int(minv),
                                             int(maxv),
                                             int)
            else:
                raise TypeError

        except TypeError:
            # could not create a slider
            if type(val) == bool:
                return BoolParameter(parname, val)
            elif all([type(s) == str for s in val]):
                return EnumParameter(parname, val[0], val)
    except:
        raise Exception("Invalid parameter specification")


class SpotlobParameter(object):
    def __init__(self, name, value, type_, description=""):
        self.name = name
        self._value = value
        self.type = type_
        self.description = description
        self.preview_enabled = False
        super(SpotlobParameter, self).__init__()

    def __repr__(self):
        return "<SpotlobParameter(%s) %s: %s>"\
            % (self.type, self.name, self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        self._value = new_val

    def __str__(self):
        return "%s: %s" % (self.name, self.value)


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

    def __str__(self):
        lines = ["- %s\n" % p for p in self.parameters]
        return "".join(lines)


class FilepathParameter(SpotlobParameter):
    def __init__(self, name, path):
        super(FilepathParameter, self).__init__(name, path, str, "")


class EnumParameter(SpotlobParameter):
    def __init__(self, name, value, options, description=""):
        self.options = options
        super(EnumParameter, self).__init__(
            name, value, str, description)


class NumericRangeParameter(SpotlobParameter):
    def __init__(self, name, value, minvalue, maxvalue,
                 type_=int, step=1, description=""):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.step = step
        super(NumericRangeParameter, self).__init__(
            name, value, type_, description)


class BoolParameter(SpotlobParameter):
    def __init__(self, name, value):
        super(BoolParameter, self).__init__(name, value, bool, description="")
