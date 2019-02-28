import numpy as np

from .spim import SpimStage
from .parameters import SpotlobParameterSet


class SpotlobProcessStep(object):
    """A super class for process steps"""

    # stage that a spim should have before this Process can be applied
    input_stage = None

    # a process is outdated, if it has not been
    # applied since the parameters changed
    outdated = True

    def __init__(self, function, parameters, add_to_register=True):
        self.function = function
        self.parameters = SpotlobParameterSet(parameters)

        # import needs to be delayed to avoid circular imports
        from .register import PROCESS_REGISTER

        if add_to_register:
            PROCESS_REGISTER.register_process(self)

    def preview(self, spim):
        """
        This function takes spim at an undefined stage and draws
        the effect of the process on top, to provide a preview for the
        user on how the funciton will work. No storage or sideeffects
        should take place.
        In contrast to the apply function it must always return an image
        """
        raise NotImplementedError("abstract: to be implemented by subclass")

    def apply(self, *input_args):
        output = self.function(*input_args, **self.parameters.to_dict())
        self.outdated = False
        return output
