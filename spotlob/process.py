"""
This module contains abstract super classes for concrete ProcessSteps.
Processes are applied to a Spim to bring it to the next stage.
"""


import numpy as np

from .spim import SpimStage
from .parameters import SpotlobParameterSet


class SpotlobProcessStep(object):
    """An abstract super class for process steps. A process step can be
    applied to bring a spim from one stage to another. It is supposed to save
    as internal state, if it has already been applied or needs to be applied
    again, because the parameters have changed. This is stored in the
    `outdated` parameter"""

    # stage that a spim should have before this Process can be applied
    input_stage = None

    # a process is outdated, if it has not been
    # applied since the parameters changed
    outdated = True

    def __init__(self, function, parameters, add_to_register=True):
        """
        A process step can be applied to bring a spim from one stage
        to another

        Parameters
        ----------
        function : function(spim, args) -> spim
            a function to be applied on spim with optional additional
            parameters
        parameters : list of SpotlobParameter
            the parameters that will be used for the function as soon as
            it is applied
        add_to_register : bool, optional
            if this is True, the process will be registered centrally now, i.e.
            upon creation of the SpotlobProcessStep object. This allows the
            user to know which processes are available (the default is True)

        """
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

        Parameters
        ----------
        spim : Spim
            Spim to draw the preview on. It must contain an image

        """
        raise NotImplementedError("abstract: to be implemented by subclass")

    def apply(self, *input_args):
        output = self.function(*input_args, **self.parameters.to_dict())
        self.outdated = False
        return output
