"""
The detection of features within an image with spotlob is split up into
an abstract but fixed sequence of processes. Any of these process steps is
applied onto a Spim and returns a new Spim. The new Spim contains the
information added by the process step.

.. graphviz::

    digraph seq {
        rankdir="LR";
        "spim1" -> "spim2" [label="spim1.function(process)"];
    }

For example, a `Spim` at stage `SpimStage.loaded` can be converted, using
a concrete subclass of `Converter`, named `process_opencv.GreyscaleConverter`
in the following way:

.. code-block:: python

    from spotlob.defaults import load_image
    from spotlob.process_opencv import GreyscaleConverter

    loaded_spim = load_image("color_image.jpg")
    my_converter = GreyscaleConverter()
    converted_spim = loaded_spim.convert(my_converter)
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
        function : callable
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
        """Apply the process step onto input data (eg. images or contours)
        and yield the output data (again, could be a modified image or gathered
        data). The actual return values depend on the concrete implementation
        of the function. Additional arguments for the function are the
        parameters of the process as stored in the `parameters` field of each
        process.

        Returns
        -------
        Contours or images
            Output of the concrete process implementation
        """

        output = self.function(*input_args, **self.parameters.to_dict())
        self.outdated = False
        return output

    def __str__(self):
        classname = str(type(self).__name__)
        parameters_str = str(self.parameters)
        return "%s\n%s" % (classname, parameters_str)
