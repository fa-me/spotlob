"""The register is meant to be used to keep track of available
process steps. Using decorators, a function can be internally
turned into a :class:`~spotlob.SpotlobProcessStep` subclass, to be used
within a :class:`~spotlob.Pipeline`. This way, using only minimal code,
new functionality can be added to Spotlob and directly used
within its workflow

Example using a decorator
-------------------------
Create a process register

.. code-block:: python

    from spotlob.register import ProcessRegister
    register = ProcessRegister()

Create an alternative function, that should replace a single
:class:`~spotlob.SpotlobProcessStep`. Here, a binarization function is defined,
using upper and lower value boundaries, with `numpy`.

.. code-block:: python

    import numpy as np

    def my_threshold(image, lower_threshold, upper_threshold, invert):
        out = np.logical_and( image > lower_threshold,
                            image < upper_threshold)
        out = out.astype(np.uint8)*255
        if invert:
            out = ~out
        return out

To add this function to the register, use the methods of
:class:`~spotlob.ProcessRegister` as decorators, giving a list of parameter
specifications

.. code-block:: python

    @register.binarization_plugin([("lower_threshold",(0,255,100)),
                                   ("upper_threshold",(0,255,200)),
                                   ("invert", True)])
    def my_threshold(image, lower_threshold, upper_threshold, invert):
        out = np.logical_and( image > lower_threshold,
                            image < upper_threshold)
        out = out.astype(np.uint8)*255
        if invert:
            out = ~out
        return out
"""

from .parameters import parameter_from_spec
from .process_steps import Binarization, Converter, Preprocessor,\
    Postprocessor, Analysis


class ProcessRegister(object):

    def __init__(self):
        self.available_processes = dict()

    def binarization_plugin(self, param_spec):
        """
        register your binarization function as a plugin by using
        this function as a decorator
        """
        wrapper = self._get_wrapper(Binarization, param_spec)
        return wrapper

    def convert_plugin(self, param_spec):
        wrapper = self._get_wrapper(Converter, param_spec)
        return wrapper

    def preprocess_plugin(self, param_spec):
        wrapper = self._get_wrapper(Preprocessor, param_spec)
        return wrapper

    def postprocess_plugin(self, param_spec):
        wrapper = self._get_wrapper(Postprocessor, param_spec)
        return wrapper

    def analysis_plugin(self, param_spec):
        wrapper = self._get_wrapper(Analysis, param_spec)
        return wrapper

    def _get_wrapper(self, process_class_name, process_param_spec):
        def wrapper(process_function):
            # create SpotlobParameters out of definition in decorator
            spotlob_parameters = [parameter_from_spec(
                spec) for spec in process_param_spec]

            # instantiate given class name, handing over parameters,
            # do not register the object
            proc = process_class_name(
                process_function, spotlob_parameters, add_to_register=False)

            # register the object now, but via the name of the function
            self.available_processes[process_function.__name__] = proc
            return process_function

        return wrapper

    def register_process(self, process_instance):
        self.available_processes[process_instance.__class__] = process_instance


PROCESS_REGISTER = ProcessRegister()
