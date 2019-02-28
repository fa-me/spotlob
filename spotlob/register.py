from .process_steps import *
from .parameters import parameter_from_spec


class ProcessRegister(object):
    available_processes = dict()

    def binarisation_plugin(self, param_spec):
        """register your binarisation function as a plugin by using this function as a decorator"""
        wrapper = self._get_wrapper(Binarisation, param_spec)
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

            # instantiate given class name, handing over parameters, do not register the object
            proc = process_class_name(
                process_function, spotlob_parameters, add_to_register=False)

            # register the object now, but via the name of the function
            self.available_processes[process_function.__name__] = proc
            return process_function

        return wrapper

    def register_process(self, process_instance):
        self.available_processes[process_instance.__class__] = process_instance


PROCESS_REGISTER = ProcessRegister()
