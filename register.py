import process
from parameters import parameter_from_spec

# TODO: should be possible to turn this into non-static class


class ProcessRegister(object):
    available_processes = dict()

    @classmethod
    def binarisation_plugin(cls, param_spec):
        """register your binarisation function as a plugin by using this function as a decorator"""
        wrapper = cls._get_wrapper(process.Binarisation, param_spec)
        return wrapper

    @classmethod
    def preprocess_plugin(cls, param_spec):
        wrapper = cls._get_wrapper(process.Preprocessor, param_spec)
        return wrapper

    @classmethod
    def postprocess_plugin(cls, param_spec):
        wrapper = cls._get_wrapper(process.Postprocessor, param_spec)
        return wrapper

    @classmethod
    def analysis_plugin(cls, param_spec):
        wrapper = cls._get_wrapper(process.Analysis, param_spec)
        return wrapper

    @classmethod
    def _get_wrapper(cls, process_class_name, process_param_spec):
        def wrapper(process_function):
            # create SpotlobParameters out of definition in decorator
            spotlob_parameters = [parameter_from_spec(
                spec) for spec in process_param_spec]

            # instantiate given class name, handing over parameters, do not register the object
            proc = process_class_name(
                process_function, spotlob_parameters, add_to_register=False)

            # register the object now, but via the name of the function
            cls.available_processes[process_function.__name__] = proc
            return process_function

        return wrapper

    @classmethod
    def register_process(cls, process_instance):
        cls.available_processes[process_instance.__class__] = process_instance
