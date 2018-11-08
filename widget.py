from ipywidgets import IntSlider, FloatSlider, Text, Dropdown, VBox
from IPython.display import display

from parameters import *


def parameter_as_widget(parameter):
    v = parameter.value
    ty = parameter.type
    name = parameter.name

    try:
        min_ = parameter.minvalue
        max_ = parameter.maxvalue
        step = parameter.step

        # use a slider
        if ty == float:
            WidgetClass = FloatSlider
        elif ty == int:
            WidgetClass = IntSlider
        else:
            NotImplementedError("Unsupported parameter type")

        widget = WidgetClass(min=min_, max=max_,
                             step=step, value=v,
                             description=name,
                             continuous_update=False)

    except AttributeError:
        # if isinstance(parameter, FilepathParameter):
        #    # TODO: return filepath selector
        #    pass
        if isinstance(parameter, EnumParameter):
            widget = Dropdown(options=parameter.options,
                              value=parameter.value, description=name)

        elif ty == str:
            widget = Text(value=v, description=name, disabled=False)

    def update_to_widgetstate(widgetstate):
        parameter.value = widgetstate["new"]

    widget.observe(update_to_widgetstate, names="value")
    return widget


def parameterset_as_widgets(parset):
    return VBox([parameter_as_widget(p) for p in parset.parameters])


def pipeline_as_widget(pipeline):
    return VBox([
        parameterset_as_widgets(process.parameters) for process in pipeline.processes if process
    ])
