import numpy as np

from ipywidgets import IntSlider, FloatSlider, Text,\
    Dropdown, VBox, Checkbox, Button, Output
from IPython.display import display, clear_output

from .parameters import EnumParameter, BoolParameter
from .spim import Spim, SpimStage


class SpotlobNotebookGui(object):
    def __init__(self, pipeline, preview_screen, spim):
        self.pipeline = pipeline
        self.dummyspim = spim  # Spim.from_file(image_filepath, cached=True)
        self.preview_screen = preview_screen

    # def new_image(self, image_filepath):
    #     self.dummyspim = Spim.from_file(image_filepath, cached=True)
    #     self.run()

    def run(self):
        self.dummyspim = self.pipeline.apply_all_steps(self.dummyspim)
        self.update_preview(self.pipeline.process_stage_dict[SpimStage.loaded])

    def update_preview(self, process_that_changed):
        spim_before = self.dummyspim.get_at_stage(
            process_that_changed.input_stage)

        preview_image = process_that_changed.preview(spim_before)
        self.preview_screen.update(preview_image)

    def make_widgets(self):
        self.run()
        return self.pipeline_as_widget(self.pipeline)

    def show_preview_screen(self, *args, **kwargs):
        self.preview_screen.make_new(
            self.dummyspim.get_at_stage(SpimStage.loaded).image,
            *args, **kwargs)

    def parameter_as_widget(self, parameter, parent_process):
        v = parameter.value
        ty = parameter.type
        name = parameter.name

        # if isinstance(parameter, FilepathParameter):
        #     return fileselect.SelectFilesButton()
        # TODO: handle other than a string
        try:
            min_ = parameter.minvalue
            max_ = parameter.maxvalue
            step = parameter.step

            # use a slider
            if ty == float:
                widget_class = FloatSlider
            elif ty == int:
                widget_class = IntSlider
            else:
                NotImplementedError("Unsupported parameter type")

            widget = widget_class(min=min_, max=max_,
                                  step=step, value=v,
                                  description=name,
                                  continuous_update=False)

        except AttributeError:
            if isinstance(parameter, EnumParameter):
                widget = Dropdown(options=parameter.options,
                                  value=parameter.value, description=name)
            elif isinstance(parameter, BoolParameter):
                widget = Checkbox(value=parameter.value,
                                  description=name)
            elif ty == str:
                widget = Text(value=v, description=name, disabled=False)
            else:
                raise NotImplementedError(
                    "Could not construct widget for parameter type %s" % ty)

        def update_to_widgetstate(widgetstate):
            parameter.value = widgetstate["new"]

            # redo the whole invalidated pipeline up to the parent process
            self.dummyspim = self.pipeline.apply_outdated_up_to_stage(
                self.dummyspim, parent_process.input_stage)
            # update the preview of the process
            self.update_preview(parent_process)

            # mark process as outdated --> it has to be reapplied for others
            parent_process.outdated = True

        # bind functionality to widget ###########
        widget.observe(update_to_widgetstate, names="value")

        return widget

    def pipeline_as_widget(self, pipeline):
        return VBox([
            VBox([self.parameter_as_widget(p, process)
                  for p in process.parameters.parameters])
            for process in pipeline.processes if process
        ])

    def run_button(self):
        rb = Button(description="Evaluate")

        def run_inner(btn):
            self.run()

            fresh_predecessor = self.dummyspim.get_at_stage(SpimStage.loaded)

            # draw results
            image = np.copy(fresh_predecessor.image)

            before_analyzed = SpimStage.analyzed-1

            analyis_process = self.pipeline.process_stage_dict[before_analyzed]
            image_with_results = analyis_process.draw_results(
                image, self.results())
            self.preview_screen.update(image_with_results)

        rb.on_click(run_inner)
        return rb

    def results(self):
        return self.dummyspim.metadata["results"]

    def results_button(self):
        rb = Button(description="Show Results")

        def results_button_inner(btn):
            clear_output()
            display(rb)
            display(self.results())

        rb.on_click(results_button_inner)
        return rb
