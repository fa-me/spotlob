"""
The pipeline structure is used to define a sequence of processes
to be applied one after another onto a spim, to automate a
detection task consisting of multiple process steps.
"""

import dill


class Pipeline(object):
    """A pipeline is a sequence of processes, that can be applied one after
    another.
    The processes are stored in a Dictionary, along with the SpimStage at which
    they can be applied. The pipeline can be applied completely using the
    `apply_all_steps` method or partially using the `apply_from_stage_to_stage`
    method.
    """

    def __init__(self, processes):
        self.process_stage_dict = dict([(p.input_stage, p) for p in processes])

    @property
    def processes(self):
        return self.process_stage_dict.values()

    def replaced_with(self, new_process):
        """This will give a new pipeline, where one process is replaced with
        the given one

        Parameters
        ----------
        new_process : SpotlobProcessStep
            the new process to be inserted

        Returns
        ----------
        Pipeline
            the pipeline, that includes `new_process`

        """
        new_dict = self.process_stage_dict.copy()
        new_dict[new_process.input_stage] = new_process

        return Pipeline(new_dict.values())

    def apply_from_stage_to_stage(self, spim, from_stage, to_stage):
        """Recursively applies the pipeline-processes from a given stage up to
        another given stage.

        Parameters
        ----------
        spim : Spim
            The image item to apply parts of the pipeline to
        from_stage : int
            SpimStage at which stage the first process should be applied
        to_stage : int
            SpimStage at which stage the last process should be applied

        Returns
        ----------
        Spim
            The processed Spim at stage `to_stage`

        Raises
        ------
        Exception:
            If to_stage is before from_stage

        """
        if from_stage == to_stage:
            return spim
        elif from_stage > to_stage:
            # return self.apply_from_stage_to_stage(spim, to_stage, from_stage)
            raise Exception("invalid apply request")
        else:
            # recursive
            intermediate_spim = self.apply_from_stage_to_stage(
                spim, from_stage, to_stage-1)
            last_process = self.process_stage_dict[to_stage-1]
            outspim = intermediate_spim.do_process_at_stage(last_process)
            return outspim

    def _maxstage(self):
        return max(self.process_stage_dict.keys())+1

    def apply_all_steps(self, spim):
        """Apply the complete pipeline on a given spim

        Parameters
        ----------
        spim : Spim
            the spotlob image item to apply the complete pipeline to

        Returns
        -------
        Spim
            a spotlob image item a the stage after the last process
        """

        minstage = min(self.process_stage_dict.keys())
        maxstage = self._maxstage()
        return self.apply_from_stage_to_stage(spim, minstage, maxstage)

    def apply_at_stage(self, spim):
        """Applies all steps following the stage of the spim

        Parameters
        ----------
        spim : Spim
            the spotlob image item to apply the pipeline to

        Returns
        ----------
        Spim
            the processed Spim at stage `to_stage`
        """

        startstage = spim.stage
        maxstage = self._maxstage()
        return self.apply_from_stage_to_stage(spim, startstage, maxstage)

    def apply_outdated_up_to_stage(self, spim, up_to_stage):
        """Applies all processes since the first outdated one on spim up to
        a given stage if no process is outdated or if the outdated stage
        is past up_to_stage, spim is processed up to up_to_stage,
        or a predecessor is returned at up_to_stage

        Parameters
        ----------
        spim : Spim
            Spim to apply the pipeline to
        up_to_stage : int
            SpimStage up to which the pipeline should be applied

        Returns
        -------
        Spim
            the processed Spim at stage `up_to_stage`
        """
        first_outdated_stage = -1

        # find first stage that is outdated
        for st, process in sorted(self.process_stage_dict.items()):
            if process.outdated:
                first_outdated_stage = st
                break

        if first_outdated_stage >= 0:
            if first_outdated_stage < up_to_stage:
                return self.apply_from_stage_to_stage(
                    spim, first_outdated_stage, up_to_stage)

        # nothing is outdated or outdated stage is past up_to_stage
        if up_to_stage > spim.stage:
            return self.apply_from_stage_to_stage(spim,
                                                  spim.stage,
                                                  up_to_stage)
        else:
            return spim.get_at_stage(up_to_stage)

    def save(self, target_path):
        """Store the pipeline including process paramaters for
        later use.

        Parameters
        ----------
        target_path : str
            path of the file to store the pipeline to

        Notes
        -----
        This creates a file that is also suitable for batch process
        and parallelization.

        See also
        --------
        Pipeline.from_file
            Use `Pipeline.from_file` to restore the pipeline object
            from storage
        """
        with open(target_path, "wb") as dill_file:
            dill.dump(self, dill_file)

    @classmethod
    def from_file(cls, filepath):
        """Restore a pipeline from a file

        Parameters
        ----------
        filepath : str
            filepath of the pipeline file

        Returns
        -------
        Pipepline
            the restored pipeline object
        """

        with open(filepath, "rb") as dill_file:
            restored_pipe = dill.load(dill_file)
        return restored_pipe

    def __str__(self):
        out = []
        for stage in sorted(self.process_stage_dict.keys()):
            process = self.process_stage_dict[stage]
            out += [str(process)]
        return "".join(out)
