import dill


class Pipeline(object):
    def __init__(self, processes):
        self.process_stage_dict = dict([(p.input_stage, p) for p in processes])

    @property
    def processes(self):
        return self.process_stage_dict.values()

    def apply_from_stage_to_stage(self, spim, from_stage, to_stage):
        """recursively applies all pipeline-processes up to the given stage"""
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
            print("return at stage %s" % outspim.stage)
            return outspim

    def _maxstage(self):
        return max(self.process_stage_dict.keys())+1

    def apply_all_steps(self, spim):
        """return a spim with all processes within this pipeline applied in the defined order"""
        minstage = min(self.process_stage_dict.keys())
        maxstage = self._maxstage()
        return self.apply_from_stage_to_stage(spim, minstage, maxstage)

    def apply_at_stage(self, spim):
        """applies all steps following the stage of the spim"""
        startstage = spim.stage
        maxstage = self._maxstage()
        return self.apply_from_stage_to_stage(spim, startstage, maxstage)

    def apply_outdated_up_to_stage(self, spim, up_to_stage):
        """applies all processes since the first outdated one on spim up to a given stage
        if no process is outdated or if the outdated stage is past up_to_stage, spim is processed up to up_to_stage,
        or a predecessor is returned at up_to_stage
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
            return self.apply_from_stage_to_stage(spim, spim.stage, up_to_stage)
        else:
            return spim.get_at_stage(up_to_stage)

    def save(self, target_path):
        """store the pipeline including process paramaters for later use (suitable for batch process and parallelization)
        use from_file to load the pipeline"""
        with open(target_path, "wb") as dill_file:
            dill.dump(self, dill_file)

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, "rb") as dill_file:
            restored_pipe = dill.load(dill_file)
        return restored_pipe
