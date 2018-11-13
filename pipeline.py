class Pipeline(object):
    def __init__(self, processes):
        self.process_stage_dict = dict([(p.input_stage, p) for p in processes])

    @property
    def processes(self):
        return self.process_stage_dict.values()

    def apply_from_stage_to_stage(self, spim, from_stage, to_stage):
        """recursively applies all pipeline-processes up to the given stage"""
        if from_stage >= to_stage:
            return spim
        else:
            intermediate_spim = self.apply_from_stage_to_stage(
                spim, from_stage, to_stage-1)
            last_process = self.process_stage_dict[to_stage-1]
            return intermediate_spim.do_process_at_stage(last_process)

    def _maxstage(self):
        return max(self.process_stage_dict.keys())+1

    def apply_all_steps(self, spim):
        minstage = min(self.process_stage_dict.keys())
        maxstage = self._maxstage()
        return self.apply_from_stage_to_stage(spim, minstage, maxstage)

    def apply_at_stage(self, spim):
        """applies all steps following the stage of the spim"""
        startstage = spim.stage
        maxstage = self._maxstage()
        return self.apply_from_stage_to_stage(spim, startstage, maxstage)

    def apply_outdated_up_to_stage(self, spim, up_to_stage):
        """applies all processes since the first outdated one on spim up to a given stage"""
        first_outdated_stage = -1

        # find first stage that is outdated
        for st, process in self.process_stage_dict.items():
            if process.outdated:
                first_outdated_stage = st
                break

        if first_outdated_stage >= 0:
            return self.apply_from_stage_to_stage(
                spim, first_outdated_stage, up_to_stage)
        else:
            # nothing is outdated, don't do anything
            return spim
