import unittest

from pkg_resources import resource_filename
import numpy as np
import numpy.testing

from ..defaults import default_pipeline
from ..spim import Spim, SpimStage


class TestSpimLifecycle(unittest.TestCase):
    def test_portable_pipeline(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/test_save.pipe")

        # store
        mypipe.save(filename)

        # restore
        loaded_pipe = mypipe.from_file(filename)

        # restored dicts have the same length
        self.assertTrue(len(loaded_pipe.process_stage_dict) ==
                        len(mypipe.process_stage_dict))

        restored_processes = loaded_pipe.process_stage_dict.values()
        original_processes = mypipe.process_stage_dict.values()

        # # restored parameters are the same
        for restored, original_process in zip(restored_processes,
                                              original_processes):
            restored_params = restored.parameters.parameters
            orig_params = original_process.parameters.parameters

            for p_r, p_o in zip(restored_params, orig_params):
                self.assertTrue(p_r.name == p_o.name)
                self.assertTrue(p_r.value == p_o.value)
                self.assertTrue(p_r.type == p_o.type)

        # test processes are the same
        sample_grey_image = (np.random.rand(500, 500)*255).astype(np.uint8)

        preproc_mypipe = mypipe.process_stage_dict[SpimStage.converted].apply(
            sample_grey_image)
        preproc_loaded = mypipe.process_stage_dict[SpimStage.converted].apply(
            sample_grey_image)

        numpy.testing.assert_array_equal(preproc_mypipe, preproc_loaded)

    def test_pipeline_apply_all_stages(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")
        s0 = Spim.from_file(filename)

        s_final = mypipe.apply_all_steps(s0)

        self.assertEqual(s_final.stage, SpimStage.analyzed)

    def test_pipeline_intermediate_stages(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")
        s0 = Spim.from_file(filename, cached=True)

        stages = mypipe.process_stage_dict.keys()

        for stage in stages:
            self.assertTrue(stage in range(max(stages)+1))

        tempspim = s0

        for stage in stages:
            before_stage = tempspim.stage
            after_stage = before_stage + 1

            process = mypipe.process_stage_dict[before_stage]
            tempspim = tempspim.func_at_stage(before_stage)(process)

            self.assertEqual(tempspim.stage, after_stage)

    def test_pipeline_apply_from_to_stage(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")
        s0 = Spim.from_file(filename, cached=True)

        for intermediate_stage in range(SpimStage.analyzed):
            # apply up to all the different stages
            # and check stage of intermediate result
            s_temp = mypipe.apply_from_stage_to_stage(
                s0,
                from_stage=s0.stage,
                to_stage=intermediate_stage)
            self.assertEqual(s_temp.stage, intermediate_stage)

    def test_pipeline_outdated(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")
        s0 = Spim.from_file(filename, cached=True)

        # apply all stages
        s1_final = mypipe.apply_all_steps(s0)

        # change parameter in one process
        binarize_process = mypipe.process_stage_dict[SpimStage.postprocessed]
        assert binarize_process.input_stage == SpimStage.postprocessed

        # set one process to outdated
        binarize_process.outdated = True

        # apply_outdated_up_to_stage
        # should redo everything after binarization
        last_unchanged_stage = SpimStage.binarized-1
        final_stage = SpimStage.analyzed

        s1_final_updated = mypipe.apply_outdated_up_to_stage(
            s1_final, final_stage)

        s_unchanged_ancestor = s1_final.get_at_stage(last_unchanged_stage)

        self.assertEqual(s1_final_updated.stage, final_stage)

        print(s1_final_updated.predecessors)
        # common predecessor should be the same
        self.assertIs(s1_final_updated.get_at_stage(last_unchanged_stage),
                      s_unchanged_ancestor)
