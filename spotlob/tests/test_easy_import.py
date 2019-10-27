import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        import spotlob

        pipe = spotlob.default_pipeline()
        spim = spotlob.Spim(None,
                            dict(),
                            spotlob.SpimStage.new,
                            cached=False,
                            predecessors=dict())
