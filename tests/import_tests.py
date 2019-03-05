class TestImports(object):
    def test_imports(self):
        from cw_nets.data import datagen
        from cw_nets.data import io
        from cw_nets.data import transform
        from cw_nets.models.configs import callbacks, infer, io, \
            losses, metrics, train, zoo
        import cw_nets.utils as utils
