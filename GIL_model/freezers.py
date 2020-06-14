
class SimultaneousFreezer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def maybe_freeze(self, step):
        pass



class IterativeFreezer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def maybe_freeze(self, step):
        for ix in range(5):
            key = 'module_%d' % ix

            if self.config.train.unfreezing[key] == step:
                # unfreeze block to be learned
                self.model.unfreeze_block(ix)

                # freeze other blocks
                for i in range(len(self.model.gim_modules)):
                    if i != ix:
                        self.model.freeze_block(i)

                # report
                print(' [Freezer] Module %d unfreezed' % ix)
