class BaseFreezer:
    def maybe_freeze(self, step):
        raise NotImplementedError



class IterativeFreezer(BaseFreezer):
    '''Freezer that allows gradient decent optimizing in only one module'''
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def maybe_freeze(self, step):
        '''Depends on the step of training can freeze or unfreeze modules'''
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



class SimultaneousFreezer(BaseFreezer):
    '''Empty freezer, all modules will learn together with it'''
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def maybe_freeze(self, step):
        pass
