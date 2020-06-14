from torch import no_grad
        
    
        
class EmptyModifier():
    
    def __enter__(self):
        pass#rint('-> with grad in')#ass
    
    def __exit__(self, *args):
        pass#rint('-> with grad out')#ass