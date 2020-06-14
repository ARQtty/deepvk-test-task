from os.path import getsize
from torchaudio import load



class CPCAudioPredicates:
    '''
    Papers based on CPC model requires some audio parameter limits. Its
    limits implemented here as predicates for filtering audio dataset items
    '''
    @staticmethod
    def longer_then_sample_length(path):
        '''
        Returns True if audio is longer then 1.28sec that is required for
        CPC models from papers
        '''
        filesize = getsize(path)

        # we should check if audio is longer then 1,28sec
        # it's about 40kB

        # sampling rate is always 16k in the LibriSpeech dataset, so if
        # file is larger then 50kB, it is defenetly longer then 1.28sec
        if filesize > 50000:
            return True

        # else we check file length
        audio, bitrate = load(path)
        return audio.size(1) > 20480
