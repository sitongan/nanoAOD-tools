from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
import os
import numpy as np

import correctionlib

class puWeightProducer(Module):
    def __init__(self,
                 era,
                 name="puWeight",
                 norm=True,
                 verbose=False,
                 nvtx_var="Pileup_nTrueInt",
                 doSysVar=True
     ):
        self.inputFilePath = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/"
        eramap = {"UL2016_preVFP": "2016preVFP_UL",
           "UL2016":"2016postVFP_UL",
           "UL2017":"2017_UL",
           "UL2018":"2018_UL"}
        self.era = eramap[era]
        self.name = name
        self.norm = norm
        self.verbose = verbose
        self.nvtxVar = nvtx_var
        self.doSysVar = doSysVar
        
    def beginJob(self):
        clibfile_list = list(correctionlib.CorrectionSet.from_file(self.inputFilePath + self.era +"/" + "puWeights.json.gz").values())
        assert(len(clibfile_list) == 1)
        self.calibration = clibfile_list[0]
        
    def endJob(self):
        pass
    
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch(self.name, "F")
        if self.doSysVar:
            self.out.branch(self.name + "Up", "F")
            self.out.branch(self.name + "Down", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        if hasattr(event, self.nvtxVar):
            nvtx = int(getattr(event, self.nvtxVar))
            weight = self.calibration.evaluate(np.array([nvtx]), 'nominal')
            
            if self.doSysVar:
                weight_plus = self.calibration.evaluate(np.array([nvtx]), 'up')
                weight_minus = self.calibration.evaluate(np.array([nvtx]), 'down')
        else:
            weight = 1
        self.out.fillBranch(self.name, weight)
        if self.doSysVar:
            self.out.fillBranch(self.name + "Up", weight_plus)
            self.out.fillBranch(self.name + "Down", weight_minus)
        return True
    
puWeight2016_UL = lambda: puWeightProducer("UL2016",
                                           verbose=False,
                                           doSysVar=True)

puWeight2016_UL_preVFP = lambda: puWeightProducer("UL2016_preVFP",
                                           verbose=False,
                                           doSysVar=True)

puWeight2017_UL = lambda: puWeightProducer("UL2017",
                                           verbose=False,
                                           doSysVar=True)

puWeight2018_UL = lambda: puWeightProducer("UL2018",
                                           verbose=False,
                                           doSysVar=True)
