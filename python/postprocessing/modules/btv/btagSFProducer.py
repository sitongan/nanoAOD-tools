from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import os
import numpy as np

import correctionlib

class btagSFProducer(Module):
    """Calculate btagging scale factors
    """

    def __init__(
            self, era, algo='deepJet', selectedWPs=['M', 'shape_corr'],
            sfFileName=None, verbose=0, jesSystsForShape=["jes"]
    ):
        #algo = 'deepJet' or 'deepCSV'
        
        eramap = {"UL2016_preVFP": "2016preVFP_UL",
                   "UL2016":"2016postVFP_UL",
                   "UL2017":"2017_UL",
                   "UL2018":"2018_UL"}
        
        
        self.era = eramap[era]
        self.algo = algo
        self.selectedWPs = selectedWPs
        self.verbose = verbose
        self.jesSystsForShape = jesSystsForShape
        # CV: Return value of BTagCalibrationReader::eval_auto_bounds() is zero
        # in case jet abs(eta) > 2.4 !!
        self.max_abs_eta = 2.4
        # define measurement type for each flavor
        self.inputFilePath = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/"
        self.inputFileName = sfFileName
        self.measurement_types = None
        self.supported_wp = ["L", "M", "T", "shape_corr"]
       

        # define systematic uncertainties
        
        self.central_and_systs = ["central", "up", "down"]

        self.systs_shape_corr_bjet = []
        for syst in ['lf', 'hf',
                     'hfstats1', 'hfstats2',
                     'lfstats1', 'lfstats2'] + self.jesSystsForShape:
            self.systs_shape_corr_bjet.append("up_%s" % syst)
            self.systs_shape_corr_bjet.append("down_%s" % syst)
        
        self.central_and_systs_shape_corr_common = ["central"]
        #for syst in self.jesSystsForShape:
        #    self.central_and_systs_shape_corr_common.append("up_%s" % syst)
        #    self.central_and_systs_shape_corr_common.append("down_%s" % syst)
        
        self.systs_shape_corr_cjet = ['up_cferr1','down_cferr1', 'up_cferr2','down_cferr2']

        self.branchNames_central_and_systs = {}
        for wp in self.selectedWPs:
            branchNames = {}
            if wp == 'shape_corr':
                central_and_systs = self.central_and_systs_shape_corr_common + self.systs_shape_corr_bjet + self.systs_shape_corr_cjet
                baseBranchName = 'Jet_btagSF_{}_shape'.format(self.algo.lower())
            else:
                central_and_systs = self.central_and_systs
                baseBranchName = 'Jet_btagSF_{}_{}'.format(self.algo.lower(), wp)
            for central_or_syst in central_and_systs:
                if central_or_syst == "central":
                    branchNames[central_or_syst] = baseBranchName
                else:
                    branchNames[central_or_syst] = baseBranchName + \
                        '_' + central_or_syst
            self.branchNames_central_and_systs[wp] = branchNames
            
        self.discr = None
        if self.algo == "deepCSV":
            self.discr = "btagDeepB"
        elif self.algo == "deepJet":
            self.discr = "btagDeepFlavB"
        else:
            raise ValueError("ERROR: Invalid algorithm '%s'!" % self.algo)
            
    def beginJob(self):
        
        self.calibration = correctionlib.CorrectionSet.from_file(self.inputFilePath + self.era +"/" + "btagging.json.gz")
        
    def endJob(self):
        pass
    

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        for central_or_syst in list(self.branchNames_central_and_systs.values()):
            for branch in list(central_or_syst.values()):
                self.out.branch(branch, "F", lenVar="nJet")
               
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        jets = Collection(event, "Jet")
        jet_flav = np.array([jet.hadronFlavour for jet in jets])
        jet_pt = np.array([jet.pt for jet in jets])
        jet_eta = []
        epsilon = 1.e-3
        for jet in jets:
            if abs(jet.eta) >= +self.max_abs_eta:
                jet_eta.append(+self.max_abs_eta - epsilon)
            else:
                jet_eta.append(abs(jet.eta))
        jet_eta = np.array(jet_eta)
        
        for wp in self.selectedWPs:
            isShape = (wp == 'shape_corr')
            if isShape:
                c_jets = np.where(jet_flav == 4)
                non_c_jets = np.where(jet_flav != 4)
                jet_disc = np.array([getattr(jet, self.discr) for jet in jets])
                
                for central_or_syst in self.central_and_systs_shape_corr_common:
                    sfs = self.calibration[self.algo + "_shape"].evaluate(central_or_syst, jet_flav, jet_eta, jet_pt, jet_disc)
                    sfs[np.where(sfs < 0.01)] = 1.0
                    self.out.fillBranch(self.branchNames_central_and_systs[wp][central_or_syst], list(sfs))
                    
                for central_or_syst in self.systs_shape_corr_bjet:
                    sfs = np.ones_like(jet_pt)
                    sfs[c_jets] = self.calibration[self.algo + "_shape"].evaluate("central", jet_flav[c_jets], jet_eta[c_jets], jet_pt[c_jets], jet_disc[c_jets])
                    sfs[non_c_jets] = self.calibration[self.algo + "_shape"].evaluate(central_or_syst, jet_flav[non_c_jets], jet_eta[non_c_jets], jet_pt[non_c_jets], jet_disc[non_c_jets])
                    sfs[np.where(sfs < 0.01)] = 1.0
                    self.out.fillBranch(self.branchNames_central_and_systs[wp][central_or_syst], list(sfs))
                    
                for central_or_syst in self.systs_shape_corr_cjet:
                    sfs = np.ones_like(jet_pt)
                    sfs[c_jets] = self.calibration[self.algo + "_shape"].evaluate(central_or_syst, jet_flav[c_jets], jet_eta[c_jets], jet_pt[c_jets], jet_disc[c_jets])
                    sfs[non_c_jets] = self.calibration[self.algo + "_shape"].evaluate("central", jet_flav[non_c_jets], jet_eta[non_c_jets], jet_pt[non_c_jets], jet_disc[non_c_jets])
                    sfs[np.where(sfs < 0.01)] = 1.0
                    self.out.fillBranch(self.branchNames_central_and_systs[wp][central_or_syst], list(sfs))
            
            else:
                for central_or_syst in self.central_and_systs:
                    sfs = np.ones_like(jet_pt)
                    light_jets = np.where(jet_flav == 0)
                    non_light_jets = np.where(jet_flav != 0)
                    sfs[light_jets] = self.calibration[self.algo + "_incl"].evaluate(central_or_syst, wp, jet_flav[light_jets], jet_eta[light_jets], jet_pt[light_jets])
                    sfs[non_light_jets] = self.calibration[self.algo + "_comb"].evaluate(central_or_syst, wp, jet_flav[non_light_jets], jet_eta[non_light_jets], jet_pt[non_light_jets])
                    sfs[np.where(sfs < 0.01)] = 1.0
                    self.out.fillBranch(self.branchNames_central_and_systs[wp][central_or_syst], list(sfs))
                
        return True
    
    
btagSF2016_UL_preVFP = lambda: btagSFProducer("UL2016_preVFP")
btagSF2016_UL_postVFP = lambda: btagSFProducer("UL2016")
btagSF2017_UL = lambda: btagSFProducer("UL2017")
btagSF2018_UL = lambda: btagSFProducer("UL2018")
                    
                
                
                