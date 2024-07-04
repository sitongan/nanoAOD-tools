import ROOT
import os
import types
from math import *
from PhysicsTools.HeppyCore.utils.deltar import *
import correctionlib._core as core

class JetReCalibrator:
    def __init__(
        self,
        era,
        globalTag,
        jetFlavour,
        doResidualJECs,
        upToLevel=3,
     ):
        """Create a corrector object that reads the payloads from the text
        dumps of a global tag under CMGTools/RootTools/data/jec (see the
        getJec.py there to make the dumps). It will apply the L1,L2,L3 and
        possibly the residual corrections to the jets. If configured to do so,
        it will also compute the type1 MET corrections."""
        self.era = era
        self.globalTag = globalTag
        self.jetType = jetFlavour
        self.doResidualJECs = doResidualJECs
        self.upToLevel = upToLevel
        self.pogdir = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
        levels = {1: "L1FastJet", 2:'L2Relative', 3:'L3Absolute'}
        #['', '', '', '']
        uptolvls = list(range(1, upToLevel+1))
        if 'Puppi' in self.jetType and 1 in uptolvls:
            uptolvls.remove(1)
        if doResidualJECs and upToLevel == 3:
            self.level = ["L1L2L3Res"]
            #Puppijet doesn't have L1 correction. But using 'L1L2L3Res' is correct - tested.
        else:
            self.level=[]
            for i in uptolvls:
                self.level.append(levels[i])
            if doResidualJECs:
                self.level.append('L2L3Residual')
            
        
        if "AK4" in self.jetType:
            fname = os.path.join(self.pogdir, f"POG/JME/{self.era}/jet_jerc.json.gz")
            print("\n JetReCalibrator Loading JSON file: {}".format(fname))
            self.cset = core.CorrectionSet.from_file(os.path.join(fname))
        elif "AK8" in self.jetType:
            fname_ak8 = os.path.join(self.pogdir, f"POG/JME/{self.era}/fatJet_jerc.json.gz")
            print("\n JetReCalibrator Loading JSON file: {}".format(fname_ak8))
            self.cset = core.CorrectionSet.from_file(os.path.join(fname_ak8))

    def __deprecated_getCorrection(self, jet, rho, delta=0, corrector=None):
        if not corrector:
            corrector = self.JetCorrector
        if corrector != self.JetCorrector and delta != 0:
            raise RuntimeError('Configuration not supported')
        corrector.setJetPhi(jet.phi)
        corrector.setJetEta(jet.eta)
        corrector.setJetPt(jet.pt * (1. - jet.rawFactor))
        corrector.setJetA(jet.area)
        corrector.setRho(rho)
        corr = corrector.getCorrection()
        if delta != 0:
            if not self.JetUncertainty:
                raise RuntimeError(
                    "Jet energy scale uncertainty shifts requested, but not available"
                )
            self.JetUncertainty.setJetPhi(jet.phi)
            self.JetUncertainty.setJetEta(jet.eta)
            self.JetUncertainty.setJetPt(corr * jet.pt * (1. - jet.rawFactor))
            try:
                jet.jetEnergyCorrUncertainty = self.JetUncertainty.getUncertainty(
                    True)
            except RuntimeError as r:
                print(
                    "Caught %s when getting uncertainty for jet of pt %.1f, eta %.2f\n"
                    % (r, corr * jet.pt * (1. - jet.rawFactor), jet.eta))
                jet.jetEnergyCorrUncertainty = 0.5
            corr *= max(0, 1 + delta * jet.jetEnergyCorrUncertainty)
        return corr

    def correct(self,
                jet,
                rho):
        """Corrects a jet energy (optionally shifting it also by delta times
        the JEC uncertainty)

       If addCorr, set jet.corr to the correction.
       If addShifts, set also the +1 and -1 jet shifts 

       The metShift vector will accumulate the x and y changes to the MET
       from the JEC, i.e. the  negative difference between the new and old jet
       momentum, for jets eligible for type1 MET corrections, and after
       subtracting muons. The pt cut is applied to the new corrected pt. This
       shift can be applied on top of the *OLD TYPE1 MET*, but only if there
       was no change in the L1 corrections nor in the definition of the type1
       MET (e.g. jet pt cuts).

        """
        raw = 1. - jet.rawFactor
        corr = 1.0
        for l in self.level:
                key = "{}_{}_{}".format(self.globalTag, l, self.jetType)
                if l == "L1L2L3Res":
                    sf = self.cset.compound[key]
                else:
                    sf = self.cset[key]
                if l == "L1L2L3Res" or l=="L1FastJet":
                    inputs=[jet.area, jet.eta, jet.pt * raw, rho]
                else:
                    inputs=[jet.eta, jet.pt * raw]
                corr *= float(sf.evaluate(*inputs))
        #corr = self.getCorrection(jet, rho, delta)
        
        #for puppi and uptoLevel==1, len(self.level)==0 and corr==1.0 (no correction, return raw jet pt and mass)
        if corr <= 0:
            return (jet.pt, jet.mass)
        newpt = jet.pt * raw * corr
        newmass = jet.mass * raw * corr
        return (newpt, newmass)
