from PhysicsTools.NanoAODTools.postprocessing.tools import matchObjectCollection, matchObjectCollectionMultiple
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
import ROOT
import math
import os
import tarfile
import tempfile
import shutil
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
import correctionlib._core as core

class jetSmearer(Module):
    def __init__(
            self,
            era,
            globalTag,
            jerTag,
            jetType,
            ):

        # -------------------------------------------------------------------
        # CV: globalTag and jetType not yet used, as there is no consistent
        # set of txt files for JES uncertainties and JER scale factors and
        # uncertainties yet
        # --------------------------------------------------------------------

        # read jet energy resolution (JER) and JER scale factors and uncertainties
        # (the txt files were downloaded from https://github.com/cms-jet/JRDatabase/tree/master/textFiles/ )
        # Text files are now tarred so must extract first

        # initialize random number generator
        # (needed for jet pT smearing)
        self.era = era
        self.jecTag = globalTag
        self.jerTag = jerTag
        
        self.jetType = jetType
        #if "AK8" in jetType:
        if True:
            if "16" in era or "17" in era or "18" in era: #run2
                self.run=2
                self.jetType="AK4PFchs" #only this is available for run2 
            elif "22" in era or "23" in era:
                self.run=3
                self.jetType="AK4PFPuppi"
            
        
        self.rnd = ROOT.TRandom3(12345)
        self.pogdir = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
        #jmr_vals=[1.09, 1.14, 1.04]?
        

    def beginJob(self):
        # initialize JER scale factors and uncertainties
        # (cf. PhysicsTools/PatUtils/interface/SmearedJetProducerT.h )
        #fname_jersmear = os.path.join(self.pogdir, "POG/JME/jer_smear.json.gz")
        #print("\n jetSmearer Loading JSON file: {}".format(fname_jersmear))
        #self.cset_jersmear = core.CorrectionSet.from_file(fname_jersmear)
        
        #used for jet sf and pt resolution
        if "AK4" in self.jetType:
            fname = os.path.join(self.pogdir, f"POG/JME/{self.era}/jet_jerc.json.gz")
            print("\n jetSmearer Loading JSON file: {}".format(fname))
            self.cset = core.CorrectionSet.from_file(os.path.join(fname))
        elif "AK8" in self.jetType:
        # AK8
            #fname_ak8 = os.path.join(__this_dir__, f"POG/JME/{self.era}/fatJet_jerc.json.gz")
            fname_ak8 = os.path.join(self.pogdir, f"POG/JME/{self.era}/jet_jerc.json.gz")
            print("\n jetSmearer Loading JSON file: {}".format(fname_ak8))
            self.cset = core.CorrectionSet.from_file(os.path.join(fname_ak8))

    def endJob(self):
        pass

    def setSeed(self, event):
        """Set seed deterministically."""
        # (cf. https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/PatUtils/interface/SmearedJetProducerT.h)
        runnum = int(event.run) << 20
        luminum = int(event.luminosityBlock) << 10
        evtnum = event.event
        jet0eta = int(event.Jet_eta[0] / 0.01 if event.nJet > 0 else 0)
        seed = 1 + runnum + evtnum + luminum + jet0eta
        self.rnd.SetSeed(seed)

    def getSmearedJetPt(self, jet, genJet, rho):
        (jet_pt_nomVal, jet_pt_jerUpVal,
         jet_pt_jerDownVal) = self.getSmearValsPt(jet, genJet, rho)
        return (jet_pt_nomVal * jet.pt, jet_pt_jerUpVal * jet.pt,
                jet_pt_jerDownVal * jet.pt)

    def getSmearValsPt(self, jetIn, genJetIn, rho):

        if hasattr(jetIn, "p4"):
            jet = jetIn.p4()
        else:
            jet = jetIn
        if hasattr(genJetIn, "p4"):
            genJet = genJetIn.p4()
        else:
            genJet = genJetIn

        # --------------------------------------------------------------------------------------------
        # CV: Smear jet pT to account for measured difference in JER between data and simulation.
        #     The function computes the nominal smeared jet pT simultaneously with the JER up and down shifts,
        #     in order to use the same random number to smear all three (for consistency reasons).
        #
        #     The implementation of this function follows PhysicsTools/PatUtils/interface/SmearedJetProducerT.h
        #
        # --------------------------------------------------------------------------------------------

        if not (jet.Perp() > 0.):
            print("WARNING: jet pT = %1.1f !!" % jet.Perp())
            return (jet.Perp(), jet.Perp(), jet.Perp())

        # --------------------------------------------------------------------------------------------
        # CV: define enums needed to access JER scale factors and uncertainties
        #    (cf. CondFormats/JetMETObjects/interface/JetResolutionObject.h)
        enum_nominal = 'nom'
        enum_shift_up = 'up'
        enum_shift_down = 'down'
        # --------------------------------------------------------------------------------------------

        jet_pt_sf_and_uncertainty = {}
        for enum_central_or_shift in [
                enum_nominal, enum_shift_up, enum_shift_down
        ]:
            key = "{}_{}_{}".format(self.jerTag, "ScaleFactor", self.jetType)
            #print(key)
            sf = self.cset[key]
            if self.run == 2:
                inputs = [jet.Eta(), enum_central_or_shift]
            elif self.run == 3:
                inputs = [jet.Eta(),jet.Perp(), enum_central_or_shift]
            #print(inputs)
            jersf_value = sf.evaluate(*inputs)
            jet_pt_sf_and_uncertainty[
                enum_central_or_shift] = jersf_value

        smear_vals = {}
        if genJet:
            for central_or_shift in [
                    enum_nominal, enum_shift_up, enum_shift_down
            ]:
                #
                # Case 1: we have a "good" generator level jet matched to
                # the reconstructed jet
                #
                dPt = jet.Perp() - genJet.Perp()
                smearFactor = 1. + \
                    (jet_pt_sf_and_uncertainty[central_or_shift] - 1.) * dPt /  jet.Perp()
                smear_vals[central_or_shift] = smearFactor
        else:
            key = "{}_{}_{}".format(self.jerTag, "PtResolution", self.jetType)
            sf = self.cset[key]
            inputs = [jet.Eta(), jet.Perp(), rho]
            jet_pt_resolution = sf.evaluate(*inputs)

            rand = self.rnd.Gaus(0, jet_pt_resolution)
            for central_or_shift in [
                    enum_nominal, enum_shift_up, enum_shift_down
            ]:
                if jet_pt_sf_and_uncertainty[central_or_shift] > 1.:
                    #
                    # Case 2: we don't have a generator level jet. Smear jet
                    # pT using a random Gaussian variation
                    #
                    smearFactor = 1. + rand * \
                        math.sqrt(
                            jet_pt_sf_and_uncertainty[central_or_shift]**2 - 1.)
                else:
                    #
                    # Case 3: we cannot smear this jet, as we don't have a
                    # generator level jet and the resolution in data is better
                    # than the resolution in the simulation, so we would need
                    # to randomly "unsmear" the jet, which is impossible
                    #
                    smearFactor = 1.
                smear_vals[central_or_shift] = smearFactor

        for central_or_shift in [enum_nominal, enum_shift_up, enum_shift_down]:
            # check that smeared jet energy remains positive,
            # as the direction of the jet would change ("flip")
            # otherwise - and this is not what we want
            if smear_vals[central_or_shift] * jet.E() < 1.e-2:
                smear_vals[central_or_shift] = 1.e-2 / jet.E()

        return (smear_vals[enum_nominal], smear_vals[enum_shift_up],
                smear_vals[enum_shift_down])

    def getSmearValsM(self, jetIn, genJetIn):

        # ---------------------------------------------------------------------
        # LC: Procedure outline in: https://twiki.cern.ch/twiki/bin/view/Sandbox/PUPPIJetMassScaleAndResolution
        # ---------------------------------------------------------------------

        if hasattr(jetIn, "p4"):
            jet = jetIn.p4()
        else:
            jet = jetIn
        if hasattr(genJetIn, "p4"):
            genJet = genJetIn.p4()
        else:
            genJet = genJetIn

        # ---------------------------------------------------------------------
        # CV: Smear jet m to account for measured difference in JER between 
        # data and simulation. The function computes the nominal smeared jet m
        # simultaneously with the JER up and down shifts, in order to use the
        # same random number to smear all three (for consistency reasons).
        #
        # The implementation of this function follows:
        # PhysicsTools/PatUtils/interface/SmearedJetProducerT.h
        #
        # ---------------------------------------------------------------------

        if not (jet.M() > 0.):
            print("WARNING: jet m = %1.1f !!" % jet.M())
            return (jet.M(), jet.M(), jet.M())

        # ---------------------------------------------------------------------
        # CV: define enums needed to access JER scale factors and uncertainties
        #    (cf. CondFormats/JetMETObjects/interface/JetResolutionObject.h)
        enum_nominal = 'nom'
        enum_shift_up = 'up'
        enum_shift_down = 'down'
        # ---------------------------------------------------------------------

        jet_m_sf_and_uncertainty = {}
        for enum_central_or_shift in [
                enum_nominal, enum_shift_up, enum_shift_up
        ]:
            key = "{}_{}_{}".format(self.jerTag, "ScaleFactor", self.jetTpe)
            sf = self.cset[key]
            inputs = [jet.Eta(), enum_central_or_shift]
            jersf_value = sf.evaluate(*inputs)
            jet_m_sf_and_uncertainty[
                enum_central_or_shift] = jersf_value

        smear_vals = {}
        if genJet:
            for central_or_shift in [
                    enum_nominal, enum_shift_up, enum_shift_down
            ]:
                #
                # Case 1: we have a "good" generator level jet matched to the
                # reconstructed jet
                #
                dM = jet.M() - genJet.M()
                smearFactor = 1. + \
                    (jet_m_sf_and_uncertainty[central_or_shift] - 1.) * dM / jet.M()
                # check that smeared jet energy remains positive,
                # as the direction of the jet would change ("flip")
                # otherwise - and this is not what we want
                if (smearFactor * jet.M()) < 1.e-2:
                    smearFactor = 1.e-2
                smear_vals[central_or_shift] = smearFactor

        else:
            
            # Get mass resolution
            key = "{}_{}_{}".format(self.jerTag, "PtResolution", self.jetType)
            sf = self.cset[key]
            inputs = [jet.Eta(), jet.Perp(), rho]
            jet_m_resolution = sf.evaluate(*inputs)
            
            #if abs(jet.Eta()) <= 1.3:
            #    jet_m_resolution = self.puppisd_resolution_cen.Eval(jet.Pt())
            #else:
            #    jet_m_resolution = self.puppisd_resolution_for.Eval(jet.Pt())
            
            #only ptresolution is available. 
            #https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py
            #instructed to use the same smear factor as pt for mass
            
            rand = self.rnd.Gaus(0, jet_m_resolution)
            for central_or_shift in [
                    enum_nominal, enum_shift_up, enum_shift_down
            ]:
                if jet_m_sf_and_uncertainty[central_or_shift] > 1.:
                    #
                    # Case 2: we don't have a generator level jet. Smear jet m
                    # using a random Gaussian variation
                    #
                    smearFactor = rand * \
                        math.sqrt(
                            jet_m_sf_and_uncertainty[central_or_shift]**2 - 1.)
                else:
                    #
                    # Case 3: we cannot smear this jet, as we don't have a
                    # generator level jet and the resolution in data is better
                    # than the resolution in the simulation, so we would need
                    # to randomly "unsmear" the jet, which is impossible
                    #
                    smearFactor = 1.

                # check that smeared jet energy remains positive,
                # as the direction of the jet would change ("flip")
                # otherwise - and this is not what we want
                if (smearFactor * jet.M()) < 1.e-2:
                    smearFactor = 1.e-2
                smear_vals[central_or_shift] = smearFactor

        return (smear_vals[enum_nominal], smear_vals[enum_shift_up],
                smear_vals[enum_shift_down])
