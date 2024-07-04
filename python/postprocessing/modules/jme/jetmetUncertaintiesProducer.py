from PhysicsTools.NanoAODTools.postprocessing.modules.jme.JetReCalibrator import JetReCalibrator
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetSmearer import jetSmearer
from PhysicsTools.NanoAODTools.postprocessing.tools import matchObjectCollection, matchObjectCollectionMultiple
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
import ROOT
import math
import os
import re
import tarfile
import tempfile
import shutil
import numpy as np
import itertools
import correctionlib._core as core
ROOT.PyConfig.IgnoreCommandLineOptions = True


class jetmetUncertaintiesProducer(Module):
    def __init__(self,
                 era,
                 globalTag,
                 jesUncertainties=["Total"],
                 jetType="AK4PFchs",
                 metBranchName="MET",
                 jerTag="",
                 isData=False,
                 applySmearing=True,
                 splitJER=False,
                 saveMETUncs=['T1', 'T1Smear']
     ):

        if "AK8" in jetType:
            if "16" in era or "17" in era or "18" in era: #run2
                self.replacement_jetType="AK4PFchs" #only this is available for run2 
            elif "22" in era or "23" in era:
                self.replacement_jetType="AK4PFPuppi"
        
        
        
        self.jecTag = globalTag
        self.jerTag = jerTag
        self.era = era
        self.isData = isData
        # if set to true, Jet_pt_nom will have JER applied. not to be
        # switched on for data.
        self.applySmearing = applySmearing if not isData else False
        self.splitJER = splitJER
        if self.splitJER:
            self.splitJERIDs = list(range(6))
        else:
            self.splitJERIDs = [""]  # "empty" ID for the overall JER
        self.metBranchName = metBranchName
        self.rhoBranchName = "fixedGridRhoFastjetAll"
        # --------------------------------------------------------------------
        # CV: globalTag and jetType not yet used in the jet smearer, as there
        # is no consistent set of txt files for JES uncertainties and JER scale
        # factors and uncertainties yet
        # --------------------------------------------------------------------

        self.jesUncertainties = jesUncertainties

        # Calculate and save uncertainties on T1Smear MET if this flag is set
        # to True. Otherwise calculate and save uncertainties on T1 MET
        self.saveMETUncs = saveMETUncs

        # smear jet pT to account for measured difference in JER between data
        # and simulation.

        self.jetSmearer = jetSmearer(era, globalTag, jerTag, jetType)
        self.pogdir = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
        
        #used for jer resolution and jes uncertainty
        self.jetType = jetType
        if "AK4" in jetType:
            self.jetBranchName = "Jet"
            self.genJetBranchName = "GenJet"
            self.genSubJetBranchName = None
            fname = os.path.join(self.pogdir, f"POG/JME/{self.era}/jet_jerc.json.gz")
            print("\n jetmetUncertainties Loading JSON file: {}".format(fname))
            self.cset = core.CorrectionSet.from_file(os.path.join(fname))
            
        elif "AK8" in jetType:
            self.jetBranchName = "FatJet"
            self.genJetBranchName = "GenJet"
            
            #fname_ak8 = os.path.join(__this_dir__, f"POG/JME/{self.era}/fatJet_jerc.json.gz")
            #ak8 jer resolution unavailable
            fname_ak8 = os.path.join(self.pogdir, f"POG/JME/{self.era}/jet_jerc.json.gz")
            print("\n jetSmearer Loading JSON file: {}".format(fname_ak8))
            self.cset = core.CorrectionSet.from_file(os.path.join(fname_ak8))
            
        else:
            raise ValueError("ERROR: Invalid jet type = '%s'!" % jetType)
        self.lenVar = "n" + self.jetBranchName
        

        # to fully re-calculate type-1 MET the JEC that are currently
        # applied are also needed. IS THAT EVEN CORRECT?

        # Define the jet recalibrator
        self.jetReCalibrator = JetReCalibrator(
            era,
            globalTag,
            jetType,
            True)

        # Define the recalibrator for level 1 corrections only
        self.jetReCalibratorL1 = JetReCalibrator(
            era,
            globalTag,
            jetType,
            False,
            upToLevel=1)

        # Define the recalibrators for GT used in nanoAOD production
        # (only needed to reproduce 2017 v2 MET)
        # disabled
        self.jetReCalibratorProd = False
        self.jetReCalibratorProdL1 = False

        # define energy threshold below which jets are considered as "unclustered energy"
        # cf. JetMETCorrections/Type1MET/python/correctionTermsPfMetType1Type2_cff.py
        self.unclEnThreshold = 15.

    def getJERsplitID(self, pt, eta):
        if not self.splitJER:
            return ""
        if abs(eta) < 1.93:
            return 0
        elif abs(eta) < 2.5:
            return 1
        elif abs(eta) < 3:
            if pt < 50:
                return 2
            else:
                return 3
        else:
            if pt < 50:
                return 4
            else:
                return 5

    def beginJob(self):

        if not self.isData:
            self.jetSmearer.beginJob()

    def endJob(self):
        if not self.isData:
            self.jetSmearer.endJob()

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("%s_pt_raw" % self.jetBranchName,
                        "F",
                        lenVar=self.lenVar)
        self.out.branch("%s_pt_nom" % self.jetBranchName,
                        "F",
                        lenVar=self.lenVar)
        self.out.branch("%s_mass_raw" % self.jetBranchName,
                        "F",
                        lenVar=self.lenVar)
        self.out.branch("%s_mass_nom" % self.jetBranchName,
                        "F",
                        lenVar=self.lenVar)
        self.out.branch("%s_corr_JEC" % self.jetBranchName,
                        "F",
                        lenVar=self.lenVar)
        self.out.branch("%s_corr_JER" % self.jetBranchName,
                        "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_T1_pt" % self.metBranchName, "F")
        self.out.branch("%s_T1_phi" % self.metBranchName, "F")

        if not self.isData:
            self.out.branch("%s_T1Smear_pt" % self.metBranchName, "F")
            self.out.branch("%s_T1Smear_phi" % self.metBranchName, "F")

            for shift in ["Up", "Down"]:
                for jerID in self.splitJERIDs:
                    self.out.branch("%s_pt_jer%s%s" %
                                    (self.jetBranchName, jerID, shift),
                                    "F",
                                    lenVar=self.lenVar)
                    self.out.branch("%s_mass_jer%s%s" %
                                    (self.jetBranchName, jerID, shift),
                                    "F",
                                    lenVar=self.lenVar)
                    if 'T1' in self.saveMETUncs:
                        self.out.branch(
                            "%s_T1_pt_jer%s%s" %
                            (self.metBranchName, jerID, shift), "F")
                        self.out.branch(
                            "%s_T1_phi_jer%s%s" %
                            (self.metBranchName, jerID, shift), "F")
                    if 'T1Smear' in self.saveMETUncs:
                        self.out.branch(
                            "%s_T1Smear_pt_jer%s%s" %
                            (self.metBranchName, jerID, shift), "F")
                        self.out.branch(
                            "%s_T1Smear_phi_jer%s%s" %
                            (self.metBranchName, jerID, shift), "F")

                for jesUncertainty in self.jesUncertainties:
                    self.out.branch(
                        "%s_pt_jes%s%s" %
                        (self.jetBranchName, jesUncertainty, shift),
                        "F",
                        lenVar=self.lenVar)
                    self.out.branch(
                        "%s_mass_jes%s%s" %
                        (self.jetBranchName, jesUncertainty, shift),
                        "F",
                        lenVar=self.lenVar)
                    if 'T1' in self.saveMETUncs:
                        self.out.branch(
                            "%s_T1_pt_jes%s%s" %
                            (self.metBranchName, jesUncertainty, shift), "F")
                        self.out.branch(
                            "%s_T1_phi_jes%s%s" %
                            (self.metBranchName, jesUncertainty, shift), "F")
                    if 'T1Smear' in self.saveMETUncs:
                        self.out.branch(
                            "%s_T1Smear_pt_jes%s%s" %
                            (self.metBranchName, jesUncertainty, shift), "F")
                        self.out.branch(
                            "%s_T1Smear_phi_jes%s%s" %
                            (self.metBranchName, jesUncertainty, shift), "F")

                self.out.branch(
                    "%s_T1_pt_unclustEn%s" % (self.metBranchName, shift), "F")
                self.out.branch(
                    "%s_T1_phi_unclustEn%s" % (self.metBranchName, shift), "F")
                self.out.branch(
                    "%s_T1Smear_pt_unclustEn%s" % (self.metBranchName, shift), "F")
                self.out.branch(
                    "%s_T1Smear_phi_unclustEn%s" % (self.metBranchName, shift), "F")


    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail,
        go to next event)"""
        jets = Collection(event, self.jetBranchName)
        if "AK4" in self.jetType:
            nJet = event.nJet
            lowPtJets = Collection(event,
                                   "CorrT1METJet") 
            # to subtract out of the jets for proper type-1 MET corrections
            muons = Collection(event, "Muon")
            # prepare the low pt jets (they don't have a rawFactor)
            for jet in lowPtJets:
                jet.pt = jet.rawPt
                jet.rawFactor = 0
                jet.mass = 0
                # the following dummy values should be removed once the values
                # are kept in nanoAOD
                jet.neEmEF = 0
                jet.chEmEF = 0
            
            
        if not self.isData:
            genJets = Collection(event, self.genJetBranchName)



        if not self.isData:
            self.jetSmearer.setSeed(event)

        jets_pt_raw = []
        jets_pt_jer = []
        jets_pt_nom = []

        jets_mass_raw = []
        jets_mass_nom = []

        jets_corr_JEC = []
        jets_corr_JER = []

        jets_pt_jesUp = {}
        jets_pt_jesDown = {}

        jets_mass_jesUp = {}
        jets_mass_jesDown = {}

        jets_pt_jerUp = {}
        jets_pt_jerDown = {}
        jets_mass_jerUp = {}
        jets_mass_jerDown = {}
        for jerID in self.splitJERIDs:
            jets_pt_jerUp[jerID] = []
            jets_pt_jerDown[jerID] = []
            jets_mass_jerUp[jerID] = []
            jets_mass_jerDown[jerID] = []

        for jesUncertainty in self.jesUncertainties:
            jets_pt_jesUp[jesUncertainty] = []
            jets_pt_jesDown[jesUncertainty] = []
            jets_mass_jesUp[jesUncertainty] = []
            jets_mass_jesDown[jesUncertainty] = []
            
        if "AK4" in self.jetType:

            met = Object(event, self.metBranchName)
            rawmet = Object(event, "RawMET")
            if "Puppi" in self.metBranchName:
                rawmet = Object(event, "RawPuppiMET")
            defmet = Object(event, "MET")

            (t1met_px, t1met_py) = (met.pt * math.cos(met.phi),
                                    met.pt * math.sin(met.phi))
            (def_met_px, def_met_py) = (defmet.pt * math.cos(defmet.phi),
                                        defmet.pt * math.sin(defmet.phi))
            (met_px, met_py) = (rawmet.pt * math.cos(rawmet.phi),
                                rawmet.pt * math.sin(rawmet.phi))
            (met_T1_px, met_T1_py) = (met_px, met_py)
            (met_T1Smear_px, met_T1Smear_py) = (met_px, met_py)

            if 'T1' in self.saveMETUncs:
                (met_T1_px_jerUp, met_T1_px_jerDown, met_T1_py_jerUp,
                 met_T1_py_jerDown) = ({}, {}, {}, {})
            if 'T1Smear' in self.saveMETUncs:
                (met_T1Smear_px_jerUp, met_T1Smear_px_jerDown,
                 met_T1Smear_py_jerUp, met_T1Smear_py_jerDown) = ({}, {}, {}, {})
            for jerID in self.splitJERIDs:
                if 'T1' in self.saveMETUncs:
                    (met_T1_px_jerUp[jerID], met_T1_py_jerUp[jerID]) = (met_px,
                                                                        met_py)
                    (met_T1_px_jerDown[jerID], met_T1_py_jerDown[jerID]) = (met_px,
                                                                            met_py)
                if 'T1Smear' in self.saveMETUncs:
                    (met_T1Smear_px_jerUp[jerID],
                     met_T1Smear_py_jerUp[jerID]) = (met_px, met_py)
                    (met_T1Smear_px_jerDown[jerID],
                     met_T1Smear_py_jerDown[jerID]) = (met_px, met_py)

            if 'T1' in self.saveMETUncs:
                (met_T1_px_jesUp, met_T1_py_jesUp) = ({}, {})
                (met_T1_px_jesDown, met_T1_py_jesDown) = ({}, {})
                for jesUncertainty in self.jesUncertainties:
                    met_T1_px_jesUp[jesUncertainty] = met_px
                    met_T1_py_jesUp[jesUncertainty] = met_py
                    met_T1_px_jesDown[jesUncertainty] = met_px
                    met_T1_py_jesDown[jesUncertainty] = met_py
            if 'T1Smear' in self.saveMETUncs:
                (met_T1Smear_px_jesUp, met_T1Smear_py_jesUp) = ({}, {})
                (met_T1Smear_px_jesDown, met_T1Smear_py_jesDown) = ({}, {})
                for jesUncertainty in self.jesUncertainties:
                    met_T1Smear_px_jesUp[jesUncertainty] = met_px
                    met_T1Smear_py_jesUp[jesUncertainty] = met_py
                    met_T1Smear_px_jesDown[jesUncertainty] = met_px
                    met_T1Smear_py_jesDown[jesUncertainty] = met_py


        rho = getattr(event, self.rhoBranchName)

        # match reconstructed jets to generator level ones
        # (needed to evaluate JER scale factors and uncertainties)
    
        def resolution_matching(jet, genjet):
            '''Helper function to match to gen based on pt difference'''
            #params = ROOT.PyJetParametersWrapper()
            #params.setJetEta(jet.eta)
            #params.setJetPt(jet.pt)
            #params.setRho(rho)
            #resolution = self.jetSmearer.jer.getResolution(params)
            
            #this is not available for AK8Puppi, so use self.algo (AK4PFchs for Run2, AK4PFPuppi for Run3)
            this_jettype = self.jetType
            if "AK8" in this_jettype: this_jettype = self.replacement_jetType  
            key = "{}_{}_{}".format(self.jerTag, "PtResolution", this_jettype)
            sf = self.cset[key]
            inputs = [jet.eta, jet.pt, rho]
            resolution = sf.evaluate(*inputs)

            return abs(jet.pt - genjet.pt) < 3 * resolution * jet.pt

        if not self.isData:
            pairs = matchObjectCollection(jets,
                                          genJets,
                                          dRmax=0.2,
                                          presel=resolution_matching)
            if "AK4" in self.jetType:
                lowPtPairs = matchObjectCollection(lowPtJets,
                                                   genJets,
                                                   dRmax=0.2,
                                                   presel=resolution_matching)
                pairs.update(lowPtPairs)
            
            ###
            #pairs is a dict every every jet/lowPtJet is a key, and if matched the v is a genjet object, if unmatched the v is None
        if "AK4" in self.jetType:
            _iter = enumerate(itertools.chain(jets, lowPtJets))
        else:
            _iter = enumerate(jets)
        for iJet, jet in _iter:
            # jet pt and mass corrections
            jet_pt = jet.pt
            jet_mass = jet.mass
            jet_pt_orig = jet_pt
            rawFactor = jet.rawFactor

            if hasattr(jet, "rawFactor"):
                jet_rawpt = jet_pt * (1 - jet.rawFactor)
                jet_rawmass = jet_mass * (1 - jet.rawFactor)
            else:
                jet_rawpt = -1.0 * jet_pt  # If factor not present factor will be saved as -1
                jet_rawmass = -1.0 * jet_mass  # If factor not present factor will be saved as -1

            (jet_pt, jet_mass) = self.jetReCalibrator.correct(jet, rho)
            (jet_pt_l1, jet_mass_l1) = self.jetReCalibratorL1.correct(jet, rho)
            jet.pt = jet_pt
            jet.mass = jet_mass

            # Get the JEC factors
            jec = jet_pt / jet_rawpt
            jecL1 = jet_pt_l1 / jet_rawpt
            
            if not self.isData:
                genJet = pairs[jet]
                
            if "AK4" in self.jetType:

                # get the jet for type-1 MET
                newjet = ROOT.TLorentzVector()
                newjet.SetPtEtaPhiM(
                    jet_pt_orig * (1 - jet.rawFactor) *
                    (1 - jet.muonSubtrFactor), jet.eta, jet.phi, jet.mass)
                muon_pt = jet_pt_orig * \
                    (1 - jet.rawFactor) * jet.muonSubtrFactor


                # set the jet pt to the muon subtracted raw pt
                jet.pt = newjet.Pt()
                jet.rawFactor = 0
                # get the proper jet pts for type-1 MET
                jet_pt_noMuL1L2L3 = jet.pt * jec
                jet_pt_noMuL1 = jet.pt * jecL1


                # setting jet back to central values
                jet.pt = jet_pt
                jet.rawFactor = rawFactor

            # evaluate JER scale factors and uncertainties
            # cf. https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution and
            # https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution
            if not self.isData:
                # Get the smearing factors for MET correction
                (jet_pt_jerNomVal, jet_pt_jerUpVal,
                 jet_pt_jerDownVal) = self.jetSmearer.getSmearValsPt(
                     jet, genJet, rho)
            else:
                # if you want to do something with JER in data, please add it here.
                (jet_pt_jerNomVal, jet_pt_jerUpVal, jet_pt_jerDownVal) = (1, 1,
                                                                          1)

            # these are the important jet pt values
            #jet_pt_nom = jet_pt if jet_pt > 0 else 0
            jet_pt_nom = jet_pt * jet_pt_jerNomVal if self.applySmearing else jet_pt
            
            if "AK4" in self.jetType:
                jet_pt_L1L2L3 = jet_pt_noMuL1L2L3 + muon_pt
                jet_pt_L1 = jet_pt_noMuL1 + muon_pt



            

            jet_mass_nom = jet_pt_jerNomVal * jet_mass if self.applySmearing else jet_mass
            if jet_mass_nom < 0.0:
                jet_mass_nom *= -1.0

            # don't store the low pt jets in the Jet_pt_nom branch
            if "AK8" in self.jetType or iJet < nJet:
                jets_pt_raw.append(jet_rawpt)
                jets_pt_nom.append(jet_pt_nom)
                jets_mass_raw.append(jet_rawmass)
                jets_mass_nom.append(jet_mass_nom)
                jets_corr_JEC.append(jet_pt / jet_rawpt)
                # can be used to undo JER
                jets_corr_JER.append(jet_pt_jerNomVal)

            if not self.isData:
                jet_pt_jerUp = {
                    jerID: jet_pt_nom
                    for jerID in self.splitJERIDs
                }
                jet_pt_jerDown = {
                    jerID: jet_pt_nom
                    for jerID in self.splitJERIDs
                }
                jet_mass_jerUp = {
                    jerID: jet_mass_nom
                    for jerID in self.splitJERIDs
                }
                jet_mass_jerDown = {
                    jerID: jet_mass_nom
                    for jerID in self.splitJERIDs
                }
                thisJERID = self.getJERsplitID(jet_pt_nom, jet.eta)
                jet_pt_jerUp[thisJERID] = jet_pt_jerUpVal * jet_pt
                jet_pt_jerDown[thisJERID] = jet_pt_jerDownVal * jet_pt
                jet_mass_jerUp[thisJERID] = jet_pt_jerUpVal * jet_mass
                jet_mass_jerDown[thisJERID] = jet_pt_jerDownVal * jet_mass

                # evaluate JES uncertainties
                jet_pt_jesUp = {}
                jet_pt_jesDown = {}
                jet_pt_jesUpT1 = {}
                jet_pt_jesDownT1 = {}

                jet_mass_jesUp = {}
                jet_mass_jesDown = {}

                # don't store the low pt jets in the Jet_pt_nom branch
                if "AK8" in self.jetType or iJet < nJet:
                    for jerID in self.splitJERIDs:
                        jets_pt_jerUp[jerID].append(jet_pt_jerUp[jerID])
                        jets_pt_jerDown[jerID].append(jet_pt_jerDown[jerID])
                        jets_mass_jerUp[jerID].append(jet_mass_jerUp[jerID])
                        jets_mass_jerDown[jerID].append(
                            jet_mass_jerDown[jerID])

                for jesUncertainty in self.jesUncertainties:
                    # cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetCorUncertainties
                    # cf. https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
                    #self.jesUncertainty[jesUncertainty].setJetPt(
                    #    jet_pt_nom)
                    #self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                    #delta = self.jesUncertainty[
                    #    jesUncertainty].getUncertainty(True)
                    
                    uncertainty_jetType = self.jetType
                    if "AK8" in uncertainty_jetType:
                        uncertainty_jetType = self.replacement_jetType
                    
                    key = "{}_{}_{}".format(self.jecTag, jesUncertainty, uncertainty_jetType)
                    sf = self.cset[key]
                    inputs = [jet.eta, jet_pt_nom]
                    delta = sf.evaluate(*inputs)

                    jet_pt_jesUp[jesUncertainty] = jet_pt_nom * \
                        (1. + delta)
                    jet_pt_jesDown[jesUncertainty] = jet_pt_nom * \
                        (1. - delta)
                    jet_mass_jesUp[jesUncertainty] = jet_mass_nom * \
                        (1. + delta)
                    jet_mass_jesDown[jesUncertainty] = jet_mass_nom * \
                        (1. - delta)

                    # redo JES variations for T1 MET
                    #self.jesUncertainty[jesUncertainty].setJetPt(
                    #    jet_pt_L1L2L3)
                    #self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                    #delta = self.jesUncertainty[
                    #    jesUncertainty].getUncertainty(True)
                    if "AK4" in self.jetType:
                        inputs = [jet.eta, jet_pt_L1L2L3]
                        delta = sf.evaluate(*inputs)

                        jet_pt_jesUpT1[jesUncertainty] = jet_pt_L1L2L3 * \
                            (1. + delta)
                        jet_pt_jesDownT1[jesUncertainty] = jet_pt_L1L2L3 * \
                            (1. - delta)

                    if "AK8" in self.jetType or iJet < nJet:
                        jets_pt_jesUp[jesUncertainty].append(
                            jet_pt_jesUp[jesUncertainty])
                        jets_pt_jesDown[jesUncertainty].append(
                            jet_pt_jesDown[jesUncertainty])
                        jets_mass_jesUp[jesUncertainty].append(
                            jet_mass_jesUp[jesUncertainty])
                        jets_mass_jesDown[jesUncertainty].append(
                            jet_mass_jesDown[jesUncertainty])

            # progate JER and JES corrections and uncertainties to MET.
            # Only propagate JECs to MET if the corrected pt without the muon
            # is above the threshold
            #only consider AK4 jets
            if "AK4" in self.jetType and jet_pt_noMuL1L2L3 > self.unclEnThreshold and (jet.neEmEF +
                                                             jet.chEmEF) < 0.9:
                # do not re-correct for jets that aren't included in METv2 recipe
                #if not (self.metBranchName == 'METFixEE2017'
                #        and 2.65 < abs(jet.eta) < 3.14 and jet.pt *
                #        (1 - jet.rawFactor) < 50):
                if True:
                    jet_cosPhi = math.cos(jet.phi)
                    jet_sinPhi = math.sin(jet.phi)
                    met_T1_px = met_T1_px - \
                        (jet_pt_L1L2L3 - jet_pt_L1) * jet_cosPhi
                    met_T1_py = met_T1_py - \
                        (jet_pt_L1L2L3 - jet_pt_L1) * jet_sinPhi
                    if not self.isData:
                        met_T1Smear_px = met_T1Smear_px - \
                            (jet_pt_L1L2L3 * jet_pt_jerNomVal -
                             jet_pt_L1) * jet_cosPhi
                        met_T1Smear_py = met_T1Smear_py - \
                            (jet_pt_L1L2L3 * jet_pt_jerNomVal -
                             jet_pt_L1) * jet_sinPhi
                        # Variations of T1 MET
                        if 'T1' in self.saveMETUncs:
                            for jerID in self.splitJERIDs:
                                # For uncertainties on T1 MET, the up/down
                                # variations are just the centrally smeared
                                # MET values
                                jerUpVal, jerDownVal = jet_pt_jerNomVal, jet_pt_jerNomVal
                                met_T1_px_jerUp[jerID] = met_T1_px_jerUp[jerID] - \
                                    (jet_pt_L1L2L3 * jerUpVal -
                                     jet_pt_L1) * jet_cosPhi
                                met_T1_py_jerUp[jerID] = met_T1_py_jerUp[jerID] - \
                                    (jet_pt_L1L2L3 * jerUpVal -
                                     jet_pt_L1) * jet_sinPhi
                                met_T1_px_jerDown[jerID] = met_T1_px_jerDown[
                                    jerID] - (jet_pt_L1L2L3 * jerDownVal -
                                              jet_pt_L1) * jet_cosPhi
                                met_T1_py_jerDown[jerID] = met_T1_py_jerDown[
                                    jerID] - (jet_pt_L1L2L3 * jerDownVal -
                                              jet_pt_L1) * jet_sinPhi

                            # Calculate JES uncertainties on unsmeared MET
                            for jesUncertainty in self.jesUncertainties:
                                met_T1_px_jesUp[
                                    jesUncertainty] = met_T1_px_jesUp[
                                        jesUncertainty] - (
                                            jet_pt_jesUpT1[jesUncertainty]
                                            - jet_pt_L1) * jet_cosPhi
                                met_T1_py_jesUp[
                                    jesUncertainty] = met_T1_py_jesUp[
                                        jesUncertainty] - (
                                            jet_pt_jesUpT1[jesUncertainty]
                                            - jet_pt_L1) * jet_sinPhi
                                met_T1_px_jesDown[
                                    jesUncertainty] = met_T1_px_jesDown[
                                        jesUncertainty] - (
                                            jet_pt_jesDownT1[jesUncertainty]
                                            - jet_pt_L1) * jet_cosPhi
                                met_T1_py_jesDown[
                                    jesUncertainty] = met_T1_py_jesDown[
                                        jesUncertainty] - (
                                            jet_pt_jesDownT1[jesUncertainty]
                                            - jet_pt_L1) * jet_sinPhi
                        # Variations of T1Smear MET
                        if 'T1Smear' in self.saveMETUncs:
                            for jerID in self.splitJERIDs:
                                jerUpVal, jerDownVal = jet_pt_jerNomVal, jet_pt_jerNomVal
                                if jerID == self.getJERsplitID(
                                        jet_pt_nom, jet.eta):
                                    jerUpVal, jerDownVal = jet_pt_jerUpVal, jet_pt_jerDownVal
                                met_T1Smear_px_jerUp[
                                    jerID] = met_T1Smear_px_jerUp[jerID] - (
                                        jet_pt_L1L2L3 * jerUpVal -
                                        jet_pt_L1) * jet_cosPhi
                                met_T1Smear_py_jerUp[
                                    jerID] = met_T1Smear_py_jerUp[jerID] - (
                                        jet_pt_L1L2L3 * jerUpVal -
                                        jet_pt_L1) * jet_sinPhi
                                met_T1Smear_px_jerDown[
                                    jerID] = met_T1Smear_px_jerDown[jerID] - (
                                        jet_pt_L1L2L3 * jerDownVal -
                                        jet_pt_L1) * jet_cosPhi
                                met_T1Smear_py_jerDown[
                                    jerID] = met_T1Smear_py_jerDown[jerID] - (
                                        jet_pt_L1L2L3 * jerDownVal -
                                        jet_pt_L1) * jet_sinPhi

                            # Calculate JES uncertainties on smeared MET
                            for jesUncertainty in self.jesUncertainties:
                                jesUp_correction_forT1SmearMET = (
                                    jet_pt_L1L2L3 * jet_pt_jerNomVal -
                                    jet_pt_L1) + (
                                        jet_pt_jesUpT1[jesUncertainty] -
                                        jet_pt_L1L2L3)
                                jesDown_correction_forT1SmearMET = (
                                    jet_pt_L1L2L3 * jet_pt_jerNomVal -
                                    jet_pt_L1) + (
                                        jet_pt_jesDownT1[jesUncertainty] -
                                        jet_pt_L1L2L3)
                                met_T1Smear_px_jesUp[jesUncertainty] = met_T1Smear_px_jesUp[jesUncertainty] - \
                                    jesUp_correction_forT1SmearMET * jet_cosPhi
                                met_T1Smear_py_jesUp[jesUncertainty] = met_T1Smear_py_jesUp[jesUncertainty] - \
                                    jesUp_correction_forT1SmearMET * jet_sinPhi
                                met_T1Smear_px_jesDown[jesUncertainty] = met_T1Smear_px_jesDown[jesUncertainty] - \
                                    jesDown_correction_forT1SmearMET * jet_cosPhi
                                met_T1Smear_py_jesDown[jesUncertainty] = met_T1Smear_py_jesDown[jesUncertainty] - \
                                    jesDown_correction_forT1SmearMET * jet_sinPhi

        # propagate "unclustered energy" uncertainty to MET
        
        if "AK4" in self.jetType and (not self.isData):
            (met_T1_px_unclEnUp, met_T1_py_unclEnUp) = (met_T1_px, met_T1_py)
            (met_T1_px_unclEnDown, met_T1_py_unclEnDown) = (met_T1_px, met_T1_py)
            (met_T1Smear_px_unclEnUp, met_T1Smear_py_unclEnUp) = (met_T1Smear_px, met_T1Smear_py)
            (met_T1Smear_px_unclEnDown, met_T1Smear_py_unclEnDown) = (met_T1Smear_px, met_T1Smear_py)
            met_deltaPx_unclEn = getattr(
                event, self.metBranchName + "_MetUnclustEnUpDeltaX")
            met_deltaPy_unclEn = getattr(
                event, self.metBranchName + "_MetUnclustEnUpDeltaY")
            met_T1_px_unclEnUp = met_T1_px_unclEnUp + met_deltaPx_unclEn
            met_T1_py_unclEnUp = met_T1_py_unclEnUp + met_deltaPy_unclEn
            met_T1_px_unclEnDown = met_T1_px_unclEnDown - met_deltaPx_unclEn
            met_T1_py_unclEnDown = met_T1_py_unclEnDown - met_deltaPy_unclEn
            met_T1Smear_px_unclEnUp = met_T1Smear_px_unclEnUp + met_deltaPx_unclEn
            met_T1Smear_py_unclEnUp = met_T1Smear_py_unclEnUp + met_deltaPy_unclEn
            met_T1Smear_px_unclEnDown = met_T1Smear_px_unclEnDown - met_deltaPx_unclEn
            met_T1Smear_py_unclEnDown = met_T1Smear_py_unclEnDown - met_deltaPy_unclEn

        self.out.fillBranch("%s_pt_raw" % self.jetBranchName, jets_pt_raw)
        self.out.fillBranch("%s_pt_nom" % self.jetBranchName, jets_pt_nom)
        self.out.fillBranch("%s_corr_JEC" % self.jetBranchName, jets_corr_JEC)
        self.out.fillBranch("%s_corr_JER" % self.jetBranchName, jets_corr_JER)
        if not self.isData:
            for jerID in self.splitJERIDs:
                self.out.fillBranch(
                    "%s_pt_jer%sUp" % (self.jetBranchName, jerID),
                    jets_pt_jerUp[jerID])
                self.out.fillBranch(
                    "%s_pt_jer%sDown" % (self.jetBranchName, jerID),
                    jets_pt_jerDown[jerID])

        if "AK4" in self.jetType:
            self.out.fillBranch("%s_T1_pt" % self.metBranchName,
                                math.sqrt(met_T1_px**2 + met_T1_py**2))
            self.out.fillBranch("%s_T1_phi" % self.metBranchName,
                                math.atan2(met_T1_py, met_T1_px))

        self.out.fillBranch("%s_mass_raw" % self.jetBranchName, jets_mass_raw)
        self.out.fillBranch("%s_mass_nom" % self.jetBranchName, jets_mass_nom)

        if not self.isData:
            for jerID in self.splitJERIDs:
                self.out.fillBranch(
                    "%s_mass_jer%sUp" % (self.jetBranchName, jerID),
                    jets_mass_jerUp[jerID])
                self.out.fillBranch(
                    "%s_mass_jer%sDown" % (self.jetBranchName, jerID),
                    jets_mass_jerDown[jerID])

        if not self.isData:
            
            if "AK4" in self.jetType:
                self.out.fillBranch(
                    "%s_T1Smear_pt" % self.metBranchName,
                    math.sqrt(met_T1Smear_px**2 + met_T1Smear_py**2))
                self.out.fillBranch("%s_T1Smear_phi" % self.metBranchName,
                                    math.atan2(met_T1Smear_py, met_T1Smear_px))

                if 'T1' in self.saveMETUncs:
                    for jerID in self.splitJERIDs:
                        self.out.fillBranch(
                            "%s_T1_pt_jer%sUp" % (self.metBranchName, jerID),
                            math.sqrt(met_T1_px_jerUp[jerID]**2 +
                                      met_T1_py_jerUp[jerID]**2))
                        self.out.fillBranch(
                            "%s_T1_phi_jer%sUp" % (self.metBranchName, jerID),
                            math.atan2(met_T1_py_jerUp[jerID],
                                       met_T1_px_jerUp[jerID]))
                        self.out.fillBranch(
                            "%s_T1_pt_jer%sDown" % (self.metBranchName, jerID),
                            math.sqrt(met_T1_px_jerDown[jerID]**2 +
                                      met_T1_py_jerDown[jerID]**2))
                        self.out.fillBranch(
                            "%s_T1_phi_jer%sDown" % (self.metBranchName, jerID),
                            math.atan2(met_T1_py_jerDown[jerID],
                                       met_T1_px_jerDown[jerID]))

                if 'T1Smear' in self.saveMETUncs:
                    for jerID in self.splitJERIDs:
                        self.out.fillBranch(
                            "%s_T1Smear_pt_jer%sUp" % (self.metBranchName, jerID),
                            math.sqrt(met_T1Smear_px_jerUp[jerID]**2 +
                                      met_T1Smear_py_jerUp[jerID]**2))
                        self.out.fillBranch(
                            "%s_T1Smear_phi_jer%sUp" % (self.metBranchName, jerID),
                            math.atan2(met_T1Smear_py_jerUp[jerID],
                                       met_T1Smear_px_jerUp[jerID]))
                        self.out.fillBranch(
                            "%s_T1Smear_pt_jer%sDown" %
                            (self.metBranchName, jerID),
                            math.sqrt(met_T1Smear_px_jerDown[jerID]**2 +
                                      met_T1Smear_py_jerDown[jerID]**2))
                        self.out.fillBranch(
                            "%s_T1Smear_phi_jer%sDown" %
                            (self.metBranchName, jerID),
                            math.atan2(met_T1Smear_py_jerDown[jerID],
                                       met_T1Smear_px_jerDown[jerID]))

            for jesUncertainty in self.jesUncertainties:
                self.out.fillBranch(
                    "%s_pt_jes%sUp" % (self.jetBranchName, jesUncertainty),
                    jets_pt_jesUp[jesUncertainty])
                self.out.fillBranch(
                    "%s_pt_jes%sDown" % (self.jetBranchName, jesUncertainty),
                    jets_pt_jesDown[jesUncertainty])
                
                if "AK4" in self.jetType:
                    if 'T1' in self.saveMETUncs:
                        self.out.fillBranch(
                            "%s_T1_pt_jes%sUp" %
                            (self.metBranchName, jesUncertainty),
                            math.sqrt(met_T1_px_jesUp[jesUncertainty]**2 +
                                      met_T1_py_jesUp[jesUncertainty]**2))
                        self.out.fillBranch(
                            "%s_T1_phi_jes%sUp" %
                            (self.metBranchName, jesUncertainty),
                            math.atan2(met_T1_py_jesUp[jesUncertainty],
                                       met_T1_px_jesUp[jesUncertainty]))
                        self.out.fillBranch(
                            "%s_T1_pt_jes%sDown" %
                            (self.metBranchName, jesUncertainty),
                            math.sqrt(met_T1_px_jesDown[jesUncertainty]**2 +
                                      met_T1_py_jesDown[jesUncertainty]**2))
                        self.out.fillBranch(
                            "%s_T1_phi_jes%sDown" %
                            (self.metBranchName, jesUncertainty),
                            math.atan2(met_T1_py_jesDown[jesUncertainty],
                                       met_T1_px_jesDown[jesUncertainty]))

                    if 'T1Smear' in self.saveMETUncs:
                        self.out.fillBranch(
                            "%s_T1Smear_pt_jes%sUp" %
                            (self.metBranchName, jesUncertainty),
                            math.sqrt(met_T1Smear_px_jesUp[jesUncertainty]**2 +
                                      met_T1Smear_py_jesUp[jesUncertainty]**2))
                        self.out.fillBranch(
                            "%s_T1Smear_phi_jes%sUp" %
                            (self.metBranchName, jesUncertainty),
                            math.atan2(met_T1Smear_py_jesUp[jesUncertainty],
                                       met_T1Smear_px_jesUp[jesUncertainty]))
                        self.out.fillBranch(
                            "%s_T1Smear_pt_jes%sDown" %
                            (self.metBranchName, jesUncertainty),
                            math.sqrt(met_T1Smear_px_jesDown[jesUncertainty]**2 +
                                      met_T1Smear_py_jesDown[jesUncertainty]**2))
                        self.out.fillBranch(
                            "%s_T1Smear_phi_jes%sDown" %
                            (self.metBranchName, jesUncertainty),
                            math.atan2(met_T1Smear_py_jesDown[jesUncertainty],
                                       met_T1Smear_px_jesDown[jesUncertainty]))

                self.out.fillBranch(
                    "%s_mass_jes%sUp" % (self.jetBranchName, jesUncertainty),
                    jets_mass_jesUp[jesUncertainty])
                self.out.fillBranch(
                    "%s_mass_jes%sDown" % (self.jetBranchName, jesUncertainty),
                    jets_mass_jesDown[jesUncertainty])

            if "AK4" in self.jetType:
                self.out.fillBranch(
                    "%s_T1_pt_unclustEnUp" % self.metBranchName,
                    math.sqrt(met_T1_px_unclEnUp**2 + met_T1_py_unclEnUp**2))
                self.out.fillBranch("%s_T1_phi_unclustEnUp" % self.metBranchName,
                                    math.atan2(met_T1_py_unclEnUp, met_T1_px_unclEnUp))
                self.out.fillBranch(
                    "%s_T1_pt_unclustEnDown" % self.metBranchName,
                    math.sqrt(met_T1_px_unclEnDown**2 + met_T1_py_unclEnDown**2))
                self.out.fillBranch(
                    "%s_T1_phi_unclustEnDown" % self.metBranchName,
                    math.atan2(met_T1_py_unclEnDown, met_T1_px_unclEnDown))
                self.out.fillBranch(
                    "%s_T1Smear_pt_unclustEnUp" % self.metBranchName,
                    math.sqrt(met_T1Smear_px_unclEnUp**2 + met_T1Smear_py_unclEnUp**2))
                self.out.fillBranch("%s_T1Smear_phi_unclustEnUp" % self.metBranchName,
                                    math.atan2(met_T1Smear_py_unclEnUp, met_T1Smear_px_unclEnUp))
                self.out.fillBranch(
                    "%s_T1Smear_pt_unclustEnDown" % self.metBranchName,
                    math.sqrt(met_T1Smear_px_unclEnDown**2 + met_T1Smear_py_unclEnDown**2))
                self.out.fillBranch(
                    "%s_T1Smear_phi_unclustEnDown" % self.metBranchName,
                    math.atan2(met_T1Smear_py_unclEnDown, met_T1Smear_px_unclEnDown))

        return True


    
# define modules using the syntax 'name = lambda : constructor' to avoid
# having them loaded when not needed
jetmetUncertainties2016 = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"])
jetmetUncertainties2016All = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"])

jetmetUncertainties2017 = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"])
jetmetUncertainties2017METv2 = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", metBranchName='METFixEE2017')
jetmetUncertainties2017All = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"])

jetmetUncertainties2018 = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"])
jetmetUncertainties2018Data = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_RunB_V8_DATA", archive="Autumn18_V8_DATA", isData=True)
jetmetUncertainties2018All = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"])

jetmetUncertainties2016AK4Puppi = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"], jetType="AK4PFPuppi")
jetmetUncertainties2016AK4PuppiAll = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"], jetType="AK4PFPuppi")

jetmetUncertainties2017AK4Puppi = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"], jetType="AK4PFPuppi")
jetmetUncertainties2017AK4PuppiAll = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"], jetType="AK4PFPuppi")

jetmetUncertainties2018AK4Puppi = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"], jetType="AK4PFPuppi")
jetmetUncertainties2018AK4PuppiAll = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"], jetType="AK4PFPuppi")

jetmetUncertaintiesUL2018 = lambda: jetmetUncertaintiesProducer(
    "2018_UL", "Summer19UL18_V5_MC", jerTag="Summer19UL18_JRV2_MC")

jetmetUncertaintiesUL2018_fj = lambda: jetmetUncertaintiesProducer(
    "2018_UL", "Summer19UL18_V5_MC", jerTag="Summer19UL18_JRV2_MC", jetType="AK8PFPuppi")