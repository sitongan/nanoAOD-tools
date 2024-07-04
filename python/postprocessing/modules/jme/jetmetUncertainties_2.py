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
ROOT.PyConfig.IgnoreCommandLineOptions = True

def get_corr_inputs(input_dict, corr_obj):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = [input_dict[inp.name] for inp in corr_obj.inputs]
    return input_values

class jetmetUncertaintiesProducer(Module):
    def __init__(self,
                 baseDir,
                 jecJson,
                 jecTag,
                 jesUncertainties=["Total"],
                 metBranchName="MET",
                 jerJson="",
                 jerTag="",
                 algo="",
                 isData=False,
                 applyJEC=False, #for Run2UL and Run3, nanoaod by default has JEC already applied
                 applySmearing=True,
                 splitJER=False,
                 saveMETUncs=['T1', 'T1Smear']
     ):
        self.jesUncertainties = jesUncertainties
        self.jecJson = os.path.join(baseDir, jecJson)
        self.jerJson = os.path.join(baseDir, "jer_smear.json.gz")
        self.jecTag = jecTag
        self.jerTag = jerTag
        self.algo = algo
        self.isData = isData
        self.applyJEC = applyJEC
        self.applySmearing = applySmearing if not isData else False
        
        #output related
        self.metBranchName = metBranchName
        self.rhoBranchName = "fixedGridRhoFastjetAll"
        if "AK4" in jetType:
            self.jetBranchName = "Jet"
            self.genJetBranchName = "GenJet"
            self.genSubJetBranchName = None
        else:
            raise ValueError("ERROR: Invalid jet type = '%s'!" % jetType)
        self.lenVar = "n" + self.jetBranchName
        
        self.splitJER = splitJER
        if self.splitJER:
            self.splitJERIDs = list(range(6))
        else:
            self.splitJERIDs = [""]  # "empty" ID for the overall JER
        

    def beginJob(self):

        print("Loading jet energy scale (JES) uncertainties from file '%s'" %
              self.jecJson)
        self.cset = core.CorrectionSet.from_file(jecJson)
        if self.applySmearing:
            self.cset_jersmear = core.CorrectionSet.from_file(self.jerJson)
        

    def endJob(self):
        pass

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
                    #met
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
                    #met
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

                #met unclust
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
        nJet = event.nJet
        lowPtJets = Collection(event,
                               "CorrT1METJet")
        # to subtract out of the jets for proper type-1 MET corrections
        muons = Collection(event, "Muon")
        if not self.isData:
            genJets = Collection(event, self.genJetBranchName)

        # prepare the low pt jets (they don't have a rawFactor)
        for jet in lowPtJets:
            jet.pt = jet.rawPt
            jet.rawFactor = 0
            jet.mass = 0
            # the following dummy values should be removed once the values
            # are kept in nanoAOD
            jet.neEmEF = 0
            jet.chEmEF = 0
        # !!!!! is lowptjet still needed?

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
            
            key = "{}_{}_{}".format(self.jerTag, "PtResolution", self.algo)
            sf = self.cset[key]

            sf_input_names = [inp.name for inp in sf.inputs]
            print("Inputs: " + ", ".join(sf_input_names))

            inputs = get_corr_inputs(example_value_dict, sf)
            jer_value = sf.evaluate(*inputs)

            return abs(jet.pt - genjet.pt) < 3 * resolution * jet.pt

        if not self.isData:
            pairs = matchObjectCollection(jets,
                                          genJets,
                                          dRmax=0.2,
                                          presel=resolution_matching)
            lowPtPairs = matchObjectCollection(lowPtJets,
                                               genJets,
                                               dRmax=0.2,
                                               presel=resolution_matching)
            pairs.update(lowPtPairs)
            
            ###
            #pairs is a dict every every jet/lowPtJet is a key, and if matched the v is a genjet object / genjet.p4 TLorentzVector, if unmatched the v is None
            

        for iJet, jet in enumerate(itertools.chain(jets, lowPtJets)):
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
            if self.jetReCalibratorProd:
                jecProd = (
                    self.jetReCalibratorProd.correct(jet, rho)[0] / jet_rawpt)
                jecL1Prod = (
                    self.jetReCalibratorProdL1.correct(jet, rho)[0] / jet_rawpt)

            if not self.isData:
                genJet = pairs[jet]

            # get the jet for type-1 MET
            newjet = ROOT.TLorentzVector()
            if self.isV5NanoAOD:
                newjet.SetPtEtaPhiM(
                    jet_pt_orig * (1 - jet.rawFactor) *
                    (1 - jet.muonSubtrFactor), jet.eta, jet.phi, jet.mass)
                muon_pt = jet_pt_orig * \
                    (1 - jet.rawFactor) * jet.muonSubtrFactor
            else:
                newjet.SetPtEtaPhiM(jet_pt_orig * (1 - jet.rawFactor), jet.eta,
                                    jet.phi, jet.mass)
                muon_pt = 0
                if hasattr(jet, 'muonIdx1'):
                    if jet.muonIdx1 > -1:
                        if muons[jet.muonIdx1].isGlobal:
                            newjet = newjet - muons[jet.muonIdx1].p4()
                            muon_pt += muons[jet.muonIdx1].pt
                    if jet.muonIdx2 > -1:
                        if muons[jet.muonIdx2].isGlobal:
                            newjet = newjet - muons[jet.muonIdx2].p4()
                            muon_pt += muons[jet.muonIdx2].pt

            # set the jet pt to the muon subtracted raw pt
            jet.pt = newjet.Pt()
            jet.rawFactor = 0
            # get the proper jet pts for type-1 MET
            jet_pt_noMuL1L2L3 = jet.pt * jec
            jet_pt_noMuL1 = jet.pt * jecL1

            # this step is only needed for v2 MET in 2017 when different JECs
            # are applied compared to the nanoAOD production
            if self.jetReCalibratorProd:
                jet_pt_noMuProdL1L2L3 = jet.pt * jecProd if jet.pt * \
                    jecProd > self.unclEnThreshold else jet.pt
                jet_pt_noMuProdL1 = jet.pt * jecL1Prod if jet.pt * \
                    jecProd > self.unclEnThreshold else jet.pt
            else:
                jet_pt_noMuProdL1L2L3 = jet_pt_noMuL1L2L3
                jet_pt_noMuProdL1 = jet_pt_noMuL1

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
            jet_pt_L1L2L3 = jet_pt_noMuL1L2L3 + muon_pt
            jet_pt_L1 = jet_pt_noMuL1 + muon_pt

            # not nice, but needed for METv2 in 2017
            jet_pt_prodL1L2L3 = jet_pt_noMuProdL1L2L3 + muon_pt
            jet_pt_prodL1 = jet_pt_noMuProdL1 + muon_pt

            if self.metBranchName == 'METFixEE2017':
                # get the delta for removing L1L2L3-L1 corrected jets
                # (corrected with GT from nanoAOD production!!) in the
                # EE region from the default MET branch.
                if jet_pt_prodL1L2L3 > self.unclEnThreshold and 2.65 < abs(
                        jet.eta) < 3.14 and jet_rawpt < 50:
                    delta_x_T1Jet += (jet_pt_prodL1L2L3 - jet_pt_prodL1) * \
                        math.cos(jet.phi) + jet_rawpt * math.cos(jet.phi)
                    delta_y_T1Jet += (jet_pt_prodL1L2L3 - jet_pt_prodL1) * \
                        math.sin(jet.phi) + jet_rawpt * math.sin(jet.phi)

                # get the delta for removing raw jets in the EE region from
                # the raw MET
                if jet_pt_prodL1L2L3 > self.unclEnThreshold and 2.65 < abs(
                        jet.eta) < 3.14 and jet_rawpt < 50:
                    delta_x_rawJet += jet_rawpt * math.cos(jet.phi)
                    delta_y_rawJet += jet_rawpt * math.sin(jet.phi)

            jet_mass_nom = jet_pt_jerNomVal * jet_mass if self.applySmearing else jet_mass
            if jet_mass_nom < 0.0:
                jet_mass_nom *= -1.0

            # don't store the low pt jets in the Jet_pt_nom branch
            if iJet < nJet:
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
                if iJet < nJet:
                    for jerID in self.splitJERIDs:
                        jets_pt_jerUp[jerID].append(jet_pt_jerUp[jerID])
                        jets_pt_jerDown[jerID].append(jet_pt_jerDown[jerID])
                        jets_mass_jerUp[jerID].append(jet_mass_jerUp[jerID])
                        jets_mass_jerDown[jerID].append(
                            jet_mass_jerDown[jerID])

                for jesUncertainty in self.jesUncertainties:
                    # cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetCorUncertainties
                    # cf. https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
                    if jesUncertainty == "HEMIssue":
                        delta = 1.
                        if iJet < nJet and jet_pt_nom > 15 and jet.jetId & 2 and jet.phi > -1.57 and jet.phi < -0.87:
                            if jet.eta > -2.5 and jet.eta < -1.3:
                                delta = 0.8
                            elif jet.eta <= -2.5 and jet.eta > -3:
                                delta = 0.65
                        jet_pt_jesUp[jesUncertainty] = jet_pt_nom
                        jet_pt_jesDown[jesUncertainty] = delta * jet_pt_nom
                        jet_mass_jesUp[jesUncertainty] = jet_mass_nom
                        jet_mass_jesDown[jesUncertainty] = delta * jet_mass_nom

                        jet_pt_jesUpT1[jesUncertainty] = jet_pt_L1L2L3
                        jet_pt_jesDownT1[jesUncertainty] = delta * \
                            jet_pt_L1L2L3

                    else:
                        self.jesUncertainty[jesUncertainty].setJetPt(
                            jet_pt_nom)
                        self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                        delta = self.jesUncertainty[
                            jesUncertainty].getUncertainty(True)
                        jet_pt_jesUp[jesUncertainty] = jet_pt_nom * \
                            (1. + delta)
                        jet_pt_jesDown[jesUncertainty] = jet_pt_nom * \
                            (1. - delta)
                        jet_mass_jesUp[jesUncertainty] = jet_mass_nom * \
                            (1. + delta)
                        jet_mass_jesDown[jesUncertainty] = jet_mass_nom * \
                            (1. - delta)

                        # redo JES variations for T1 MET
                        self.jesUncertainty[jesUncertainty].setJetPt(
                            jet_pt_L1L2L3)
                        self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                        delta = self.jesUncertainty[
                            jesUncertainty].getUncertainty(True)
                        jet_pt_jesUpT1[jesUncertainty] = jet_pt_L1L2L3 * \
                            (1. + delta)
                        jet_pt_jesDownT1[jesUncertainty] = jet_pt_L1L2L3 * \
                            (1. - delta)

                    if iJet < nJet:
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
            if jet_pt_noMuL1L2L3 > self.unclEnThreshold and (jet.neEmEF +
                                                             jet.chEmEF) < 0.9:
                # do not re-correct for jets that aren't included in METv2 recipe
                if not (self.metBranchName == 'METFixEE2017'
                        and 2.65 < abs(jet.eta) < 3.14 and jet.pt *
                        (1 - jet.rawFactor) < 50):
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
        if self.metBranchName == 'METFixEE2017':
            # Remove the L1L2L3-L1 corrected jets in the EE region from the
            # default MET branch
            def_met_px += delta_x_T1Jet
            def_met_py += delta_y_T1Jet

            # get unclustered energy part that is removed in the v2 recipe
            met_unclEE_x = def_met_px - t1met_px
            met_unclEE_y = def_met_py - t1met_py

            # finalize the v2 recipe for the rawMET by removing the unclustered
            # part in the EE region
            met_T1_px += delta_x_rawJet - met_unclEE_x
            met_T1_py += delta_y_rawJet - met_unclEE_y

            if not self.isData:
                # apply v2 recipe correction factor also to JER central value
                # and variations
                met_T1Smear_px += delta_x_rawJet - met_unclEE_x
                met_T1Smear_py += delta_y_rawJet - met_unclEE_y

            if 'T1' in self.saveMETUncs:
                for jerID in self.splitJERIDs:
                    met_T1_px_jerUp[jerID] += delta_x_rawJet - met_unclEE_x
                    met_T1_py_jerUp[jerID] += delta_y_rawJet - met_unclEE_y
                    met_T1_px_jerDown[jerID] += delta_x_rawJet - met_unclEE_x
                    met_T1_py_jerDown[jerID] += delta_y_rawJet - met_unclEE_y

                for jesUncertainty in self.jesUncertainties:
                    met_T1_px_jesUp[jesUncertainty] += delta_x_rawJet - \
                        met_unclEE_x
                    met_T1_py_jesUp[jesUncertainty] += delta_y_rawJet - \
                        met_unclEE_y
                    met_T1_px_jesDown[jesUncertainty] += delta_x_rawJet - \
                        met_unclEE_x
                    met_T1_py_jesDown[jesUncertainty] += delta_y_rawJet - \
                        met_unclEE_y

            if 'T1Smear' in self.saveMETUncs:
                for jerID in self.splitJERIDs:
                    met_T1Smear_px_jerUp[jerID] += delta_x_rawJet - \
                        met_unclEE_x
                    met_T1Smear_py_jerUp[jerID] += delta_y_rawJet - \
                        met_unclEE_y
                    met_T1Smear_px_jerDown[jerID] += delta_x_rawJet - \
                        met_unclEE_x
                    met_T1Smear_py_jerDown[jerID] += delta_y_rawJet - \
                        met_unclEE_y

                for jesUncertainty in self.jesUncertainties:
                    met_T1Smear_px_jesUp[jesUncertainty] += delta_x_rawJet - \
                        met_unclEE_x
                    met_T1Smear_py_jesUp[jesUncertainty] += delta_y_rawJet - \
                        met_unclEE_y
                    met_T1Smear_px_jesDown[
                        jesUncertainty] += delta_x_rawJet - met_unclEE_x
                    met_T1Smear_py_jesDown[
                        jesUncertainty] += delta_y_rawJet - met_unclEE_y

        if not self.isData:
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
