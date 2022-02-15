
from rdkit import Chem
from rdkit.Chem import MolSurf as MOE
from rdkit.Chem.EState import EState_VSA as EVSA
import pandas as pd



class MoeDescriptors:

    def CalculateLabuteASA(self, mol):
        """
        Calculation of Labute's Approximate Surface Area (ASA from MOE)
        """
        res = {}
        temp = MOE.pyLabuteASA(mol, includeHs=1)
        res['LabuteASA'] = round(temp, 3)
        return res


    def CalculateTPSA(self, mol):
        """
        Calculation of topological polar surface area based on fragments.
        """
        res = {}
        temp = MOE.TPSA(mol)
        res['MTPSA'] = round(temp, 3)
        return res


    def CalculateSLOGPVSA(self, mol, bins=None):
        """
        MOE-type descriptors using LogP contributions and surface
        area contributions.
        """
        temp = MOE.SlogP_VSA_(mol, bins, force=1)
        res = {}
        for i, j in enumerate(temp):
            res['slogPVSA' + str(i)] = round(j, 3)
        return res


    def CalculateSMRVSA(self, mol, bins=None):
        """
        MOE-type descriptors using MR contributions and surface
        area contributions.
        """
        temp = MOE.SMR_VSA_(mol, bins, force=1)
        res = {}
        for i, j in enumerate(temp):
            res['MRVSA' + str(i)] = round(j, 3)
        return res


    def CalculatePEOEVSA(self, mol, bins=None):
        """
        MOE-type descriptors using partial charges and surface
        area contributions.
        """
        temp = MOE.PEOE_VSA_(mol, bins, force=1)
        res = {}
        for i, j in enumerate(temp):
            res['PEOEVSA' + str(i)] = round(j, 3)
        return res


    def CalculateEstateVSA(self, mol, bins=None):
        """
        MOE-type descriptors using Estate indices and surface area
        contributions.
        """
        temp = EVSA.EState_VSA_(mol, bins, force=1)
        res = {}
        for i, j in enumerate(temp):
            res['EstateVSA' + str(i)] = round(j, 3)
        return res


    def CalculateVSAEstate(self, mol, bins=None):
        """
        MOE-type descriptors using Estate indices and surface
        area contributions.
        """
        temp = EVSA.VSA_EState_(mol, bins, force=1)
        res = {}
        for i, j in enumerate(temp):
            res['VSAEstate' + str(i)] = round(j, 3)
        return res


    def GetMOEofMol(self, mol):
        """
        The calculation of MOE-type descriptors for a molecule.
        """
        result = {}
        result.update(self.CalculateLabuteASA(mol))
        result.update(self.CalculateTPSA(mol))
        result.update(self.CalculateSLOGPVSA(mol, bins=None))
        result.update(self.CalculateSMRVSA(mol, bins=None))
        result.update(self.CalculatePEOEVSA(mol, bins=None))
        result.update(self.CalculateEstateVSA(mol, bins=None))
        result.update(self.CalculateVSAEstate(mol, bins=None))
        return result


    def getMOE(self, df_x):
        """
        Calculates all MOE descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                moe_descriptors: pandas.DataFrame
                    MOE Descriptors DataFrame
        """
        labels = ['LabuteASA','MTPSA']
        for i in range(12):
            labels.append('slogPVSA' + str(i))
        for i in range(11):
            labels.append('VSAEstate' + str(i))
        for i in range(11):
            labels.append('EstateVSA' + str(i))
        for i in range(14):
            labels.append('PEOEVSA' + str(i))
        for i in range(10):
            labels.append('MRVSA' + str(i))
        r = {}
        for key in labels:
            r[key] = []

        i=0
        for m in df_x['Smiles']:
            i=i+1
            mol = Chem.MolFromSmiles(m)
            res = self.GetMOEofMol(mol)
            for key in labels:
                r[key].append(res[key])
        moe_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(moe_descriptors)

