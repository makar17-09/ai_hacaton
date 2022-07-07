
from rdkit import Chem
from rdkit.Chem import rdPartialCharges as GMCharge
import pandas as pd
import numpy

iter_step = 12

import warnings
warnings.filterwarnings("ignore")


class ChargeDescriptors:


    def __init__(self):

        self._Charge = {'SPP': self.CalculateSubmolPolarityPara,
                        'LDI': self.CalculateLocalDipoleIndex,
                        'Rnc': self.CalculateRelativeNCharge,
                        'Rpc': self.CalculateRelativePCharge,
                        'Mac': self.CalculateMeanAbsoulteCharge,
                        'Tac': self.CalculateTotalAbsoulteCharge,
                        'Mnc': self.CalculateMeanNCharge,
                        'Tnc': self.CalculateTotalNCharge,
                        'Mpc': self.CalculateMeanPCharge,
                        'Tpc': self.CalculateTotalPCharge,
                        'Qass': self.CalculateAllSumSquareCharge,
                        'QOss': self.CalculateOSumSquareCharge,
                        'QNss': self.CalculateNSumSquareCharge,
                        'QCss': self.CalculateCSumSquareCharge,
                        'QHss': self.CalculateHSumSquareCharge,
                        'Qmin': self.CalculateAllMaxNCharge,
                        'Qmax': self.CalculateAllMaxPCharge,
                        'QOmin': self.CalculateOMaxNCharge,
                        'QNmin': self.CalculateNMaxNCharge,
                        'QCmin': self.CalculateCMaxNCharge,
                        'QHmin': self.CalculateHMaxNCharge,
                        'QOmax': self.CalculateOMaxPCharge,
                        'QNmax': self.CalculateNMaxPCharge,
                        'QCmax': self.CalculateCMaxPCharge,
                        'QHmax': self.CalculateHMaxPCharge,
                        }


    def _CalculateElementMaxPCharge(self, mol, AtomicNum=6):
        """
        **Internal used only**
        Most positive charge on atom with atomic number equal to n
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            if atom.GetAtomicNum() == AtomicNum:
                res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            return round(max(res), 3)


    def _CalculateElementMaxNCharge(self, mol, AtomicNum=6):
        """
        **Internal used only**
        Most negative charge on atom with atomic number equal to n
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            if atom.GetAtomicNum() == AtomicNum:
                res.append(float(atom.GetProp('_GasteigerCharge')))
        if res == []:
            return 0
        else:
            return round(min(res), 3)


    def CalculateHMaxPCharge(self, mol):
        """
        Most positive charge on H atoms
        """
        return self._CalculateElementMaxPCharge(mol, AtomicNum=1)


    def CalculateCMaxPCharge(self, mol):
        """
        Most positive charge on C atoms
        """
        return self._CalculateElementMaxPCharge(mol, AtomicNum=6)


    def CalculateNMaxPCharge(self, mol):
        """
        Most positive charge on N atoms
        """
        return self._CalculateElementMaxPCharge(mol, AtomicNum=7)


    def CalculateOMaxPCharge(self, mol):
        """
        Most positive charge on O atoms
        """
        return self._CalculateElementMaxPCharge(mol, AtomicNum=8)


    def CalculateHMaxNCharge(self, mol):
        """
        Most negative charge on H atoms
        """
        return self._CalculateElementMaxNCharge(mol, AtomicNum=1)


    def CalculateCMaxNCharge(self, mol):
        """
        Most negative charge on C atoms
        """
        return self._CalculateElementMaxNCharge(mol, AtomicNum=6)


    def CalculateNMaxNCharge(self, mol):
        """
        Most negative charge on N atoms
        """
        return self._CalculateElementMaxNCharge(mol, AtomicNum=7)


    def CalculateOMaxNCharge(self, mol):
        """
        Most negative charge on O atoms
        """
        return self._CalculateElementMaxNCharge(mol, AtomicNum=8)


    def CalculateAllMaxPCharge(self, mol):
        """
        Most positive charge on ALL atoms
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
        if res == []:
            return 0
        else:
            return round(max(res), 3)


    def CalculateAllMaxNCharge(self, mol):
        """
        Most negative charge on all atoms
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
        if res == []:
            return 0
        else:
            return round(min(res), 3)


    def _CalculateElementSumSquareCharge(self, mol, AtomicNum=6):
        """
        **Internal used only**
        Ths sum of square Charges on all atoms with atomicnumber equal to n
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            if atom.GetAtomicNum() == AtomicNum:
                res.append(float(atom.GetProp('_GasteigerCharge')))
        if res == []:
            return 0
        else:
            return round(sum(numpy.square(res)), 3)


    def CalculateHSumSquareCharge(self, mol):
        """
        The sum of square charges on all H atoms
        """
        return self._CalculateElementSumSquareCharge(mol, AtomicNum=1)


    def CalculateCSumSquareCharge(self, mol):
        """
        The sum of square charges on all C atoms
        """
        return self._CalculateElementSumSquareCharge(mol, AtomicNum=6)


    def CalculateNSumSquareCharge(self, mol):
        """
        The sum of square charges on all N atoms
        """
        return self._CalculateElementSumSquareCharge(mol, AtomicNum=7)


    def CalculateOSumSquareCharge(self, mol):
        """
        The sum of square charges on all O atoms
        """
        return self._CalculateElementSumSquareCharge(mol, AtomicNum=8)


    def CalculateAllSumSquareCharge(self, mol):
        """
        The sum of square charges on all atoms
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            return round(sum(numpy.square(res)), 3)


    def CalculateTotalPCharge(self, mol):
        """
        The total postive charge
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            return round(sum(cc[cc > 0]), 3)


    def CalculateMeanPCharge(self, mol):
        """
        The average postive charge
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            return round(numpy.mean(cc[cc > 0]), 3)


    def CalculateTotalNCharge(self, mol):
        """
        The total negative charge
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            return round(sum(cc[cc < 0]), 3)


    def CalculateMeanNCharge(self, mol):
        """
        The average negative charge
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            return round(numpy.mean(cc[cc < 0]), 3)


    def CalculateTotalAbsoulteCharge(self, mol):
        """
        The total absolute charge
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            return round(sum(numpy.absolute(cc)), 3)


    def CalculateMeanAbsoulteCharge(self, mol):
        """
        The average absolute charge
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            return round(numpy.mean(numpy.absolute(cc)), 3)


    def CalculateRelativePCharge(self, mol):
        """
        The partial charge of the most positive atom divided by
        the total positive charge.
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            if sum(cc[cc > 0]) == 0:
                return 0
            else:
                return round(max(res) / sum(cc[cc > 0]), 3)


    def CalculateRelativeNCharge(self, mol):
        """
        The partial charge of the most negative atom divided
        by the total negative charge.
        """
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))

        if res == []:
            return 0
        else:
            cc = numpy.array(res, 'd')
            if sum(cc[cc < 0]) == 0:
                return 0
            else:
                return round(min(res) / sum(cc[cc < 0]), 3)


    def CalculateLocalDipoleIndex(self, mol):
        """
        Calculation of local dipole index (D)
        """

        GMCharge.ComputeGasteigerCharges(mol, iter_step)
        res = []
        for atom in mol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
        cc = [numpy.absolute(res[x.GetBeginAtom().GetIdx()] - res[x.GetEndAtom().GetIdx()]) for x in mol.GetBonds()]
        B = len(mol.GetBonds())
        if B != 0:
            return round(sum(cc) / B, 3)
        else:
            return 0


    def CalculateSubmolPolarityPara(self, mol):
        """
        Calculation of submolecular polarity parameter(SPP)
        """

        return round(self.CalculateAllMaxPCharge(mol) - self.CalculateAllMaxNCharge(mol), 3)


    def GetChargeforMol(self, mol):
        """
        Get the dictionary of constitutional descriptors for given moelcule mol
        """
        result = {}
        for DesLabel in self._Charge.keys():
            result[DesLabel] = self._Charge[DesLabel](mol)
        return result

    def getCharge(self, df_x):
        """
        Calculates all Charge descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                charge_descriptors: pandas.DataFrame
                    Charge Descriptors DataFrame
        """

        r = {}
        for key in self._Charge.keys():
            r[key] = []
        for m in df_x['Smiles']:
            mol = Chem.MolFromSmiles(m)
            res = self.GetChargeforMol(mol)
            for key in self._Charge.keys():
                r[key].append(res[key])
        charge_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(charge_descriptors)

