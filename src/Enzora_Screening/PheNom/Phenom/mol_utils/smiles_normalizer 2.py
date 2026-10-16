import os
import pickle
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from papyrus_structure_pipeline import standardizer as Papyrus_standardizer
except ModuleNotFoundError:
    Papyrus_standardizer = None

from functools import wraps
from typing import Optional, Callable, TypeVar, Any
import threading
from dataclasses import dataclass

T = TypeVar('T')

class TimeoutHandler:
    """Clean timeout handler that can be used with any function."""
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        self._result: Optional[Any] = None
        self._exception: Optional[Exception] = None
        
    def run_with_timeout(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """Run function with timeout."""
        def worker():
            try:
                self._result = func(*args, **kwargs)
            except Exception as e:
                self._exception = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True  # Allow program to exit if thread is still running
        
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            return None  # Timeout occurred
        if self._exception is not None:
            raise self._exception
        return self._result

@dataclass
class NormalizationResult:
    smiles: Optional[str]
    status: str
    error_message: Optional[str] = None

class MoleculeNormalizer:
    """Clean molecule normalization class with timeout handling."""
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_handler = TimeoutHandler(timeout_seconds)
        
    def _standardize_molecule(self, mol):
        """Internal standardization function."""
        if Papyrus_standardizer is None:
            return None
        return Papyrus_standardizer.standardize(
            mol,
            filter_inorganic=False,
            small_molecule_min_mw=100,
            small_molecule_max_mw=2000,
            return_type=True
        )
    
    def standardize_normalize(self, smiles: str) -> NormalizationResult:
        """Normalize a SMILES string with timeout handling."""
        try:
            # Convert SMILES to mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return NormalizationResult(None, "INVALID_SMILES", "Could not parse SMILES")

            if Papyrus_standardizer is None:
                fallback = self.normalize(smiles)
                if fallback.smiles is not None:
                    fallback.status = "SUCCESS_NO_CHEMBL_STD"
                return fallback
            
            # Run standardization with timeout
            result = self.timeout_handler.run_with_timeout(self._standardize_molecule, mol)
            if result is None:
                return NormalizationResult(None, "TIMEOUT", "Standardization timeout")
            
            # Process standardization result
            out_type = result[1].value
            status_mapping = {
                1: ("SUCCESS", None),
                2: ("NON_SMALL_MOLECULE", "Molecular weight outside allowed range"),
                3: ("INORGANIC_MOLECULE", "Molecule contains non-organic elements"),
                4: ("MIXTURE", "Multiple fragments detected"),
                5: ("STANDARDIZATION_ERROR", "General standardization error")
            }
            
            if out_type != 1:
                status, message = status_mapping.get(
                    out_type,
                    ("UNKNOWN_ERROR", f"Unknown standardization result type: {out_type}")
                )
                return NormalizationResult(None, status, message)
            
            # Post-process
            mol = result[0]
            mol = Chem.RemoveHs(mol, sanitize=True)
            mol = AllChem.RemoveAllHs(mol)
            normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return NormalizationResult(normalized_smiles, "SUCCESS", None)
            
        except Exception as e:
            return NormalizationResult(None, "GENERAL_ERROR", str(e))
    
    def normalize(self, smiles: str) -> NormalizationResult:
        """Normalize a SMILES string with timeout handling."""
        try:
            # Convert SMILES to mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return NormalizationResult(None, "INVALID_SMILES", "Could not parse SMILES")
    
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol, sanitize=True)
            mol = AllChem.RemoveAllHs(mol)

            normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            if normalized_smiles:
                return NormalizationResult(normalized_smiles, "SUCCESS_NO_STD", None)
            else:
                return NormalizationResult(None, "GENERAL_ERROR", str(e))
        except Exception as e:
            return NormalizationResult(None, "GENERAL_ERROR", str(e))
    
