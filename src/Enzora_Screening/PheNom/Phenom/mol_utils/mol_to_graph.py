from typing import Any, Dict, List
from collections import OrderedDict

import torch
import numpy as np
import torch_geometric

x_map = OrderedDict([
    # 'atomic_num':
    # list(range(0, 119)),
    ('chirality',[
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ]),
    ('degree', list(range(0, 11))),
    ('formal_charge', list(range(-5, 10))),
    ('num_hs',list(range(0, 9))),
    ('num_radical_electrons', list(range(0, 5))),
    ('hybridization', [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ]),
    ('is_aromatic', [False, True]),
    ('is_in_ring', [False, True])
])

e_map = OrderedDict([
    ('bond_type', [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ]),
    ('stereo', [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ]),
    ('is_conjugated', [False, True]),
])


ATOM_CODES = {}
metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))


atom_classes = [
    (5, 'B'),
    (6, 'C'),
    (7, 'N'),
    (8, 'O'),
    (15, 'P'),
    (16, 'S'),
    (34, 'Se'),
    ## halogen
    ([9, 17, 35, 53], 'halogen'),
    ## halogen
    (metals, 'metal')
]


NUM_ATOM_CLASSES = len(atom_classes)
for code, (atom, name) in enumerate(atom_classes):
    if type(atom) is list:
        for a in atom:
            ATOM_CODES[a] = code
    else:
        ATOM_CODES[atom] = code

def map_atom_code(AtomicNum):
    try:
        return ATOM_CODES[AtomicNum] + 1
    except KeyError:
        return 0
    


def one_hot_encode_features(features, feature_lengths):
    num_samples, num_features = features.shape
    one_hot_encoded = []

    for i in range(num_features):
        feature_column = features[:, i]
        num_classes = feature_lengths[list(feature_lengths.keys())[i]]
        one_hot = torch.eye(num_classes)[feature_column]
        one_hot_encoded.append(one_hot)

    return torch.concat(one_hot_encoded, dim=1)


x_map_length = {key: len(value) for key, value in x_map.items()}
e_map_length = {key: len(value) for key, value in e_map.items()}

def from_rdmol(mol: Any, one_hot=True) -> 'torch_geometric.data.Data':
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """
    from rdkit import Chem

    from torch_geometric.data import Data

    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    atom_ids: List[int] = []
    atomic_nums: List[int] = []

    for atom in mol.GetAtoms():  # type: ignore
        row: List[int] = []
        AtomicNum = atom.GetAtomicNum()
        atom_ids.append(map_atom_code(AtomicNum))
        atomic_nums.append(AtomicNum)

        row.append(x_map['chirality'].index(str(atom.GetChiralTag())) if str(atom.GetChiralTag()) in x_map['chirality'] else 0)
        row.append(x_map['degree'].index(atom.GetTotalDegree()) if atom.GetTotalDegree() in x_map['degree'] else len(x_map['degree']) - 1)
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()) if atom.GetFormalCharge() in x_map['formal_charge'] else len(x_map['formal_charge']) - 1)
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()) if atom.GetTotalNumHs() in x_map['num_hs'] else len(x_map['num_hs']) - 1)
        row.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()) if atom.GetNumRadicalElectrons() in x_map['num_radical_electrons'] else len(x_map['num_radical_electrons']) - 1)
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())) if str(atom.GetHybridization()) in x_map['hybridization'] else 0)
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)
        
    atom_ids = torch.tensor(atom_ids,dtype=torch.long).view(-1, 1)
    atomic_nums = torch.tensor(atomic_nums,dtype=torch.long).view(-1, 1) 
    x = torch.tensor(xs, dtype=torch.long).view(-1, 8)
    
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    if one_hot:
        x = one_hot_encode_features(x,x_map_length)
        edge_attr = one_hot_encode_features(edge_attr,e_map_length)
    else:
        x = x.float()
        edge_attr = edge_attr.float()
    return atom_ids, atomic_nums, x, edge_index, edge_attr


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = from_rdmol(mol)
    data.smiles = smiles
    return data


def to_rdmol(
    data: 'torch_geometric.data.Data',
    kekulize: bool = False,
) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a
    :class:`rdkit.Chem.Mol` instance.

    Args:
        data (torch_geometric.data.Data): The molecular graph data.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(data.x[i, 1])])
        atom.SetFormalCharge(x_map['formal_charge'][int(data.x[i, 3])])
        atom.SetNumExplicitHs(x_map['num_hs'][int(data.x[i, 4])])
        atom.SetNumRadicalElectrons(x_map['num_radical_electrons'][int(
            data.x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(
            data.x[i, 6])])
        atom.SetIsAromatic(bool(data.x[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0])]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[int(data.edge_attr[i, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return mol


def to_smiles(
    data: 'torch_geometric.data.Data',
    kekulize: bool = False,
) -> str:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem
    mol = to_rdmol(data, kekulize=kekulize)
    return Chem.MolToSmiles(mol, isomericSmiles=True)