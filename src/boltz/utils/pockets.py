"""Helpers for inferring ligand-contact pockets from structural templates."""

from __future__ import annotations

from typing import Dict

import numpy as np
from Bio import pairwise2

from boltz.data import const
from boltz.data.types import StructureV2


def build_template_to_query_mapping(
    template_sequence: str,
    query_sequence: str,
) -> Dict[int, int]:
    """Map template residue indices to query residue indices using alignment."""

    if not template_sequence or not query_sequence:
        return {}

    alignment = pairwise2.align.globalms(
        template_sequence.upper(),
        query_sequence.upper(),
        2,
        -1,
        -0.5,
        -0.1,
        one_alignment_only=True,
    )[0]

    mapping: Dict[int, int] = {}
    template_pos = 0
    query_pos = 0
    for template_char, query_char in zip(alignment.seqA, alignment.seqB):
        if template_char != "-" and query_char != "-":
            mapping[template_pos] = query_pos
        if template_char != "-":
            template_pos += 1
        if query_char != "-":
            query_pos += 1

    return mapping


def extract_template_sequences(structure: StructureV2) -> dict[str, str]:
    """Build per-chain amino-acid sequences from a template structure."""

    sequences: dict[str, str] = {}
    protein_type = const.chain_type_ids["PROTEIN"]

    for chain in structure.chains:
        if int(chain["mol_type"]) != protein_type:
            continue

        chain_name = str(chain["name"]).strip()
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        residues = structure.residues[res_start:res_end]

        letters: list[str] = []
        for residue in residues:
            res_name = str(residue["name"]).strip()
            letters.append(const.prot_token_to_letter.get(res_name, "X"))

        sequences[chain_name] = "".join(letters)

    return sequences


def find_template_ligand_contacts(
    structure: StructureV2,
    ligand_chain: str,
    cutoff: float,
) -> list[tuple[str, int, float]]:
    """Identify template residues within ``cutoff`` Å of the ligand chain."""

    ligand_chain = str(ligand_chain)
    ligand_contacts: list[tuple[str, int, float]] = []

    chain_names = [str(name).strip() for name in structure.chains["name"]]
    try:
        ligand_index = chain_names.index(ligand_chain)
    except ValueError:
        return []

    ligand_chain_data = structure.chains[ligand_index]
    ligand_atom_start = int(ligand_chain_data["atom_idx"])
    ligand_atom_end = ligand_atom_start + int(ligand_chain_data["atom_num"])
    ligand_atoms = structure.atoms[ligand_atom_start:ligand_atom_end]
    ligand_coords = ligand_atoms["coords"][ligand_atoms["is_present"]]

    if len(ligand_coords) == 0:
        return []

    protein_type = const.chain_type_ids["PROTEIN"]
    for chain_data in structure.chains:
        if int(chain_data["mol_type"]) != protein_type:
            continue

        protein_chain_name = str(chain_data["name"]).strip()
        res_start = int(chain_data["res_idx"])
        res_end = res_start + int(chain_data["res_num"])
        residues = structure.residues[res_start:res_end]

        for res_idx, residue in enumerate(residues):
            atom_start = int(residue["atom_idx"])
            atom_end = atom_start + int(residue["atom_num"])
            residue_atoms = structure.atoms[atom_start:atom_end]
            residue_coords = residue_atoms["coords"][residue_atoms["is_present"]]

            if len(residue_coords) == 0:
                continue

            diffs = ligand_coords[:, None, :] - residue_coords[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
            min_distance = float(np.min(distances))

            if min_distance <= cutoff:
                ligand_contacts.append((protein_chain_name, res_idx, min_distance))

    return ligand_contacts


__all__ = [
    "build_template_to_query_mapping",
    "extract_template_sequences",
    "find_template_ligand_contacts",
]
