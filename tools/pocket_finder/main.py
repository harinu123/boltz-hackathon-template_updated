"""Command-line interface for extracting template-guided pocket contacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boltz.data.parse.mmcif import parse_mmcif
from boltz.data.parse.pdb import parse_pdb
from boltz.data.types import StructureV2
from boltz.utils.msa import compute_query_conservation, extract_query_from_msa
from boltz.utils.pockets import (
    build_template_to_query_mapping,
    extract_template_sequences,
    find_template_ligand_contacts,
)


def _parse_key_value(option: str, flag: str) -> tuple[str, str]:
    if "=" not in option:
        msg = f"Expected {flag} entries in the form <key>=<value>, received: '{option}'"
        raise argparse.ArgumentTypeError(msg)
    key, value = option.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        msg = f"Both key and value are required for {flag} entries: '{option}'"
        raise argparse.ArgumentTypeError(msg)
    return key, value


def _load_structure(path: Path) -> tuple[StructureV2, dict[str, str]]:
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parsed = parse_mmcif(str(path))
    else:
        parsed = parse_pdb(str(path))
    structure = parsed.data
    sequences = extract_template_sequences(structure)
    return structure, sequences


def _format_yaml_snippet(
    contacts: list[tuple[str, int]],
    binder: str,
    distance: float,
    force: bool,
    negative: bool,
) -> str:
    if negative:
        header = "negative_pocket:"
        distance_key = "min_distance"
    else:
        header = "pocket:"
        distance_key = "max_distance"

    lines = [header, f"  binder: {binder}", "  contacts:"]
    for chain_id, residue_idx in contacts:
        lines.append(f"    - [{chain_id}, {residue_idx}]")

    lines.append(f"  {distance_key}: {distance}")
    if force:
        lines.append("  force: true")

    return "\n".join(lines)


def _describe_contact(
    rank: int,
    query_chain: str,
    query_residue: int,
    conservation: float,
    distance: float,
    template_chain: str,
    template_residue: int,
) -> str:
    return (
        f"{rank:2d}. {query_chain} {query_residue:4d} | conservation={conservation:0.3f} "
        f"distance={distance:0.2f}Å | template {template_chain} {template_residue:4d}"
    )


def _resolve_msa_inputs(msa_entries: Iterable[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for entry in msa_entries:
        chain, path = _parse_key_value(entry, "--msa")
        msa_path = Path(path).expanduser().resolve()
        mapping[chain] = msa_path
    return mapping


def _resolve_chain_map(
    overrides: Iterable[str],
    template_sequences: dict[str, str],
) -> dict[str, str]:
    mapping = {chain: chain for chain in template_sequences}
    for entry in overrides:
        template_chain, query_chain = _parse_key_value(entry, "--chain-map")
        mapping[template_chain] = query_chain
    return mapping


def _build_template_mappings(
    template_sequences: dict[str, str],
    chain_map: dict[str, str],
    query_sequences: dict[str, str],
) -> dict[str, Dict[int, int]]:
    mappings: dict[str, Dict[int, int]] = {}
    for template_chain, template_sequence in template_sequences.items():
        query_chain = chain_map.get(template_chain)
        if query_chain is None:
            continue
        query_sequence = query_sequences.get(query_chain)
        if not query_sequence:
            continue
        mapping = build_template_to_query_mapping(template_sequence, query_sequence)
        if mapping:
            mappings[template_chain] = mapping
    return mappings


def _collect_conservation(msa_map: dict[str, Path], query_sequences: dict[str, str]) -> dict[str, List[float]]:
    conservation: dict[str, List[float]] = {}
    for chain, msa_path in msa_map.items():
        query_sequence = query_sequences[chain]
        try:
            scores = compute_query_conservation(msa_path, query_sequence)
        except ValueError as err:  # pragma: no cover - user input errors
            msg = f"Failed to compute conservation for chain {chain} from {msa_path}: {err}"
            raise ValueError(msg) from err
        conservation[chain] = scores.tolist()
    return conservation


def _build_output_contacts(
    structure: StructureV2,
    ligand_chain: str,
    contact_cutoff: float,
    chain_map: dict[str, str],
    template_mappings: dict[str, Dict[int, int]],
    conservation: dict[str, List[float]],
    max_contacts: int,
) -> tuple[list[tuple[str, int]], list[str], list[str]]:
    contacts = find_template_ligand_contacts(structure, ligand_chain, contact_cutoff)
    if not contacts:
        return (
            [],
            [
                f"No protein residues found within {contact_cutoff}Å of ligand chain {ligand_chain}."
            ],
            [],
        )

    warnings: list[str] = []
    scored: dict[tuple[str, int], dict[str, float]] = {}
    missing_mapping: set[str] = set()
    missing_alignment: set[str] = set()
    missing_conservation: set[str] = set()

    for template_chain, template_res_idx, distance in contacts:
        query_chain = chain_map.get(template_chain)
        if query_chain is None:
            if template_chain not in missing_mapping:
                warnings.append(
                    f"Skipping template chain {template_chain} because no query mapping was provided."
                )
                missing_mapping.add(template_chain)
            continue

        mapping = template_mappings.get(template_chain)
        if mapping is None:
            if template_chain not in missing_alignment:
                warnings.append(
                    f"No alignment available for template chain {template_chain}; skipping its contacts."
                )
                missing_alignment.add(template_chain)
            continue

        query_res_idx = mapping.get(template_res_idx)
        if query_res_idx is None:
            continue

        chain_conservation = conservation.get(query_chain)
        if chain_conservation is None:
            if query_chain not in missing_conservation:
                warnings.append(
                    f"No conservation profile loaded for query chain {query_chain}; using score -1.0."
                )
                missing_conservation.add(query_chain)
            score = -1.0
        else:
            score = float(chain_conservation[query_res_idx])

        key = (query_chain, query_res_idx)
        existing = scored.get(key)
        if (
            existing is None
            or score > existing["score"]
            or (score == existing["score"] and distance < existing["distance"])
        ):
            scored[key] = {
                "score": score,
                "distance": float(distance),
                "template_chain": template_chain,
                "template_res_idx": template_res_idx,
            }

    if not scored:
        return (
            [],
            [
                "Ligand contacts were detected but none could be mapped to the query sequences.",
                *warnings,
            ],
            [],
        )

    sorted_contacts = sorted(
        scored.items(),
        key=lambda item: (
            -item[1]["score"],
            item[1]["distance"],
            item[0][0],
            item[0][1],
        ),
    )

    formatted_contacts = [
        (chain_id, residue_idx + 1) for (chain_id, residue_idx), _ in sorted_contacts[:max_contacts]
    ]

    descriptions = [
        _describe_contact(
            rank=index + 1,
            query_chain=chain_id,
            query_residue=residue_idx + 1,
            conservation=item["score"],
            distance=item["distance"],
            template_chain=item["template_chain"],
            template_residue=item["template_res_idx"] + 1,
        )
        for index, ((chain_id, residue_idx), item) in enumerate(sorted_contacts[:max_contacts])
    ]

    return formatted_contacts, warnings, descriptions


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer ligand-contact pocket residues by combining template complexes "
            "with pre-computed MSAs."
        )
    )
    parser.add_argument("--structure", required=True, help="Template PDB/MMCIF file containing the ligand.")
    parser.add_argument("--ligand-chain", required=True, help="Chain identifier of the ligand in the template.")
    parser.add_argument(
        "--binder-chain",
        help="Optional binder chain identifier for the YAML snippet (defaults to the ligand chain).",
    )
    parser.add_argument(
        "--msa",
        action="append",
        default=[],
        help="Mapping of query chain to MSA CSV path (format: CHAIN=PATH). Repeat for multiple chains.",
    )
    parser.add_argument(
        "--chain-map",
        action="append",
        default=[],
        help=(
            "Mapping of template chains to query chains (format: TEMPLATE=QUERY). "
            "Defaults to using identical chain identifiers."
        ),
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=6.0,
        help="Distance threshold (Å) for template residue-ligand contacts.",
    )
    parser.add_argument(
        "--max-contacts",
        type=int,
        default=24,
        help="Maximum number of contacts to include in the YAML snippet.",
    )
    parser.add_argument(
        "--mode",
        choices=["positive", "negative"],
        default="positive",
        help="Generate a standard pocket (max_distance) or a negative pocket (min_distance) snippet.",
    )
    parser.add_argument("--force", action="store_true", help="Include force: true in the YAML snippet.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the YAML snippet. If omitted, the snippet is printed to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_arguments(argv)

    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.exists():
        msg = f"Structure file not found: {structure_path}"
        raise FileNotFoundError(msg)

    structure, template_sequences = _load_structure(structure_path)

    msa_map = _resolve_msa_inputs(args.msa)
    if not msa_map:
        raise ValueError("At least one --msa CHAIN=PATH entry is required to compute conservation.")

    query_sequences: dict[str, str] = {}
    for chain, msa_path in msa_map.items():
        if not msa_path.exists():
            raise FileNotFoundError(f"MSA file not found for chain {chain}: {msa_path}")
        query_sequences[chain] = extract_query_from_msa(msa_path)

    chain_map = _resolve_chain_map(args.chain_map, template_sequences)
    template_mappings = _build_template_mappings(template_sequences, chain_map, query_sequences)
    conservation = _collect_conservation(msa_map, query_sequences)

    contacts, warnings, descriptions = _build_output_contacts(
        structure=structure,
        ligand_chain=args.ligand_chain,
        contact_cutoff=args.contact_cutoff,
        chain_map=chain_map,
        template_mappings=template_mappings,
        conservation=conservation,
        max_contacts=args.max_contacts,
    )

    binder = args.binder_chain or args.ligand_chain
    snippet = _format_yaml_snippet(
        contacts,
        binder=binder,
        distance=args.contact_cutoff,
        force=args.force,
        negative=args.mode == "negative",
    )

    if warnings:
        print("\n".join(warnings))

    if descriptions:
        if warnings:
            print("")
        print("Top-ranked contacts:")
        print("\n".join(descriptions))

    if not contacts:
        print("No contacts could be mapped; the YAML snippet will only contain the binder and thresholds.")

    if args.output is not None:
        args.output.write_text(snippet + "\n")
    else:
        print("\nSuggested YAML snippet:\n")
        print(snippet)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
