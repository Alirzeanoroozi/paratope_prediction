import pickle
import pandas as pd
from os.path import isfile
import sys
from structure_processor import *
from Bio.PDB import NeighborSearch

MAX_CDR_LEN = 32  # 28 + 2 + 2


def open_dataset(summary_file, dataset_cache="processed-dataset.p"):
    if isfile(dataset_cache):
        print("Precomputed dataset found, loading...")
        with open(dataset_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = compute_entries(summary_file)
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataset, f)
    return dataset


def compute_entries(summary_file):
    cdrs, lbls, cl_w = process_dataset(summary_file)
    return {
        "cdrs": flatten(cdrs),
        "lbls": flatten(lbls),
        "max_cdr_len": MAX_CDR_LEN,
        "pos_class_weight": cl_w
    }


def flatten(l):
    return [item for sublist in l for item in sublist]


def process_dataset(summary_file):
    num_in_contact = 0
    num_residues = 0

    all_cdrs = []
    all_lbls = []

    for ag_search, ab_h_chain, ab_l_chain, _, seqs, pdb in load_chains(summary_file):
        print("Processing PDB: ", pdb)

        res = process_chains(ag_search, ab_h_chain, ab_l_chain, seqs, pdb, max_cdr_len=MAX_CDR_LEN)
        if res is None:
            continue

        cdrs, lbls, (nic, nr) = res

        num_in_contact += nic
        num_residues += nr

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)

    return all_cdrs, all_lbls, num_residues / num_in_contact


def load_chains(dataset_desc_filename, sequence_cache_file="precomputed/downloaded_seqs.p"):
    with open(sequence_cache_file, "rb") as f:
        sequences = pickle.load(f)

    df = pd.read_csv(dataset_desc_filename)
    for _, entry in df.iterrows():
        pdb_name = entry['pdb']
        ab_h_chain = entry['Hchain']
        ab_l_chain = entry['Lchain']
        ag_chain = entry['antigen_chain']

        if ag_chain == ab_h_chain or ag_chain == ab_l_chain:
            print("Data mismatch, AG chain ID is the same as one of the AB chain IDs.")

        if ab_h_chain == ab_l_chain:
            ab_l_chain = ab_l_chain.lower()

        structure = get_structure_from_pdb("data/pdb/{0}.pdb".format(pdb_name))
        model = structure[0]  # Structure only has one model

        if "|" in ag_chain:  # Several chains
            chain_ids = ag_chain.split(" | ")
            ag_atoms = [a for c in chain_ids for a in Selection.unfold_entities(model[c], 'A')]
        else:  # 1 chain
            ag_atoms = Selection.unfold_entities(model[ag_chain], 'A')

        ag_search = NeighborSearch(ag_atoms)

        ag_chain_struct = None if "|" in ag_chain else model[ag_chain]

        yield ag_search, model[ab_h_chain], model[ab_l_chain], ag_chain_struct, sequences[pdb_name], (pdb_name, ab_h_chain, ab_l_chain)


def process_chains(ag_search, ab_h_chain, ab_l_chain, sequences, pdb, max_cdr_len):
    results = get_cdrs_and_contact_info(ag_search, ab_h_chain, ab_l_chain, sequences, pdb)

    cdr_mats = []
    cont_mats = []

    if results is None:
        return None

    cdrs, contact, counters = results

    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Convert Residue entities to amino acid sequences
        cdr_chain = [r[0] for r in cdrs[cdr_name]]
        cdr_mats.append(cdr_chain)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros(max_cdr_len)
        cont_mat_pad[:cont_mat.shape[0]] = cont_mat
        cont_mats.append(cont_mat_pad)

    return cdr_mats, cont_mats, counters


def get_cdrs_and_contact_info(ag_search, ab_h_chain, ab_l_chain, sequences, pdb):
    # Extract CDRs
    cdrs = {}
    cdrs.update(extract_cdrs(ab_h_chain, sequences[pdb[1]], "H"))
    cdrs.update(extract_cdrs(ab_l_chain, sequences[pdb[2]], "L"))

    # Compute ground truth -- contact information
    num_residues = 0
    num_in_contact = 0
    contact = {}

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] = [False if res[1] is None else residue_in_contact_with(res[1], ag_search) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    if num_in_contact < 5:
        print("Antibody has very few contact residues: ", num_in_contact, file=sys.stderr)
        return None

    return cdrs, contact, (num_in_contact, num_residues)


def export_sequences(dataset):
    for ag_search, ab_h_chain, ab_l_chain, _, seqs, pdb in load_chains(dataset):
        res = get_cdrs_and_contact_info(ag_search, ab_h_chain, ab_l_chain, seqs, pdb)
        if res is None:
            continue

        cdrs, contact, _ = res
        for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
            chain_id = pdb[1] if cdr_name.startswith("H") else pdb[2]
            print("> {} {} {}".format(pdb[0], chain_id, cdr_name))
            print(" ".join(str(r[2][0])+r[2][1] for r in cdrs[cdr_name]))
            print("".join(r[0] for r in cdrs[cdr_name]))
            print("".join('1' if c else '0' for c in contact[cdr_name]))


def find_cdr(cdrs):
    for i, ab in enumerate(cdrs):
        for j, cdr in enumerate(ab):
            if cdr == list("GVNTFGLY"):
                return i, j


if __name__ == "__main__":
    df = open_dataset("data/dataset.csv")
    # print(df['cdrs'][0])
    # i = df['cdrs'].index(list("GVNTFGLY"))
    # print(i)
    #
    print(find_cdr(df['cdrs']))

    for c, l in zip(df['cdrs'][0], df['lbls'][0]):
        # l = l.tolist()
        print("".join(c))
        # print(l)
        print("".join(['0' if l0 == 0 else '1' for l0 in l]))
    # exit()
    # export_sequences("data/dataset.csv")

