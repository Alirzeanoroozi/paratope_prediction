import torch


PARAPRED_AMINO = "CSTPAGNDEQHRKMILVFYW-"
PARAPRED_TO_POS = dict([(v, i) for i, v in enumerate(PARAPRED_AMINO)])
NUM_AMINOS = len(PARAPRED_AMINO)

# Meiler features
MEILER = {
    "C": [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    "S": [1.31, 0.06, 1.6, -0.04, 5.7, 0.2, 0.28],
    "T": [3.03, 0.11, 2.6, 0.26, 5.6, 0.21, 0.36],
    "P": [2.67, 0.0, 2.72, 0.72, 6.8, 0.13, 0.34],
    "A": [1.28, 0.05, 1.0, 0.31, 6.11, 0.42, 0.23],
    "G": [0.0, 0.0, 0.0, 0.0, 6.07, 0.13, 0.15],
    "N": [1.6, 0.13, 2.95, -0.6, 6.52, 0.21, 0.22],
    "D": [1.6, 0.11, 2.78, -0.77, 2.95, 0.25, 0.2],
    "E": [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    "Q": [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    "H": [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.3],
    "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    "K": [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    "M": [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    "I": [4.19, 0.19, 4.0, 1.8, 6.04, 0.3, 0.45],
    "L": [2.59, 0.19, 4.0, 1.7, 6.04, 0.39, 0.31],
    "V": [3.67, 0.14, 3.0, 1.22, 6.02, 0.27, 0.49],
    "F": [2.94, 0.29, 5.89, 1.79, 5.67, 0.3, 0.38],
    "Y": [2.94, 0.3, 6.47, 0.96, 5.66, 0.25, 0.41],
    "W": [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    "X": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# Convert Meiler features to PyTorch Tensors
MEILER = dict([(k, torch.Tensor(v)) for k, v in MEILER.items()])
NUM_MEILER = 7

NUM_FEATURES = NUM_AMINOS + NUM_MEILER


def encode_parapred(sequence, max_length):
    """
    One-hot encode an amino acid sequence, then concatenate with Meiler features.

    :param sequence:   CDR sequence
    :param max_length: specify the maximum length for a CDR sequence

    :return: max_length x num_features tensor
    """
    # First one-hot encode the sequence, then fill the rest as the meiler feature for that amino acid
    seqlen = len(sequence)
    if max_length is None:
        encoded = torch.zeros((seqlen, NUM_FEATURES))
    else:
        encoded = torch.zeros((max_length, NUM_FEATURES))

    for i, c in enumerate(sequence):
        encoded[i][PARAPRED_TO_POS.get(c, NUM_AMINOS)] = 1
        encoded[i][-NUM_MEILER:] = MEILER[c]

    return encoded


def encode_batch(batch_of_sequences, max_length):
    """
    Encode a batch of sequences into tensors, along with their lengths

    :param batch_of_sequences:
    :param max_length:
    :return:
    """
    encoded_seqs = [encode_parapred(seq, max_length=max_length) for seq in batch_of_sequences]
    seq_lens = [len(seq) for seq in batch_of_sequences]

    encoded_seqs = torch.stack(encoded_seqs)

    return encoded_seqs, torch.as_tensor(seq_lens)


# Config constants
NUM_EXTRA_RESIDUES = 2  # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5  # Contact distance between atoms in Angstroms

chothia_cdr_def = {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
                   "H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}


def extract_cdrs(chain, sequence, chain_type):
    cdrs = {}
    pdb_residues = chain.get_unpacked_list()
    seq_residues = sorted(sequence)

    for res_id in seq_residues:
        cdr = residue_in_cdr(res_id, chain_type)
        if cdr is not None:
            pdb_res = find_pdb_residue(pdb_residues, res_id)
            cdr_seq = cdrs.get(cdr, [])
            cdr_seq.append((sequence[res_id], pdb_res, res_id))
            cdrs[cdr] = cdr_seq
    return cdrs


def extract_chains(chain, sequence, chain_type):
    chain_dict = {}
    pdb_residues = chain.get_unpacked_list()
    seq_residues = sorted(sequence)

    for res_id in seq_residues:
        pdb_res = find_pdb_residue(pdb_residues, res_id)
        chain_seq = chain_dict.get(chain_type, [])
        chain_seq.append((sequence[res_id], pdb_res, res_id))
        chain_dict[chain_type] = chain_seq
    return chain_dict


def residue_in_cdr(res_id, chain_type):
    cdr_names = [chain_type + str(e) for e in [1, 2, 3]]  # L1-3 or H1-3
    # Loop over all CDR definitions to see if the residue is in one.
    # Inefficient but easier to implement.
    for cdr_name in cdr_names:
        cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
        range_low, range_hi = -NUM_EXTRA_RESIDUES + cdr_low, cdr_hi + NUM_EXTRA_RESIDUES
        if range_low <= res_id[0] <= range_hi:
            return cdr_name
    return None


def find_pdb_residue(pdb_residues, residue_id):
    for pdb_res in pdb_residues:
        if (pdb_res.id[1], pdb_res.id[2].strip()) == residue_id:
            return pdb_res
    return None


def residue_in_contact_with(res, c_search, dist=CONTACT_DISTANCE):
    return any(len(c_search.search(a.coord, dist)) > 0 for a in res.get_unpacked_list())
