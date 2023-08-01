import json
import torch
from cnn import generate_mask
from preprocessing import encode_batch
from pytorch_model import Parapred, clean_output

MAX_PARAPRED_LEN = 40


def predict(cdr, output="output.json"):
    sequences, lengths = encode_batch([cdr], max_length=MAX_PARAPRED_LEN)

    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    p = Parapred()
    p.load_state_dict(torch.load("precomputed/parapred_pytorch.h5"))

    # Evaluation mode with no gradient computations
    _ = p.eval()
    with torch.no_grad():
        probabilities = p(sequences, m, lengths)

    # Linearise probabilities for viewing
    out = {}
    clean = clean_output(probabilities, lengths[0]).tolist()

    i_prob = [round(_, 5) for i, _ in enumerate(clean)]
    seq_to_prob = list(zip(cdr, i_prob))
    out[cdr] = seq_to_prob

    with open(output, "w") as jason:
        json.dump(out, jason)


if __name__ == "__main__":
    predict("GVNTFGLY")
