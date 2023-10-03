import pickle
import sys
import pandas as pd
import urllib.request
from lxml import html
import requests
import os


def download_annotated_seq(pdb, h_chain, l_chain):
    h_chain = h_chain.capitalize()
    l_chain = l_chain.capitalize()

    page = requests.get('https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/structureviewer/?pdb=' + pdb)

    tree = html.fromstring(page.content)

    fv_info = tree.xpath("//div[@id='chains']")
    try:
        chains_info_div_id = fv_info[0].xpath(".//div[@class='accordion-group']/a[div[contains(., '{}/{}')]]/@href"
                                              .format(h_chain, l_chain))[0]
    except IndexError:
        h_chain, l_chain = l_chain, h_chain
        chains_info_div_id = fv_info[0].xpath(".//div[@class='accordion-group']/a[div[contains(., '{}/{}')]]/@href"
                                              .format(h_chain, l_chain))[0]

    chains_info = fv_info[0].xpath(".//div[@id='{}']".format(chains_info_div_id[1:]))

    chains = chains_info[0].xpath(".//table[@class='table table-alignment']")

    if h_chain == l_chain:
        l_chain = l_chain.lower()

    output = {}
    for i, c in enumerate(chains):
        chain_id = h_chain if i == 0 else l_chain
        residues = c.xpath("./tr/th/text()")
        aa_names = c.xpath("./tr/td/text()")
        chain = {extract_number_and_letter(r.strip()): a for a, r in zip(aa_names, residues)}
        output[chain_id] = chain

    return output


def extract_number_and_letter(residue):
    if residue.isdigit():
        return int(residue), ''
    else:
        return int(residue[:-1]), residue[-1]


if __name__ == "__main__":
    df = pd.read_csv("merged_data.csv")
    if sys.argv[1] == "download_pdb":
        dir_list = os.listdir("pdb")

        for pdb_name in df['pdb']:
            if pdb_name + ".pdb" in dir_list:
                print("already existed")
                continue

            else:
                print("downloading", pdb_name)
                urllib.request.urlretrieve("https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/" +
                                           str.lower(pdb_name) + "/?scheme=chothia", "../data/pdb/" + pdb_name + ".pdb")
    elif sys.argv[1] == "download_annotate":
        try:
            with open("../parapred/precomputed/mine_downloaded_seqs.p", "rb") as f:
                seqs = pickle.load(f)
        except:
            seqs = {}
        for pdb_name, h, l in zip(df['pdb'], df['Hchain'], df['Lchain']):
            print(pdb_name, h, l)
            if pdb_name not in seqs.keys() and pdb_name != '6CF2':
                print("new added")
                seqs[pdb_name] = download_annotated_seq(pdb_name, h, l)
                with open("../parapred/precomputed/mine_downloaded_seqs.p", "wb+") as f:
                    pickle.dump(seqs, f)
