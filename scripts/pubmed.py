#! /usr/bin/env python3

import os
import gzip
import json
import subprocess
import xml.etree.ElementTree as ET

from tqdm import tqdm
from urllib.request import urlretrieve


PUBMED_FILE_LIMIT = 1


def download_pubmed_xml(output_dir, num_files=1, year='25'):
    os.makedirs(output_dir, exist_ok=True)
    base_url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

    files = []
    for i in range(1, num_files + 1):
        filename = f"pubmed{year}n{i:04d}.xml.gz"
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")

            urlretrieve(base_url + filename, filepath)

        files.append(filepath)

    return files


def parse_pubmed_to_jsonl(xml_files, output_jsonl):
    with open(output_jsonl, 'w') as out:
        for xml_file in xml_files:
            print(f"Parsing {xml_file}...")
            with gzip.open(xml_file, 'rt', encoding='utf-8') as f:
                tree = ET.parse(f)
                root = tree.getroot()

                for article in tqdm(root.findall('.//PubmedArticle')):
                    pmid_elem = article.find('.//PMID')
                    title_elem = article.find('.//ArticleTitle')
                    abstract_elem = article.find('.//Abstract/AbstractText')

                    if pmid_elem is not None:
                        title = title_elem.text if title_elem is not None else ""
                        abstract = abstract_elem.text if abstract_elem is not None else ""

                        doc = {
                            'id': pmid_elem.text,
                            'title': title,
                            'contents': f"{title} {abstract}".strip()
                        }
                        out.write(json.dumps(doc) + '\n')


def download_pubmed(output_jsonl, num_files=1):
    if os.path.exists(output_jsonl):
        print(f"Already downloaded PubMed dataset: {output_jsonl}")

        return

    xml_dir = os.path.join(os.path.dirname(output_jsonl), '../pubmed-xml')
    xml_files = download_pubmed_xml(xml_dir, num_files=num_files)
    parse_pubmed_to_jsonl(xml_files, output_jsonl)


def build_index_cmd(input_file, index_dir):
    return [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", os.path.dirname(input_file),
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "32",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]


def build_index(input_file, index_dir, cmd_generator=build_index_cmd):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Skipping existing index: {index_dir}")

        return

    os.makedirs(os.path.dirname(index_dir) or '.', exist_ok=True)

    cmd = cmd_generator(input_file, index_dir)

    subprocess.run(cmd, check=True)


def main(base_data_dir="data", base_index_dir="indexes", num_files=1):
    corpus_jsonl = os.path.join(base_data_dir, "pubmed", "corpus.jsonl")
    index_dir = os.path.join(base_index_dir, "pubmed")

    download_pubmed(corpus_jsonl, num_files=num_files)

    build_index(corpus_jsonl, index_dir)


if __name__ == "__main__":
    main(num_files=PUBMED_FILE_LIMIT)
