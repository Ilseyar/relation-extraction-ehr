# Relation Extraction from Electronic Health Records

This repository contains the source code for the [paper](https://www.sciencedirect.com/science/article/abs/pii/S1532046420300095) Alimova I., Tutubalina E. Multiple features for clinical relation extraction: A machine learning approach //Journal of Biomedical Informatics. – 2020. – Т. 103. – С. 103382.

```
@article{alimova2020multiple,
  title={Multiple features for clinical relation extraction: A machine learning approach},
  author={Alimova, Ilseyar and Tutubalina, Elena},
  journal={Journal of Biomedical Informatics},
  volume={103},
  pages={103382},
  year={2020},
  publisher={Elsevier}
}
```

# Corpora

MADE corpus is taken from Jagannatha A. et al. Overview of the first natural language processing challenge for extracting medication, indication, and adverse drug events from electronic health record notes (MADE 1.0) //Drug safety. – 2019. – Т. 42. – №. 1. – С. 99-111.

n2c2 corpus is taken from Henry S. et al. 2018 n2c2 shared task on adverse drug events and medication extraction in electronic health records //Journal of the American Medical Informatics Association. – 2019.

# Resources

<b>Word2Vec models</b> <br />

PubMed+PMC+Wikipedia - Moen S., Ananiadou T. S. S. Distributional semantics resources for biomedical text processing //Proceedings of LBM. – 2013. – С. 39-44.

BioWordVec - Zhang Y. et al. BioWordVec, improving biomedical word embeddings with subword information and MeSH //Scientific data. – 2019. – Т. 6. – №. 1. – С. 52.

Concept embeddings - Beam A. L. et al. Clinical Concept Embeddings Learned from Massive Sources of Multimodal Medical Data //arXiv preprint arXiv:1804.01486. – 2018.

<b>Sent2VecModel</b> <br />

BioSentVec - Chen Q., Peng Y., Lu Z. BioSentVec: creating sentence embeddings for biomedical texts //2019 IEEE International Conference on Healthcare Informatics (ICHI). – IEEE, 2019. – С. 1-5.

<b> UMLS semantic types</b> <br />

Unified Medical Language System - https://www.nlm.nih.gov/research/umls/

<b> MeSH concept types </b> <br />

Medical Subject Headings - https://www.nlm.nih.gov/mesh/meshhome.html

# Project Structure

features - directory with features implementation

models - directory with accessory models implementation

relation_classification.py - classifier implementation

utils.py - additional functions for resource loading

# Classifier Usage

1. install requirements
2. download neccesary additional resources listed in Resources section
3. download dataset
4. add paths to relation_classification.py file
5. run relation_classification.py
