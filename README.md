# LLMs for language learning: Difficulty Constrained Decoding

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Data Science Master’s Research Project at EPFL dlab.

![Word graph "cat"](img/word%20graph%20cat.png)

This project introduces a method to design LLMs for language learning by constraining their output from an architectural perspective to a predefined vocabulary set, such as A1 proficiency words.
The constraints are formulated as a word graph, inspired from finite-state machine and paper A General-Purpose Algorithm for Constrained Sequential Inference (Deutsch et al., 2019). Dynamic masking is achieved in the beam search decoding function by pruning tokens logits. Input-dependent tokens to prune are obtained by the transitions of the word graph matchiing state. Additionally, the beam search method is tested to improve the output quality.

The image illustrates a word graph for "cat". The states represent intermediate token combinations, and transitions represent allowed tokens to reach another state along a valid path. The notation "*" stands for special tokens.

---

## Table of Contents

- [Features](#features)
- [Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- 📦 Data processing for CEFR labeled words.
- 🌲 Trie-based word graph construction.
- ⚙️ Constrained decoding, dynamically masking tokens.
- 🌟 Efficient and scalable.

---

## Structure

- `algorithms/`: Contains all intermediate implementations leading to the final version.
- `data/`: Contains data files imported in the modules.
- `docs/`: Contains additional PDFs.
- `out/`: Stores computed result images.
- `libs/`: External submodules or dependencies.
- `src/`: Core modules for the final implementation.
- `run.py`: Top-level script to run the final version of the project.

---

## Installation

### Prerequisites
- Python 3.11

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/epfl-dlab/difficulty_constrained_decoding.git
   cd difficulty_constrained_decoding
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Pynini (optionnal):
   
Pynini is a library that supports finite-state transducers, it was tested for the word graph implementation. The word graph was finally constructed from a custom trie-based structure, which was more efficient.

| **Step**                                   | **Command**                                                                                  | **Time Estimate** |
|-------------------------------------------|---------------------------------------------------------------------------------------------|-------------------|
| **Create virtual environment for Pynini** | `conda create --name pynini python=3.8`<br>`conda activate pynini`                          | -                 |
| **Install OpenFST**                        | `wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.9.tar.gz`<br>`tar -zxvf openfst-1.6.9.tar.gz`<br>`cd openfst-1.6.9`<br>`./configure --enable-grm --prefix=/path/to/folder/openfst`<br>`make`<br>`make install` | ~10 minutes       |
| **Install Re2**                            | `git clone https://github.com/google/re2`<br>`cd re2`<br>`git checkout 2018-04-01; git pull`<br>`make`<br>`make test`<br>`export CPATH=/path/to/folder/re2_lib/usr/local/include`<br>`export LD_LIBRARY_PATH=/path/to/folder/re2_lib/usr/local/lib`<br>`export LIBRARY_PATH=/path/to/folder/re2_lib/usr/local/lib`<br>`export CPATH=${CPATH}:/path/to/folder/openfst/include`<br>`export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/folder/openfst/lib`<br>`export LIBRARY_PATH=${LIBRARY_PATH}:/path/to/folder/openfst/lib`<br>`make install DESTDIR=/path/to/folder/re2_lib`<br>`make testinstall` | ~5 minutes        |
| **Install Pynini**                         | `wget http://www.opengrm.org/twiki/pub/GRM/PyniniDownload/pynini-2.0.0.tar.gz`<br>`tar -zxvf pynini-2.0.0.tar.gz`<br>`cd pynini-2.0.0`<br>`python setup.py install` | ~20 minutes       |
| **Create virtual environment for project** | `conda create --name venvname python=3.11`<br>`conda activate venvname`<br>`pip install -r requirements.txt` | -                 |

---

## Usage

Run the top-level script, and follow indications in the command line to run the different implementations.

```bash
python run.py
```

Specifically, you can choose between:
1. Run constrained decoding (final implementation)
2. Run tasks (1-10)

---

## Documentation

- [Project Report](docs/Project_Report.pdf)
- [Project Presentation](docs/Project_Presentation.pdf)
- Related Work:
  - [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning (Geng et al., 2023)](https://arxiv.org/abs/2305.13971)
  - [A General-Purpose Algorithm for Constrained Sequential Inference (Deutsch et al., 2019)](https://aclanthology.org/K19-1045/)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Libraries: Pynini.
- Supervisors: Lars Henning Klein, Valentin Hartmann (EPFL dlab).

---

## Contact

For questions or support, feel free to reach out:
Arthur Chansel – [arthur.chansel@gmail.com](email)

Or open an issue on [GitHub](https://github.com/epfl-dlab/difficulty_constrained_decoding/issues).
