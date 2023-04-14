# PyTorch version of Polo code

## Credits
This implementation is based on 
[liu-yushan's repository](https://github.com/liu-yushan/PoLo), 
which contains the code for the paper 
[Neural Multi-Hop Reasoning With Logical Rules on Biomedical Knowledge Graphs](https://doi.org/10.48550/arXiv.2103.10367).
## How To Run 
The dependencies are specified in [environment.yml](environment.yml). 
Execute the command to create a Conda virtual environment.
```
conda env create -f environment.yml
```
To run PoLo, use one of the config files or create your own. 
For an explanation of each hyperparameter, refer to the [README file in the configs folder](configs/README.md).


**Note**: The Hetionet graph is split into ```graph_triples.txt``` (no inverse triples) and ```graph_inverses.txt``` (inverse triples) because of the file size constraints on GitHub.
These two files **need to be combined into one file** (with the name ```graph.txt```) before running the code.
Then, run the command
```
./run.sh configs/${config_file}.sh
```

## Data Format
### Triple format
KG triples need to be written in the format ```subject predicate object```, with tabs as separators. Furthermore, PoLo uses inverse relations, so it is important to add the inverse triple for each fact in the KG. The prefix  ```_``` is used before a predicate to signal the inverse relation, e.g., the inverse triple for ```Germany hasCapital Berlin``` is ```Berlin _hasCapital Germany```.
### File format
Datasets should have the following files:
```
dataset
    ├── train.txt
    ├── dev.txt
    ├── test.txt
    ├── graph.txt
    └── rules.txt
```
Where:

```train.txt``` contains all train triples.

```dev.txt``` contains all validation triples.

```test.txt``` contains all test triples.

```graph.txt``` contains all triples of the KG except for ```dev.txt```, ```test.txt```, the inverses of ```dev.txt```, and the inverses of ```test.txt```.

```rules.txt``` contains the rules as a dictionary, where the keys are the head relations. The rules for a specific relation are stored as a list of lists (sorted by decreasing confidence), where a rule is expressed as ```[confidence, head relation, body relation, ..., body relation]```.

For Hetionet, the complete graph is split into ```graph_triples.txt``` (no inverse triples) and ```graph_inverses.txt``` (inverse triples) because of the file size constraints on GitHub.

For rules learned by the method [AnyBURL](http://web.informatik.uni-mannheim.de/AnyBURL/), the script [preprocess_rule_list.py](https://github.com/liu-yushan/PoLo/blob/main/mycode/data/preprocessing_scripts/preprocess_rule_list.py) can be used to preprocess the rules into the format that is needed for PoLo.

Finally, two vocab files are needed, one for the entities and one for the relations. These can be created by using the [```create_vocab.py``` file](mycode/data/preprocessing_scripts/create_vocab.py).
