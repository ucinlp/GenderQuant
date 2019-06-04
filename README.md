# GenderQuant

If models/novels exists already, please remove it first:
```
rm -rf models/novels
```
To train the model, run the following command:
```
allennlp train config/config_simple_roc.json -s models/novels --include-package allenModel --include-package allenDatasetReader
```
To run the website demo, run the following command, and see the result in local http://localhost:8000/:
```
python -m allennlp.service.server_simple --archive-path models/novels/model.tar.gz --predictor GenderQuant --include-package allenModel --include-package allenDatasetReader --static-dir static_html
```
If could not run the line self.nlp = spacy.load('en') in allenDatasetReader.py, please download spacy model first.
