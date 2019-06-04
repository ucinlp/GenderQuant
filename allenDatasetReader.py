from typing import Dict, Any
import json
import logging

from overrides import overrides

import tqdm
import torch
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.common.util import START_SYMBOL, END_SYMBOL

import pickle
import os, sys
import numpy as np
from readerFunctions import *
import spacy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Opts():
    def __init__(self):
        self.maleTok     = "male_subject"
        self.femTok      = "female_subject"
        self.maleTok2    = "male_pronoun"
        self.femTok2     = "female_pronoun"

        self.maleTokPronoun = 'male_pronoun'
        self.femTokPronoun  = 'female_pronoun'
        self.maleTokDB      = 'male_database'
        self.femTokDB       = 'female_database'
        self.maleTokClf     = 'male_classifier'
        self.femTokClf      = 'female_classifier'

        self.titles = ['mr','mrs','miss','mr.','mrs.']
        self.labelTokens = [self.femTok2, self.maleTok2, self.femTokDB, self.maleTokDB]
        self.maxlen = 20

        males, females = getClassifierAndNamesList()
        self.maleNames = set(males)
        self.femaleNames = set(females)
        self.keepWhat = 'NGSW' # other option is ALL
        self.preprocess_maleTok = "MALE_SUBJECT"
        self.preprocess_femTok  = "FEMALE_SUBJECT"

        self.prnCheck_maleTok = "MALE_PRONOUN"
        self.prnCheck_femTok  = "FEMALE_PRONOUN"
        self.prnCheck_refl_pronoun = "HIMSELF_HERSELF"
        self.prnCheck_other_pronouns = "MIXED_PRONOUNS"
        self.mrs = ['Mrs.', 'mrs.', 'Mrs', 'mrs', 'Miss', 'miss']
        self.mr  = ['Mr.', 'mr.', 'Mr', 'mr']

        self.genderSpecificWords = [' actress ', ' actresses ', ' aunt ', ' aunts ', ' bachelor ', ' ballerina ', ' barbershop ', ' baritone ', ' beard ', ' beards ', ' beau ', ' bloke ', ' blokes ', ' boy ', ' boyfriend ', ' boyfriends ', ' boyhood ', ' boys ', ' brethren ',' bride ', ' brides ', ' brother ', ' brotherhood ', ' brothers ', ' bull ', ' bulls ', ' businessman ', ' businessmen ', ' businesswoman ', ' chairman ', ' chairwoman ', ' chap ', ' colt ', ' colts ', ' congressman ', ' congresswoman ', ' convent ', ' councilman ', ' councilmen ', ' councilwoman ', ' countryman ', ' countrymen ', ' czar ', ' dad ', ' daddy ', ' dads ', ' daughter ', ' daughters ', ' deer ', ' diva ', ' dowry ', ' dude ', ' dudes ', ' elder brother ', ' eldest son ', ' estranged husband ', ' estranged wife ', ' estrogen ', ' ex boyfriend ', ' ex girlfriend ', ' father ', ' fathered ', ' fatherhood ', ' fathers ', ' fella ', ' fellas ', ' female ', ' females ', ' feminism ', ' fiance ', ' fiancee ', ' fillies ', ' filly ', ' fraternal ', ' fraternities ', ' fraternity ', ' gal ', ' gals ', ' gelding ', ' gentleman ', ' gentlemen ', ' girl ', ' girlfriend ', ' girlfriends ', ' girls ', ' goddess ', ' godfather ', ' granddaughter ', ' granddaughters ', ' grandfather ', ' grandma ', ' grandmother ', ' grandmothers ', ' grandpa ', ' grandson ', ' grandsons ', ' guy ', ' handyman ', ' heiress ', ' hen ', ' hens ', ' heroine ', ' horsemen ', ' hostess ', ' housewife ', ' housewives ', ' hubby ', ' husband ', ' husbands ', ' king ', ' kings ', ' lad ', ' ladies ', ' lads ', ' lady ', ' lesbian ', ' lesbians ', ' lion ', ' lions ', ' ma ', ' macho ', ' maid ', ' maiden ', ' maids ', ' male ', ' males ', ' mama ', ' man ', ' mare ', ' maternal ', ' maternity ', ' matriarch ', ' men ', ' menopause ', ' mistress ', ' mom ', ' mommy ', ' moms ', ' monastery ', ' monk ', ' monks ', ' mother ', ' motherhood ', ' mothers ', ' nephew ', ' nephews ', ' niece ', ' nieces ', ' nun ', ' nuns ', ' obstetrics ', ' ovarian cancer ', ' pa ', ' paternity ', ' penis ', ' prince ', ' princes ', ' princess ', ' prostate ', ' prostate cancer ', ' queen ', ' queens ', ' salesman ', ' salesmen ', ' schoolboy ', ' schoolgirl ', ' semen ', ' sir ', ' sister ', ' sisters ', ' son ', ' sons ', ' sorority ', ' sperm ', ' spokesman ', ' spokesmen ', ' spokeswoman ', ' stallion ', ' statesman ', ' stepdaughter ', ' stepfather ', ' stepmother ', ' stepson ', ' strongman ', ' stud ', ' studs ', ' suitor ', ' suitors ', ' teenage girl ', ' teenage girls ', ' testosterone ', ' twin brother ', ' twin sister ', ' uncle ', ' uncles ', ' uterus ', ' vagina ', ' viagra ', ' waitress ', ' widow ', ' widower ', ' widows ', ' wife ', ' witch ', ' witches ', ' wives ', ' woman ', ' womb ', ' women ', ' younger brother ']


@DatasetReader.register("GenderQuant")
class GenderQuantDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.
    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        before: ``TextField``
        after: ``TextField``
        label: ``LabelField``
    where the ``label`` is derived from the venue of the paper.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 after_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 after_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._source_add_start_token = source_add_start_token

        self._after_tokenizer = after_tokenizer or WordTokenizer()
        self._after_token_indexers = after_token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.opts = Opts()
        self.nlp = spacy.load('en')

    def _read_sentence(self, sent, demo=False):
        doc = self.nlp(sent)

        # Go through the sentence and create token replacements
        newLine = []
        replacedDoc = []

        if not demo:
            if any(word in sent.lower() for word in self.opts.genderSpecificWords):
                return []
        for i, tok in enumerate(doc):
            toAppend = processNER(tok.text, tok.ent_type_, newLine, self.opts)
            # print(tok.text, tok.ent_type_, toAppend)
            replacedDoc.append((tok, toAppend.lower()))

        # Now go through it again, and create instances as needed
        instances = []
        sameName = False
        for i, (tok, replaced) in enumerate(replacedDoc):
            if replaced in self.opts.labelTokens and not sameName:
                sameName = True
                # print(replaced)
                # need to create a mention!

                before = [r for (t,r) in replacedDoc[max(i-self.opts.maxlen,0):i]]
                after = [r for (t,r) in replacedDoc[i+1:min(i+1+self.opts.maxlen, len(sent))]]
                # target = replaced #sent[i]
                target = "male" if replaced in ['male_subject', 'male_pronoun', 'male_classifier' , 'male_database'] else "female"

                before = refineContext(before, self.opts)
                after = refineContext(after, self.opts)
                # toPrintAfter = (' ').join(after)
                after = after[::-1] # Reversing xafter
                before_text = (' ').join(before)
                after_text = (' ').join(after)
                metadata = {"position": i, "token": tok.text, "before": before_text, "after": after_text}
                # print(text, target, toPrintAfter, filename)
                instances.append(self.text_to_instance(before_text, target, after_text, metadata))
            if replaced not in self.opts.labelTokens:
                sameName = False
        return instances

    def _read_text(self, text, demo=False):
        thisFile = []
        a = text.replace('\n', ' ')
        a = a.replace('review/text:','')
        # The following replaces are for reviews only
        a = a.replace('<br /><br />', '')
        a = a.replace('<br />', '')
        a = a.replace('<br />', '')
        a = a.replace('<br', '')
        a = a.replace('/>', '')

        before = self.nlp(a)
        insts = []
        for line in before.sents:
            line = str(line)
            inst = self._read_sentence(line, demo)
            if inst is not None: insts.extend(inst)
        return insts

    @overrides
    def _read(self, file_path):
        if os.path.isdir(file_path):
            instances = []
            for filename in os.listdir(file_path):
                error = None
                file = os.path.join(file_path, filename)
                with open(file, 'r') as f:
                    try:
                        a = f.readlines()
                        a = (' ').join(a)
                        a_insts = self._read_text(a)
                        instances.extend(a_insts)
                    except Exception as e:
                        print ("Some error in file: ", filename, e)
                        error = True
                if error:
                    continue
            print("====== Length of instances " , len(instances))
            for instance in instances:
                yield instance
            # return instances
        else:
            allData = []
            with open(file_path, 'rb') as f:
                allData = pickle.load(f)

            n = len(allData['Y'])
            cnt = 0
            for i in range(n):
                if cnt > 10000:
                    break
                cnt += 1
                before   = allData['sentences'][i]
                after = allData['sentAfter'][i]
                target = allData['Y'][i]
                target = "male" if target in ['male_subject', 'male_pronoun', 'male_classifier' , 'male_database'] else "female"
                # target = str(target)

                if type(before)==list:
                    before = (' ').join(before)
                    after = (' ').join(after)

                before = refineContext(before.split(), self.opts)
                after = refineContext(after.split(), self.opts)
                after = after[::-1] # Reversing xafter
                before_text = (' ').join(before)
                after_text = (' ').join(after)
                metadata = {"before": before_text, "after": after_text}
                #print(before, target, after)
                yield self.text_to_instance(before_text, target, after_text, metadata)

    @overrides
    def text_to_instance(self, before: str,
                        target: str = None,
                        after: str = str,
                        metadata: Dict[str, Any] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_before = self._tokenizer.tokenize(before)
        # print(len(tokenized_text))
        tokenized_before.insert(0, Token(START_SYMBOL))
        before_field = TextField(tokenized_before, self._token_indexers)

        tokenized_after= self._after_tokenizer.tokenize(after)
        tokenized_after.insert(0, Token(END_SYMBOL))
        # tokenized_after_seq.append(Token(END_SYMBOL))
        after_field = TextField(tokenized_after, self._after_token_indexers)

        fields = {'before': before_field, 'after': after_field}
        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)
        if target is not None:
            fields['label'] = LabelField(target)
            # fields['label'] = LabelField(str(target),skip_indexing=False)
            # fields['label'] = LabelField(int(target),skip_indexing=True)
        return Instance(fields)

    # @classmethod
    # def from_params(cls, params: Params) -> 'NewsgroupsDatasetReader':
    #     tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
    #     token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
    #     params.assert_empty(cls.__name__)
    #     return cls(tokenizer=tokenizer, token_indexers=token_indexers)
