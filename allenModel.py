from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from typing import Any
from urllib.parse import quote

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Auc
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize

@Predictor.register('GenderQuant')
class GenderQuantPredictor(Predictor):
    @overrides
    def predict_json(self,
                    json_dict: JsonDict) -> JsonDict:
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        label_idx_dict = self._model.vocab.get_token_to_index_vocabulary('labels')
        sentence = json_dict['sentence']
        insts = self._dataset_reader._read_text(sentence, demo=True)
        outputs = []
        for inst in insts:
            output = {}
            output['true_label'] = inst.fields['label'].label
            tidx = label_idx_dict[inst.fields['label'].label]
            del inst.fields["label"] # needed to get the predict working
            prediction = self.predict_instance(inst)
            output['prediction'] = prediction
            output['pred_label'] = label_dict[prediction['label']]
            output['score'] = prediction['class_probabilities'][tidx]
            output['metadata'] = inst.fields['metadata'].metadata
            outputs.append(output)
        return {"outputs": outputs, "quotedSent": quote(sentence)}

@Model.register("GenderQuant")
class GenderQuantClassifier(Model):
    """
    This ``Model`` performs text classification for a newsgroup text.  We assume we're given a
    text and we predict some output label.
    The basic model structure: we'll embed the text and encode it with
    a Seq2VecEncoder, getting a single vector representing the content.  We'll then
    the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    before_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``before context`` ``TextField`` we get as input to the model.
    before_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the input context before the mention to a vector.
    after_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``after context`` ``TextField`` we get as input to the model.
    after_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the input context after the mention to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 before_embedder: TextFieldEmbedder,
                 before_encoder: Seq2VecEncoder,
                 after_embedder: TextFieldEmbedder,
                 after_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(GenderQuantClassifier, self).__init__(vocab, regularizer)

        self._before_embedder = before_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self._before_encoder = before_encoder
        self.classifier_feedforward = classifier_feedforward

        self._after_embedder = after_embedder
        self._after_encoder = after_encoder

        if before_embedder.get_output_dim() != before_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the before_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(before_embedder.get_output_dim(),
                                                            before_encoder.get_input_dim()))
        self.metrics = {
                 "auc": Auc()
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                before: Dict[str, torch.LongTensor],
                after: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        before: Dict[str, torch.LongTensor], required
        after: Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        before_embedded_text = self._before_embedder(before)
        before_text_mask = util.get_text_field_mask(before)
        before_encoded_text = self._before_encoder(before_embedded_text, before_text_mask)

        after_embedded_text = self._after_embedder(after)
        after_text_mask = util.get_text_field_mask(after)
        after_encoded_text = self._after_encoder(after_embedded_text, after_text_mask)

        encoded_text = torch.cat([before_encoded_text, after_encoded_text], dim=-1)
        logits = self.classifier_feedforward(encoded_text)
        output_dict = {'logits' : logits}

        class_probabilities = torch.softmax(logits, dim = -1)
        output_dict["class_probabilities"] = class_probabilities

        if label is not None:
            loss = self.loss(logits.squeeze(-1), label.squeeze(-1))
        logits = logits[:, 1:2]
        logits = logits.view(logits.shape[0],)

        if label is not None:
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis = -1)
        labels = argmax_indices
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'GenderQuantClassifier':
        embedder_params = params.pop("before_embedder")
        before_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)
        before_encoder = Seq2VecEncoder.from_params(params.pop("before_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        after_embedder = TextFieldEmbedder.from_params(params.pop("after_embedder"), vocab=vocab)
        after_encoder = Seq2VecEncoder.from_params(params.pop("after_encoder"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   before_embedder=before_embedder,
                   before_encoder=before_encoder,
                   after_embedder= after_embedder,
                   after_encoder= after_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
