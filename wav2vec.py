import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers import AutoConfig, Wav2Vec2Processor

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = torch.nn.functional.log_softmax(x, dim=-1) #Added this to make the classification head work well with my HS trainer
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        # self.wav2vec2.feature_extractor._freeze_parameters()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            # attention_mask=None,
            # output_attentions=None,
            # output_hidden_states=None,
            # return_dict=None,
            # labels=None,
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values.squeeze(),
            attention_mask=None, #attention_mask,
            output_attentions=None, #output_attentions,
            output_hidden_states=None, #output_hidden_states,
            return_dict=None, #return_dict,
        )
        # print(outputs[0].shape)
        hidden_states = outputs[0] #output is a dictionary. This takes the first item.
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        
        logits = self.classifier(hidden_states)
        
        return logits
        # loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels)

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return SpeechClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

def getWav2VecCLS():
    model_name_or_path = "facebook/wav2vec2-base-960h"
    # model_name_or_path = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
    pooling_mode = "mean"
    label_list=['N', 'MS', 'MR', 'MVP', 'AS']
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        )   
    setattr(config, 'pooling_mode', pooling_mode)

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    from utils import count_parameters
    print("total params:", count_parameters(model)/1000000)
    model.freeze_feature_extractor()
    print("trainable params:", count_parameters(model)/1000000)
    return model

if __name__ =='__main__':
    # # we need to distinguish the unique labels in our SER dataset
    # label_list = ['happiness', 'disgust', 'fear', 'anger', 'sadness']
    # label_list.sort()  # Let's sort it for determinism
    # 
    # print(f"A classification problem with {num_labels} classes: {label_list}")


    from dataset import YaseenDataset
    from utils import count_parameters
    testset = YaseenDataset("/scratch/jiaqi006/others/Yaseen_CHSSUMF",'split_lists/testing_2.txt')
    # print(testset[0])


    model = getWav2VecCLS()

    print(model(testset[0][0]))
    print(count_parameters(model))
    # print(config)