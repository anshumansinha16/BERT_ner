"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: Trainer class for training BERT for sequence labeling
"""

import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler

from .args import Config
from .dataset import DataCollator, MASKED_LB_ID
from .utils.metric import get_ner_metrics
from .utils.container import CheckpointContainer

logger = logging.getLogger(__name__)


class Trainer:
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """

    def __init__(
        self, config: Config, collate_fn=None, model=None, training_dataset=None, valid_dataset=None, test_dataset=None
    ):
        if not collate_fn:
            tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)
            collate_fn = DataCollator(tokenizer)
        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn

        self._model = model
        self._optimizer = None
        self._scheduler = None
        self._loss = None
        self._device = config.device
        self._checkpoint_container = CheckpointContainer("metric-larger")
        self.initialize()

    def initialize(self):
        """
        Initialize the trainer's status and its key components including the model,
        optimizer, learning rate scheduler, and loss function.

        Returns
        -------
        self : Trainer
            Initialized Trainer instance.
        """
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss()
        return self

    def initialize_model(self):
        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self._config.bert_model_name_or_path, num_labels=self._config.n_lbs
        )
        return self

    def initialize_optimizer(self):
        """
        Initialize training optimizer
        """
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._config.lr, weight_decay=self._config.weight_decay
        )
        #self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._config.lr)

        return self

    def initialize_scheduler(self):
        """
        Initialize learning rate scheduler
        """
        num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self._config.batch_size))
        num_warmup_steps = int(np.ceil(num_update_steps_per_epoch * self._config.warmup_ratio * self._config.n_epochs))
        num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._config.n_epochs))

        self._scheduler = get_scheduler(
            self._config.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self

    def initialize_loss(self):
        """
        Initialize loss function
        """
        self._loss = torch.nn.CrossEntropyLoss(reduction="mean")
        return self

    def run(self):
        # ----- start training process -----
        logger.info("Start training...")
        for epoch_i in range(self._config.n_epochs):
            logger.info("")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.n_epochs}")

            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)
            train_loss = self.training_step(training_dataloader)
            #print('loss',train_loss)
            logger.info(f"Training loss: {train_loss:.4f}")

            self.eval_and_save()

        best_valid_result = self.test(self._valid_dataset)
        logger.info("")
        logger.info("Best validation result:")
        self.log_results(best_valid_result, detailed=True)

        test_results = self.test()
        logger.info("")
        logger.info("Test results:")
        self.log_results(test_results, detailed=True)

        return None

    def training_step(self, data_loader):
        """
        For each training epoch
        """
        train_loss = 0
        n_tks = 1e-9

        self._model.to(self._device)
        self._model.train()
        self._optimizer.zero_grad()
        #self._model.zero_grad()

        for batch in tqdm(data_loader):

            input_ids = batch.input_ids.to(self._device)
            attention_mask = batch.attention_mask.to(self._device)
            labels = batch.labels.to(self._device)
            outputs = self._model(input_ids= batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            
            logits_x = outputs.logits

            predictions = torch.argmax(logits_x, dim=2)  # Get the most likely label index for each token.
            loss = self.get_loss(outputs.logits, batch.labels)
            
            assert torch.abs(loss - outputs.loss) < 1e-6, ValueError("Loss mismatch!")

            # Update the model parameters
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()
            #self._model.zero_grad()

            # Update the summarized loss and the number of tokens
            train_loss += loss.item()*batch.input_ids.size(0) # * len(batch.input_ids)
            n_tks += batch.input_ids.size(0)

        return train_loss / n_tks

    def get_loss(self, logits, lbs):

        # Compute the loss for the batch of data.
        loss_fct = torch.nn.CrossEntropyLoss()

        #print('lbs',lbs)

        # Flatten the logits and labels to compute standard cross entropy loss.
        active_loss = lbs.view(-1) != -100 # assuming -100 is the padding value for labels.
        
        active_logits = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = lbs.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)

        # --- TODO: end of your code ---

        return loss


    def eval_and_save(self):

        valid_results = self.evaluate(self._valid_dataset)
        #print('valid_results',valid_results)
        logger.info("Validation results:")
        self.log_results(valid_results)

        # ----- check model performance and update buffer -----
        if self._checkpoint_container.check_and_update(self._model, valid_results["f1"]):
            logger.info("Model buffer is updated!")

        return None

    def evaluate(self, dataset, detailed=False):

        self._model.to(self._device)
        self._model.eval()
        data_loader = self.get_dataloader(dataset)

        pred_lbs: list[list[str]] = []
        true_lbs: list[list[str]] = []

        # TODO: Predicted labels for each sample in the dataset and stored in `pred_lbs`, a list of list of strings.
        # TODO: The string elements represent the enitity labels, such as "O" or "B-PER".
        self._id_to_label = { -100: 'O', 0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-MISC', 8: 'I-MISC' }
        
        with torch.no_grad():  # Deactivate gradients for evaluation
            for batch in data_loader:
                batch.to(self._device)
                #print('########################')
                outputs = self._model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                #print('here_loss',outputs.loss)
                logits = outputs.logits
                #print(logits.shape)
                predictions = torch.argmax(logits, dim=2)  # Get the most likely label index for each token.

                for idx, input_id in enumerate(batch.input_ids):

                    seq_len = batch.attention_mask[idx].sum().item()  # Actual sequence length

                    trues = batch.labels[idx][:seq_len].tolist() 
                    true_label_red = trues[1:-1]
                    trues_l = [label for label in true_label_red if label != -100]
                    true_labels = [self._id_to_label[p] for p in trues_l]  # Assuming you have a dict to convert ID to label
                    true_lbs.append(true_labels)
                    
                    preds_c = predictions[idx][:seq_len].tolist()  # Trim off the padding
                    predict = preds_c[1:-1]
                    preds = [pred for idx, pred in enumerate(predict) if true_label_red[idx] != -100]

                    pred_labels = [self._id_to_label[p] for p in preds]  # Assuming you have a dict to convert ID to label
                    pred_lbs.append(pred_labels)

                    
        # --- TODO: end of your code ---
        metric = get_ner_metrics(true_lbs, pred_lbs, detailed=detailed)
        return metric

    def test(self, dataset=None):
        if dataset is None:
            dataset = self._test_dataset
        self._model.load_state_dict(self._checkpoint_container.state_dict)
        metrics = self.evaluate(dataset, detailed=True)
        return metrics

    @staticmethod
    def log_results(metrics, detailed=False):
        if detailed:
            for key, val in metrics.items():
                logger.info(f"[{key}]")
                for k, v in val.items():
                    logger.info(f"  {k}: {v:.4f}.")
        else:
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}.")

    def get_dataloader(
        self,
        dataset,
        shuffle: bool = False,
        batch_size: int = 0,
    ):

        try:
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=self._collate_fn,
                batch_size=batch_size if batch_size else self._config.batch_size,
                num_workers=getattr(self._config, "num_workers", 0),
                pin_memory=getattr(self._config, "pin_memory", False),
                shuffle=shuffle,
                drop_last=False,
            )

        except Exception as e:
            logger.exception(e)
            raise e
        
        return dataloader
