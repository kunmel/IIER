import os
import logging
import torch
import config
from tqdm import tqdm, trange 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, AdamW, get_linear_schedule_with_warmup
import config
import utils
from model.model import Neighbour_scorer
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
utils.init_logger()
writer = SummaryWriter(config.board_file)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.config = RobertaConfig.from_pretrained(self.args.model_name_or_path)
        self.model = Neighbour_scorer.from_pretrained(self.args.model_name_or_path, self.args)

        self.device = self.args.device
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        no_decay = ['bias', 'LayerNorm.weight']

        bert_optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0}
        ]
        mlp_optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' not in n],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.0}
        ]
        bert_optimizer = AdamW(bert_optimizer_grouped_parameters, lr=self.args.LM_lr, eps=self.args.adam_epsilon)
        bert_schedular = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=self.args.warmup_step, num_training_steps=len(train_dataloader)*self.args.train_epochs/self.args.gradient_accumulation_steps)
        
        mlp_optimizer = AdamW(mlp_optimizer_grouped_parameters, lr=self.args.MLP_lr, eps=self.args.adam_epsilon)
        mlp_schedular = get_linear_schedule_with_warmup(mlp_optimizer, num_warmup_steps=self.args.warmup_step, num_training_steps=len(train_dataloader)*self.args.train_epochs/self.args.gradient_accumulation_steps)


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.train_epochs)
        logger.info("  Batch size = %d", self.args.batch_size)
        logger.info("  Total training steps = %d, warmup steps = %d", len(train_dataloader)*self.args.train_epochs/self.args.gradient_accumulation_steps, self.args.warmup_step)
        logger.info("  Gradient Accumulation Steps = %d", self.args.gradient_accumulation_steps) 
        logger.info("  Logging steps = %d", self.args.logging_steps)   
        logger.info("  Save steps = %d", self.args.save_steps)         

        global_step = 0
        sum_loss = 0
        self.model.zero_grad()
        train_iterator = trange(self.args.train_epochs, desc="Epoch")
        

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    "q_input_ids": batch[0],
                    "q_attention_mask": batch[1],
                    "n_input_ids": batch[2],
                    "n_attention_mask": batch[3],
                    "path_input_ids": batch[4],
                    "path_attention_mask": batch[5],
                    "valid_edges": batch[6],
                }
                outputs = self.model(**inputs)
                loss = self.model.compute_loss(outputs, batch[7])
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                sum_loss += loss.item()

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    bert_optimizer.step()
                    bert_schedular.step()
                    mlp_schedular.step()
                    mlp_optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

                    if global_step % self.args.logging_steps == 0:
                        writer.add_scalar('Loss', sum_loss/self.args.logging_steps, global_step=global_step)
                        sum_loss = 0.0

                    if global_step % self.args.save_steps == 0:
                        self.save_model(global_step)
                    if global_step % self.args.eval_steps == 0:
                        self.evaluate("dev", global_step)


    def evaluate(self, mode, global_step):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test mode are supported")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        logger.info("***** Running evaluation on %s dataset*****", mode)
        logger.info(" Num examples = %d", len(dataset))
        logger.info(" Batch size = %d", self.args.batch_size)

        eval_loss = 0.0
        eval_steps = 0
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "q_input_ids": batch[0],
                    "q_attention_mask": batch[1],
                    "n_input_ids": batch[2],
                    "n_attention_mask": batch[3],
                    "path_input_ids": batch[4],
                    "path_attention_mask": batch[5],
                    "valid_edges": batch[6],
                }
                outputs = self.model(**inputs)

                batch_loss = self.model.compute_loss(outputs, batch[7])
                eval_loss += batch_loss.mean().item()
                eval_steps += 1
        mean_loss = eval_loss / eval_steps
        writer.add_scalar('eval_loss', mean_loss, global_step=global_step)
        logger.info("***** Eval results *****")

        logger.info("  %s loss = %s", mode, mean_loss)


    def save_model(self, global_step):
        logger.info("Saving model to %s", self.args.save_dir)
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        model_path = os.path.join(self.args.save_dir, f"model_{global_step}.pt") 
        torch.save(self.model.state_dict(), model_path)  
        

    def load_model(self, model_path=None):
        logger.info("Loading model from %s", self.args.save_dir)
        state_dict = torch.load(model_path)  
        self.model.load_state_dict(state_dict)  
        