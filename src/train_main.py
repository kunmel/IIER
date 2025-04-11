import config
import data_reader
import train


args = config.ARGS
args.mock_embedding = [0.0] * args.D

processor = data_reader.DataProcessor(args)
train_dataset, val_dataset, test_dataset = processor.create_example(args)

trainer = train.Trainer(args, train_dataset, val_dataset, test_dataset)
trainer.train()

