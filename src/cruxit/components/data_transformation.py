import os
from src.cruxit.logging import logger
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.cruxit.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)


    def convert_examples_to_features(self, example_batch):
        # Pass the list of dialogues directly to the tokenizer
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True, padding='max_length')

        with self.tokenizer.as_target_tokenizer():
            # Pass the list of summaries directly to the tokenizer
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True, padding='max_length')

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True) 
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))