from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

# Global Constants
MODEL_NAME = 'distilgpt2'  # Use a smaller model
TRAIN_FILE = 'input.txt'
OUTPUT_DIR = 'output_directory'
EPOCHS = 1  # Reduce the number of epochs
BATCH_SIZE = 4  # Reduce batch size for your GPU
BLOCK_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = 1
NUM_WORKERS = 4
FP16 = True  # Enable mixed precision
LEARNING_RATE = 5e-5

def load_dataset(file_path, tokenizer, block_size=BLOCK_SIZE):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def train_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    dataset = load_dataset(TRAIN_FILE, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    accelerator = Accelerator()
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_steps=10_000,
        save_total_limit=2,
        fp16=FP16,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Prepare the model, optimizer, and dataloader with Accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

# Fine-tune model with optimizations
train_model()
