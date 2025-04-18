from .base_llm import BaseLLM
from .data import Dataset, benchmark
from .sft import tokenize
import torch

def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example(prompt: str, answer: str, reasoning: str) -> dict[str, str]:
    return {
        "question": prompt.strip(),
        "answer": reasoning.strip()
    }

class TokenizedDataset:
    """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
    """
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted)

def train_model(
    output_dir: str,
    **kwargs,
):
    from transformers import TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig

    model = BaseLLM()
    tokenizer = model.tokenizer

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model.model = get_peft_model(model.model, peft_config)

    if torch.cuda.is_available():
        model.model.enable_input_require_grads()

    train_data = Dataset("rft")
    tokenized = TokenizedDataset(tokenizer, train_data, format_example)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        num_train_epochs=10,
        logging_dir=output_dir,
        report_to="tensorboard",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model.model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(output_dir)

    test_model(output_dir)

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
