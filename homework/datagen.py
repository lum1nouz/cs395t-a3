import json
from pathlib import Path
from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    data = Dataset("train")
    result = []

    for question, correct_answer in data:
        prompt = model.format_prompt(question)
        generations = model.batched_generate(
            [prompt], 
            num_return_sequences=oversample, 
            temperature=temperature
        )[0]

        for gen in generations:
            pred = model.parse_answer(gen)
            if is_answer_valid(pred, correct_answer):
                result.append([question, correct_answer, gen])
                break 

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
