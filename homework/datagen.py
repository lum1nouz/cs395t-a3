import json
from pathlib import Path
from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str):
    model = CoTModel()
    data = Dataset("train")
    result = []
    cnt = 0

    for question, correct_answer in data:
        prompt = model.format_prompt(question)
    
        generations = model.batched_generate(
            [prompt], 
            num_return_sequences=1, 
            temperature=0.0
        )[0]

        found = False
        for gen in generations:
            pred = model.parse_answer(gen)
            if is_answer_valid(pred, correct_answer):
                result.append([question, correct_answer, gen])
                found = True
                cnt += 1
                break 

        if not found:

            generations = model.batched_generate(
                [prompt],
                num_return_sequences=10,
                temperature=0.8
            )[0]

            for gen in generations:
                pred = model.parse_answer(gen)
                if is_answer_valid(pred, correct_answer):
                    result.append([question, correct_answer, gen])
                    found = True
                    cnt += 1
                    break
                
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
