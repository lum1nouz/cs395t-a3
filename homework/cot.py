from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that solves math word problems using clear, step-by-step reasoning. "
                "Always explain your steps and end with the final answer in the format <answer>NUMBER</answer>. "
                "Be concise."
            ),
        },
        # Example 1
        {
            "role": "user",
            "content": (
                "A farmer has 3 baskets. Each basket contains 12 apples. "
                "If he gives away 10 apples, how many does he have left?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Each basket has 12 apples, and there are 3 baskets.\n"
                "So total apples = 3 * 12 = 36.\n"
                "He gives away 10 apples.\n"
                "36 - 10 = 26 apples left.\n"
                "<answer>26</answer>"
            ),
        },
        # Example 2
        {
            "role": "user",
            "content": (
                "Lily reads 5 pages of a book each day. After 6 days, how many pages has she read?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "She reads 5 pages a day for 6 days.\n"
                "5 * 6 = 30 pages.\n"
                "<answer>30</answer>"
            ),
        },
        # Example 3
        {
            "role": "user",
            "content": (
                "A train travels 60 miles per hour. How far does it go in 2.5 hours?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The speed is 60 miles per hour and time is 2.5 hours.\n"
                "Distance = 60 * 2.5 = 150 miles.\n"
                "<answer>150</answer>"
            ),
        },
        # Example 4
        {
            "role": "user",
            "content": (
                "John is twice as old as Mary. Mary is 8 years old. How old is John?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Mary is 8 years old. John is twice her age.\n"
                "2 * 8 = 16 years old.\n"
                "<answer>16</answer>"
            ),
        },
        # Example 5
        {
            "role": "user",
            "content": (
                "How does 4 years measure up in terms of week?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "There are 52 weeks in a year.\n"
                "4 * 52 = 208 weeks.\n"
                "<answer>208</answer>"
            ),
        },
        # The actual question to solve
        {
            "role": "user",
            "content": question.strip(),
        },
    ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
