from openai import OpenAI

class ChatEngine:
    """
    Wrapper class for large model inference and prompt construction:
    - build_prompt: Uses chat_template + truncation
    - chat: Actually calls generate and returns yes/no or raw text
    """
    def __init__(self, api_key, base_url, max_tokens, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens = max_tokens
        self.model=model
        self.temperature = 0
        self.top_p=1
        self.frequency_penalty=0
        self.presence_penalty=0

    def build_prompt(self, messages, max_new_tokens: int = 3):
        tmp = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        max_input_len = self.max_tokens - max_new_tokens
        if tmp.shape[1] > max_input_len:
            # Truncate from the left, keeping the most recent tokens
            tmp = tmp[:, -max_input_len:]
        return tmp

    def chat(self, instruction: str, content: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "system",
                "content": instruction
                },
                {
                "role": "user",
                "content": content
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty)
        pred = completion.choices[0].message.content

        low = pred.lower()
        if "yes" in low and "no" not in low:
            return "yes"
        if "no" in low and "yes" not in low:
            return "no"
        if low.startswith("yes"):
            return "yes"
        if low.startswith("no"):
            return "no"
        return pred

