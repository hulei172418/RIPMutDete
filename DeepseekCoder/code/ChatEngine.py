import torch
import gc


class ChatEngine:
    """
    Wrapper class for large model inference and prompt construction:
    - build_prompt: Uses chat_template + truncation
    - chat: Actually calls generate and returns yes/no or raw text
    """
    def __init__(self, model, tokenizer, max_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

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

    def chat(self, instruction: str, content: str, max_new_tokens: int = 10) -> tuple:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]
        inputs = self.build_prompt(messages, max_new_tokens=max_new_tokens)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()} if isinstance(inputs, dict) else inputs.to(self.model.device)


        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1,
                use_cache=False,
            )
            logits = self.model(inputs).logits[0, -1, :]

       # Decode newly generated tokens
        response_tokens = outputs[0][inputs.shape[1]:]
        pred = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        score_non_equiv = None
        yes_ids = self.tokenizer.encode(" yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" no", add_special_tokens=False)
        if len(yes_ids) > 0 and len(no_ids) > 0:
            yes_id, no_id = yes_ids[0], no_ids[0]
            yes_logit = logits[yes_id]
            no_logit = logits[no_id]
            probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
            p_yes = float(probs[0])
            p_no = float(probs[1])
            score_non_equiv = p_no
        
        del inputs, outputs, response_tokens
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        low = pred.lower()
        if "yes" in low and "no" not in low:
            label = "yes"
        elif "no" in low and "yes" not in low:
            label = "no"
        elif low.startswith("yes"):
            label = "yes"
        elif low.startswith("no"):
            label = "no"
        else:
            label = pred
        
        return label, score_non_equiv