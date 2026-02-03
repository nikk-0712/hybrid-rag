import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import sys

MODEL_DIR = "/home/cemilac/Documents/rag_multi_model/models/llama3.1-8b-instruct-onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

session = ort.InferenceSession(
    MODEL_DIR + "/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

def generate(prompt, max_new_tokens=200):
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"]

    outputs = session.run(None, {"input_ids": input_ids})

    logits = outputs[0]
    next_id = int(np.argmax(logits[0, -1]))
    generated = [next_id]

    for _ in range(max_new_tokens - 1):
        new_tokens = np.array([[generated[-1]]], dtype=np.int64)
        outputs = session.run(None, {"input_ids": new_tokens})
        next_id = int(np.argmax(outputs[0][0, -1]))
        generated.append(next_id)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = sys.argv[1]
    print(generate(prompt))
