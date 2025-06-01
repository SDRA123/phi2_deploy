import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import runpod

# Load model and tokenizer
MODEL_NAME = "microsoft/phi-2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.to(device)

print("Model loaded successfully.")

# Handler function
def handler(job):
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "").strip()

        if not prompt:
            return {"error": "No 'prompt' provided in input."}

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}


        # Generate text
        outputs = model.generate(
            **inputs,
            max_length=200,
        )

        # Decode output
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return str(generated_text)

    except Exception as e:
        return {"error": str(e)}

# Start the serverless handler
runpod.serverless.start({"handler": handler})
