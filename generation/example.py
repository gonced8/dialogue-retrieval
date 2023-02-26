from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gonced8/godel-multiwoz")
model = AutoModelForSeq2SeqLM.from_pretrained("gonced8/godel-multiwoz")

# Encoder input
context = [
    "USER: I need train reservations from norwich to cambridge",
    "SYSTEM: I have 133 trains matching your request. Is there a specific day and time you would like to travel?",
    "USER: I'd like to leave on Monday and arrive by 18:00.",
]

input_text = " EOS ".join(context[-5:]) + " => "

model_inputs = tokenizer(
    input_text, max_length=512, truncation=True, return_tensors="pt"
)["input_ids"]

# Decoder input
answer_start = "SYSTEM: "

decoder_input_ids = tokenizer(
    "<pad>" + answer_start,
    max_length=256,
    truncation=True,
    add_special_tokens=False,
    return_tensors="pt",
)["input_ids"]

# Generate
output = model.generate(
    model_inputs, decoder_input_ids=decoder_input_ids, max_length=256
)
output = tokenizer.decode(
    output[0], clean_up_tokenization_spaces=True, skip_special_tokens=True
)

print(output)
# SYSTEM: TR4634 arrives at 17:35. Would you like me to book that for you?
