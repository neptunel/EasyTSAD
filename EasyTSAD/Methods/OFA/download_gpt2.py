from transformers import GPT2Model, GPT2Config

# Download and save the GPT2 model
model = GPT2Model.from_pretrained('gpt2')
model.save_pretrained('./pre_train/gpt2')
print("GPT2 model downloaded and saved to ./pre_train/gpt2 directory") 