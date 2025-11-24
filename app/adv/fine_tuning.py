from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Load training data
train_data = pd.read_csv("train_data.csv")
train_examples = [
    InputExample(texts=[row["query"], row["positive_chunk"]])
    for _, row in train_data.iterrows()
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="./finetuned_model",
    show_progress_bar=True
)

print("Model fine-tuned and saved to ./finetuned_model")