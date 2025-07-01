import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Text data
text = """Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks."""


def tokenize(text):
    """Tokenize text into words."""
    pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
    return pattern.findall(text.lower())


def create_vocabulary(tokens):
    """Create word-to-id and id-to-word mappings."""
    word_to_id = {}
    id_to_word = {}
    
    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token
    
    return word_to_id, id_to_word


def generate_training_data(tokens, word_to_id, window_size):
    """Generate skip-gram training pairs."""
    pairs = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        # Get context indices
        context_indices = list(range(max(0, i - window_size), i)) + \
                         list(range(i + 1, min(n_tokens, i + window_size + 1)))
        
        # Create pairs (center_word, context_word)
        for j in context_indices:
            pairs.append((word_to_id[tokens[i]], word_to_id[tokens[j]]))
    
    return pairs


class Word2Vec(nn.Module):
    """Simple Word2Vec model using one-hot encoding."""
    
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # Embedding layers (equivalent to w1 and w2 in original)
        self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Initialize weights (PyTorch does this automatically, but we can customize)
        nn.init.normal_(self.embedding.weight)
        nn.init.normal_(self.output.weight)
    
    def forward(self, x):
        # x is one-hot encoded
        hidden = self.embedding(x)  # Shape: (batch_size, embedding_dim)
        output = self.output(hidden)  # Shape: (batch_size, vocab_size)
        return output


def train_word2vec(model, training_pairs, vocab_size, n_epochs, learning_rate):
    """Train the Word2Vec model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    history = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        # Convert all pairs to tensors at once for efficiency
        center_words = []
        context_words = []
        
        for center, context in training_pairs:
            center_words.append(center)
            context_words.append(context)
        
        # Create one-hot encoded batch for center words
        center_one_hot = torch.zeros(len(center_words), vocab_size)
        for i, word_id in enumerate(center_words):
            center_one_hot[i, word_id] = 1
        
        # Context words as indices for cross-entropy loss
        context_indices = torch.LongTensor(context_words)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(center_one_hot)
        
        # Calculate loss
        loss = criterion(outputs, context_indices)
        total_loss = loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        history.append(total_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss:.4f}')
    
    return history


def get_similar_words(model, word, word_to_id, id_to_word, vocab_size, top_k=10):
    """Get most similar words to a given word."""
    word_id = word_to_id[word]
    
    # Create one-hot encoding for the word
    word_one_hot = torch.zeros(1, vocab_size)
    word_one_hot[0, word_id] = 1
    
    # Get predictions
    with torch.no_grad():
        output = model(word_one_hot)
        probabilities = torch.softmax(output, dim=1).squeeze()
    
    # Get top k words
    values, indices = torch.topk(probabilities, top_k)
    
    similar_words = [(id_to_word[idx.item()], values[i].item()) 
                     for i, idx in enumerate(indices)]
    
    return similar_words


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Tokenize and create vocabulary
    tokens = tokenize(text)
    word_to_id, id_to_word = create_vocabulary(tokens)
    vocab_size = len(word_to_id)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens: {len(tokens)}")
    
    # Generate training data
    window_size = 2
    training_pairs = generate_training_data(tokens, word_to_id, window_size)
    print(f"Training pairs: {len(training_pairs)}")
    
    # Create and train model
    embedding_dim = 10
    model = Word2Vec(vocab_size, embedding_dim)
    
    # Training parameters
    n_epochs = 1000
    learning_rate = 0.05
    
    # Train the model
    print("\nTraining Word2Vec model...")
    history = train_word2vec(model, training_pairs, vocab_size, n_epochs, learning_rate)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, color='skyblue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Word2Vec Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Get similar words for "learning"
    print("\nWords most similar to 'learning':")
    similar_words = get_similar_words(model, "learning", word_to_id, id_to_word, vocab_size, top_k=vocab_size)
    
    for word, score in similar_words:
        print(f"{word}: {score:.4f}")
    
    # Additional analysis: Get embeddings for visualization
    print("\n\nEmbedding weights shape:")
    print(f"Embedding layer (W1): {model.embedding.weight.shape}")
    print(f"Output layer (W2): {model.output.weight.shape}")
    
    # You can access the learned embeddings
    embeddings = model.embedding.weight.data.numpy()
    print(f"\nLearned embeddings shape: {embeddings.shape}")