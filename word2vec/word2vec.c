#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STRING_LENGTH 1000
#define MAX_VOCAB_SIZE 10000
#define MAX_SENTENCE_LENGTH 10000

// Structures
typedef struct {
    char** words;
    int* word_ids;
    int count;
} Vocabulary;

typedef struct {
    int center_word;
    int context_word;
} TrainingPair;

typedef struct {
    TrainingPair* pairs;
    int count;
} TrainingData;

typedef struct {
    float** embedding_weights; // W1: vocab_size x embedding_dim
    float** output_weights;    // W2: embedding_dim x vocab_size
    int vocab_size;
    int embedding_dim;
} Word2VecModel;

// Function prototypes
char* lowercase(char* str);
int tokenize(const char* text, char tokens[][MAX_STRING_LENGTH]);
Vocabulary* create_vocabulary(char tokens[][MAX_STRING_LENGTH],
                              int token_count);
TrainingData* generate_training_data(char tokens[][MAX_STRING_LENGTH],
                                     int token_count, Vocabulary* vocab,
                                     int window_size);
Word2VecModel* create_model(int vocab_size, int embedding_dim);
void free_model(Word2VecModel* model);
float** allocate_matrix(int rows, int cols);
void free_matrix(float** matrix, int rows);
void initialize_weights(float** matrix, int rows, int cols);
void forward_pass(Word2VecModel* model, int center_word_id, float* hidden,
                  float* output);
void softmax(float* x, int size);
float cross_entropy_loss(float* predictions, int target, int size);
void backward_pass(Word2VecModel* model, int center_word_id,
                   int context_word_id, float* hidden, float* output,
                   float learning_rate);
void train_word2vec(Word2VecModel* model, TrainingData* training_data,
                    int n_epochs, float learning_rate);
void get_similar_words(Word2VecModel* model, const char* word,
                       Vocabulary* vocab, int top_k);

// Utility functions
char* lowercase(char* str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
    return str;
}

int tokenize(const char* text, char tokens[][MAX_STRING_LENGTH]) {
    int token_count = 0;
    char buffer[MAX_STRING_LENGTH];
    int buf_index = 0;

    for (int i = 0; text[i] != '\0'; i++) {
        if (isalpha(text[i])) {
            buffer[buf_index++] = tolower(text[i]);
        } else if (buf_index > 0) {
            buffer[buf_index] = '\0';
            strcpy(tokens[token_count++], buffer);
            buf_index = 0;
        }
    }

    if (buf_index > 0) {
        buffer[buf_index] = '\0';
        strcpy(tokens[token_count++], buffer);
    }

    return token_count;
}

Vocabulary* create_vocabulary(char tokens[][MAX_STRING_LENGTH],
                              int token_count) {
    Vocabulary* vocab = (Vocabulary*)malloc(sizeof(Vocabulary));
    vocab->words = (char**)malloc(MAX_VOCAB_SIZE * sizeof(char*));
    vocab->word_ids = (int*)malloc(MAX_VOCAB_SIZE * sizeof(int));
    vocab->count = 0;

    // Create unique word list
    for (int i = 0; i < token_count; i++) {
        int found = 0;
        for (int j = 0; j < vocab->count; j++) {
            if (strcmp(tokens[i], vocab->words[j]) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            vocab->words[vocab->count] =
                (char*)malloc((strlen(tokens[i]) + 1) * sizeof(char));
            strcpy(vocab->words[vocab->count], tokens[i]);
            vocab->word_ids[vocab->count] = vocab->count;
            vocab->count++;
        }
    }

    return vocab;
}

int get_word_id(Vocabulary* vocab, const char* word) {
    for (int i = 0; i < vocab->count; i++) {
        if (strcmp(vocab->words[i], word) == 0) {
            return i;
        }
    }
    return -1;
}

TrainingData* generate_training_data(char tokens[][MAX_STRING_LENGTH],
                                     int token_count, Vocabulary* vocab,
                                     int window_size) {
    TrainingData* data = (TrainingData*)malloc(sizeof(TrainingData));
    data->pairs = (TrainingPair*)malloc(token_count * window_size * 2 *
                                        sizeof(TrainingPair));
    data->count = 0;

    for (int i = 0; i < token_count; i++) {
        int center_id = get_word_id(vocab, tokens[i]);

        // Context before
        for (int j = i - window_size; j < i; j++) {
            if (j >= 0) {
                int context_id = get_word_id(vocab, tokens[j]);
                data->pairs[data->count].center_word = center_id;
                data->pairs[data->count].context_word = context_id;
                data->count++;
            }
        }

        // Context after
        for (int j = i + 1; j <= i + window_size; j++) {
            if (j < token_count) {
                int context_id = get_word_id(vocab, tokens[j]);
                data->pairs[data->count].center_word = center_id;
                data->pairs[data->count].context_word = context_id;
                data->count++;
            }
        }
    }

    return data;
}

float** allocate_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)calloc(cols, sizeof(float));
    }
    return matrix;
}

void free_matrix(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initialize_weights(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = ((float)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
}

Word2VecModel* create_model(int vocab_size, int embedding_dim) {
    Word2VecModel* model = (Word2VecModel*)malloc(sizeof(Word2VecModel));
    model->vocab_size = vocab_size;
    model->embedding_dim = embedding_dim;

    model->embedding_weights = allocate_matrix(vocab_size, embedding_dim);
    model->output_weights = allocate_matrix(embedding_dim, vocab_size);

    initialize_weights(model->embedding_weights, vocab_size, embedding_dim);
    initialize_weights(model->output_weights, embedding_dim, vocab_size);

    return model;
}

void free_model(Word2VecModel* model) {
    free_matrix(model->embedding_weights, model->vocab_size);
    free_matrix(model->output_weights, model->embedding_dim);
    free(model);
}

void forward_pass(Word2VecModel* model, int center_word_id, float* hidden,
                  float* output) {
    // Hidden layer = embedding weights for the center word (one-hot * W1)
    for (int i = 0; i < model->embedding_dim; i++) {
        hidden[i] = model->embedding_weights[center_word_id][i];
    }

    // Output layer = hidden * W2
    for (int i = 0; i < model->vocab_size; i++) {
        output[i] = 0;
        for (int j = 0; j < model->embedding_dim; j++) {
            output[i] += hidden[j] * model->output_weights[j][i];
        }
    }
}

void softmax(float* x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max)
            max = x[i];
    }

    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float cross_entropy_loss(float* predictions, int target, int size) {
    return -log(predictions[target] + 1e-10);
}

void backward_pass(Word2VecModel* model, int center_word_id,
                   int context_word_id, float* hidden, float* output,
                   float learning_rate) {
    // Apply softmax to output
    softmax(output, model->vocab_size);

    // Calculate output error (prediction - target)
    float* output_error = (float*)malloc(model->vocab_size * sizeof(float));
    for (int i = 0; i < model->vocab_size; i++) {
        output_error[i] = output[i];
    }
    output_error[context_word_id] -= 1.0;

    // Update output weights (W2)
    for (int i = 0; i < model->embedding_dim; i++) {
        for (int j = 0; j < model->vocab_size; j++) {
            model->output_weights[i][j] -=
                learning_rate * hidden[i] * output_error[j];
        }
    }

    // Calculate hidden error
    float* hidden_error = (float*)calloc(model->embedding_dim, sizeof(float));
    for (int i = 0; i < model->embedding_dim; i++) {
        for (int j = 0; j < model->vocab_size; j++) {
            hidden_error[i] += output_error[j] * model->output_weights[i][j];
        }
    }

    // Update embedding weights (W1) - only for the center word
    for (int i = 0; i < model->embedding_dim; i++) {
        model->embedding_weights[center_word_id][i] -=
            learning_rate * hidden_error[i];
    }

    free(output_error);
    free(hidden_error);
}

void train_word2vec(Word2VecModel* model, TrainingData* training_data,
                    int n_epochs, float learning_rate) {
    float* hidden = (float*)malloc(model->embedding_dim * sizeof(float));
    float* output = (float*)malloc(model->vocab_size * sizeof(float));

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        float total_loss = 0;

        for (int i = 0; i < training_data->count; i++) {
            int center = training_data->pairs[i].center_word;
            int context = training_data->pairs[i].context_word;

            // Forward pass
            forward_pass(model, center, hidden, output);

            // Calculate loss before softmax for monitoring
            float* softmax_output =
                (float*)malloc(model->vocab_size * sizeof(float));
            memcpy(softmax_output, output, model->vocab_size * sizeof(float));
            softmax(softmax_output, model->vocab_size);
            total_loss +=
                cross_entropy_loss(softmax_output, context, model->vocab_size);
            free(softmax_output);

            // Backward pass and update weights
            backward_pass(model, center, context, hidden, output,
                          learning_rate);
        }

        if ((epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.4f\n", epoch + 1, n_epochs,
                   total_loss / training_data->count);
        }
    }

    free(hidden);
    free(output);
}

void get_similar_words(Word2VecModel* model, const char* word,
                       Vocabulary* vocab, int top_k) {
    int word_id = get_word_id(vocab, word);
    if (word_id == -1) {
        printf("Word '%s' not found in vocabulary\n", word);
        return;
    }

    float* hidden = (float*)malloc(model->embedding_dim * sizeof(float));
    float* output = (float*)malloc(model->vocab_size * sizeof(float));

    forward_pass(model, word_id, hidden, output);
    softmax(output, model->vocab_size);

    // Find top k words
    typedef struct {
        int id;
        float score;
    } WordScore;

    WordScore* scores =
        (WordScore*)malloc(model->vocab_size * sizeof(WordScore));
    for (int i = 0; i < model->vocab_size; i++) {
        scores[i].id = i;
        scores[i].score = output[i];
    }

    // Sort by score (simple bubble sort for demonstration)
    for (int i = 0; i < model->vocab_size - 1; i++) {
        for (int j = 0; j < model->vocab_size - i - 1; j++) {
            if (scores[j].score < scores[j + 1].score) {
                WordScore temp = scores[j];
                scores[j] = scores[j + 1];
                scores[j + 1] = temp;
            }
        }
    }

    printf("\nWords most similar to '%s':\n", word);
    for (int i = 0; i < top_k && i < model->vocab_size; i++) {
        printf("%s: %.4f\n", vocab->words[scores[i].id], scores[i].score);
    }

    free(scores);
    free(hidden);
    free(output);
}

int main() {
    // Set random seed
    srand(42);

    // Sample text
    const char* text =
        "Machine learning is the study of computer algorithms that "
        "improve automatically through experience. It is seen as a "
        "subset of artificial intelligence. Machine learning algorithms "
        "build a mathematical model based on sample data, known as "
        "training data, in order to make predictions or decisions without "
        "being explicitly programmed to do so. Machine learning algorithms "
        "are used in a wide variety of applications, such as email filtering "
        "and computer vision, where it is difficult or infeasible to develop "
        "conventional algorithms to perform the needed tasks.";

    // Tokenize text
    char tokens[MAX_SENTENCE_LENGTH][MAX_STRING_LENGTH];
    int token_count = tokenize(text, tokens);
    printf("Total tokens: %d\n", token_count);

    // Create vocabulary
    Vocabulary* vocab = create_vocabulary(tokens, token_count);
    printf("Vocabulary size: %d\n", vocab->count);

    // Generate training data
    int window_size = 2;
    TrainingData* training_data =
        generate_training_data(tokens, token_count, vocab, window_size);
    printf("Training pairs: %d\n", training_data->count);

    // Create and train model
    int embedding_dim = 10;
    Word2VecModel* model = create_model(vocab->count, embedding_dim);

    // Training parameters
    int n_epochs = 1000;
    float learning_rate = 0.05;

    printf("\nTraining Word2Vec model...\n");
    train_word2vec(model, training_data, n_epochs, learning_rate);

    // Get similar words
    get_similar_words(model, "learning", vocab, vocab->count);

    // Cleanup
    for (int i = 0; i < vocab->count; i++) {
        free(vocab->words[i]);
    }
    free(vocab->words);
    free(vocab->word_ids);
    free(vocab);

    free(training_data->pairs);
    free(training_data);

    free_model(model);

    return 0;
}