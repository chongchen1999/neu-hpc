// main.cpp
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Matrix class for handling 2D arrays
class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    void randomInit(double mean = 0.0, double std = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(mean, std);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dis(gen);
            }
        }
    }

    std::vector<double> multiply(const std::vector<double>& vec) const {
        std::vector<double> result(rows, 0.0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += data[i][j] * vec[j];
            }
        }
        return result;
    }

    std::vector<double>
    multiplyTranspose(const std::vector<double>& vec) const {
        std::vector<double> result(cols, 0.0);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                result[j] += data[i][j] * vec[i];
            }
        }
        return result;
    }
};

// Tokenization function
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::regex word_regex("[A-Za-z]+[\\w^']*|[\\w^']*[A-Za-z]+[\\w^']*");
    std::sregex_iterator it(text.begin(), text.end(), word_regex);
    std::sregex_iterator end;

    while (it != end) {
        std::string token = it->str();
        // Convert to lowercase
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        tokens.push_back(token);
        ++it;
    }

    return tokens;
}

// Create vocabulary mappings
void createVocabulary(const std::vector<std::string>& tokens,
                      std::unordered_map<std::string, int>& word_to_id,
                      std::unordered_map<int, std::string>& id_to_word) {
    std::unordered_set<std::string> unique_tokens(tokens.begin(), tokens.end());
    int id = 0;

    for (const auto& token : unique_tokens) {
        word_to_id[token] = id;
        id_to_word[id] = token;
        id++;
    }
}

// Generate training pairs for skip-gram
std::vector<std::pair<int, int>>
generateTrainingData(const std::vector<std::string>& tokens,
                     const std::unordered_map<std::string, int>& word_to_id,
                     int window_size) {

    std::vector<std::pair<int, int>> pairs;
    int n_tokens = tokens.size();

    for (int i = 0; i < n_tokens; i++) {
        int center_id = word_to_id.at(tokens[i]);

        // Get context words
        int start = std::max(0, i - window_size);
        int end = std::min(n_tokens, i + window_size + 1);

        for (int j = start; j < end; j++) {
            if (j != i) {
                int context_id = word_to_id.at(tokens[j]);
                pairs.push_back({center_id, context_id});
            }
        }
    }

    return pairs;
}

// Softmax function
std::vector<double> softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;

    for (size_t i = 0; i < x.size(); i++) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }

    for (size_t i = 0; i < x.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

// Word2Vec model class
class Word2Vec {
private:
    Matrix embedding; // W1: vocab_size x embedding_dim
    Matrix output;    // W2: embedding_dim x vocab_size
    int vocab_size;
    int embedding_dim;

public:
    Word2Vec(int vocab_sz, int embed_dim)
        : vocab_size(vocab_sz), embedding_dim(embed_dim),
          embedding(embed_dim, vocab_sz), output(vocab_sz, embed_dim) {
        // Initialize weights with normal distribution
        embedding.randomInit(0.0, 0.1);
        output.randomInit(0.0, 0.1);
    }

    // Forward pass
    std::vector<double> forward(const std::vector<double>& one_hot) {
        // Hidden layer: W1 * one_hot
        std::vector<double> hidden = embedding.multiply(one_hot);

        // Output layer: W2 * hidden
        std::vector<double> output_vec = output.multiply(hidden);

        return output_vec;
    }

    // Train the model
    void train(const std::vector<std::pair<int, int>>& training_pairs,
               int n_epochs, double learning_rate) {

        std::vector<double> history;

        for (int epoch = 0; epoch < n_epochs; epoch++) {
            double total_loss = 0.0;

            for (const auto& pair : training_pairs) {
                int center_word = pair.first;
                int context_word = pair.second;

                // Create one-hot vector for center word
                std::vector<double> one_hot(vocab_size, 0.0);
                one_hot[center_word] = 1.0;

                // Forward pass
                std::vector<double> output_vec = forward(one_hot);
                std::vector<double> probs = softmax(output_vec);

                // Calculate cross-entropy loss
                double loss = -std::log(probs[context_word] + 1e-10);
                total_loss += loss;

                // Backward pass
                // Gradient of loss w.r.t output layer
                std::vector<double> grad_output = probs;
                grad_output[context_word] -= 1.0;

                // Get hidden layer
                std::vector<double> hidden = embedding.multiply(one_hot);

                // Update output weights (W2)
                for (int i = 0; i < vocab_size; i++) {
                    for (int j = 0; j < embedding_dim; j++) {
                        output.data[i][j] -=
                            learning_rate * grad_output[i] * hidden[j];
                    }
                }

                // Gradient w.r.t hidden layer
                std::vector<double> grad_hidden =
                    output.multiplyTranspose(grad_output);

                // Update embedding weights (W1)
                for (int i = 0; i < embedding_dim; i++) {
                    for (int j = 0; j < vocab_size; j++) {
                        embedding.data[i][j] -=
                            learning_rate * grad_hidden[i] * one_hot[j];
                    }
                }
            }

            history.push_back(total_loss / training_pairs.size());

            if ((epoch + 1) % 100 == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << n_epochs
                          << "], Loss: " << std::fixed << std::setprecision(4)
                          << total_loss / training_pairs.size() << std::endl;
            }
        }

        // Save loss history to file for plotting
        std::ofstream file("loss_history.txt");
        for (size_t i = 0; i < history.size(); i++) {
            file << i << " " << history[i] << std::endl;
        }
        file.close();
    }

    // Get similar words
    std::vector<std::pair<std::string, double>>
    getSimilarWords(const std::string& word,
                    const std::unordered_map<std::string, int>& word_to_id,
                    const std::unordered_map<int, std::string>& id_to_word,
                    int top_k = 10) {

        if (word_to_id.find(word) == word_to_id.end()) {
            return {};
        }

        int word_id = word_to_id.at(word);

        // Create one-hot vector
        std::vector<double> one_hot(vocab_size, 0.0);
        one_hot[word_id] = 1.0;

        // Get predictions
        std::vector<double> output_vec = forward(one_hot);
        std::vector<double> probs = softmax(output_vec);

        // Create pairs of (word, probability)
        std::vector<std::pair<std::string, double>> word_probs;
        for (int i = 0; i < vocab_size; i++) {
            word_probs.push_back({id_to_word.at(i), probs[i]});
        }

        // Sort by probability
        std::sort(
            word_probs.begin(), word_probs.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // Return top k
        std::vector<std::pair<std::string, double>> result;
        for (int i = 0; i < std::min(top_k, vocab_size); i++) {
            result.push_back(word_probs[i]);
        }

        return result;
    }

    // Get embedding for a word
    std::vector<double> getEmbedding(int word_id) {
        std::vector<double> embedding_vec(embedding_dim);
        for (int i = 0; i < embedding_dim; i++) {
            embedding_vec[i] = embedding.data[i][word_id];
        }
        return embedding_vec;
    }
};

int main() {
    // Text data
    std::string text =
        "Machine learning is the study of computer algorithms that "
        "improve automatically through experience. It is seen as a "
        "subset of artificial intelligence. Machine learning algorithms "
        "build a mathematical model based on sample data, known as "
        "training data, in order to make predictions or decisions without "
        "being explicitly programmed to do so. Machine learning algorithms "
        "are used in a wide variety of applications, such as email filtering "
        "and computer vision, where it is difficult or infeasible to develop "
        "conventional algorithms to perform the needed tasks.";

    // Set random seed
    std::srand(42);

    // Tokenize and create vocabulary
    std::vector<std::string> tokens = tokenize(text);
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    createVocabulary(tokens, word_to_id, id_to_word);

    int vocab_size = word_to_id.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    std::cout << "Total tokens: " << tokens.size() << std::endl;

    // Generate training data
    int window_size = 2;
    auto training_pairs = generateTrainingData(tokens, word_to_id, window_size);
    std::cout << "Training pairs: " << training_pairs.size() << std::endl;

    // Create and train model
    int embedding_dim = 10;
    Word2Vec model(vocab_size, embedding_dim);

    // Training parameters
    int n_epochs = 1000;
    double learning_rate = 0.05;

    // Train the model
    std::cout << "\nTraining Word2Vec model..." << std::endl;
    model.train(training_pairs, n_epochs, learning_rate);

    // Get similar words for "learning"
    std::cout << "\nWords most similar to 'learning':" << std::endl;
    auto similar_words =
        model.getSimilarWords("learning", word_to_id, id_to_word, vocab_size);

    for (const auto& pair : similar_words) {
        std::cout << pair.first << ": " << std::fixed << std::setprecision(4)
                  << pair.second << std::endl;
    }

    std::cout << "\nTraining complete! Loss history saved to 'loss_history.txt'"
              << std::endl;
    std::cout << "You can plot it using: gnuplot -e \"plot 'loss_history.txt' "
                 "with lines; pause -1\""
              << std::endl;

    return 0;
}