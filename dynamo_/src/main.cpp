#include <torch/torch.h>
#include <iostream>
#include <vector>

// Helper to check for CUDA availability
struct Options {
  torch::Device device = torch::kCPU;
};

// 1. Multi-Head Self-Attention Block
// ------------------------------------
// This is the core of the Transformer. It allows the model to weigh the importance
// of different patches in the sequence relative to each other.
struct MultiHeadAttentionImpl : torch::nn::Module {
  int num_heads;
  int head_dim;
  double scale;
  torch::nn::Linear qkv{nullptr};
  torch::nn::Linear proj{nullptr};
  torch::nn::Dropout attn_drop;
  torch::nn::Dropout proj_drop;

  MultiHeadAttentionImpl(int embed_dim, int num_heads, double dropout_p = 0.0)
      : num_heads(num_heads), attn_drop(dropout_p), proj_drop(dropout_p) {
    TORCH_CHECK(embed_dim % num_heads == 0,
                "Embedding dimension must be divisible by number of heads");
    head_dim = embed_dim / num_heads;
    scale = std::pow(head_dim, -0.5);

    // A single linear layer to project input to Q, K, V for efficiency
    qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
    // Output projection layer
    proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
  }

  torch::Tensor forward(torch::Tensor x) {
    int64_t B = x.size(0);  // Batch size
    int64_t N = x.size(1);  // Sequence length
    int64_t C = x.size(2);  // Embedding dimension

    // 1. Project to Q, K, V and reshape for multi-head
    // x: [B, N, C] -> qkv_proj: [B, N, 3*C]
    auto qkv_proj = qkv->forward(x);

    // Reshape to [B, N, 3, num_heads, head_dim] and permute to [3, B, num_heads, N, head_dim]
    qkv_proj = qkv_proj.reshape({B, N, 3, num_heads, head_dim})
                   .permute({2, 0, 3, 1, 4});

    // Extract Q, K, V
    auto q = qkv_proj[0];  // [B, num_heads, N, head_dim]
    auto k = qkv_proj[1];  // [B, num_heads, N, head_dim]
    auto v = qkv_proj[2];  // [B, num_heads, N, head_dim]

    // 2. Scaled Dot-Product Attention
    // (q @ k.transpose) * scale
    // attn: [B, num_heads, N, N]
    auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale;
    attn = torch::softmax(attn, -1);
    attn = attn_drop(attn);

    // 3. Attend to values
    // x: [B, num_heads, N, head_dim]
    auto x_attended = torch::matmul(attn, v);

    // 4. Reshape and project back to original embedding dimension
    // [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C]
    x_attended = x_attended.transpose(1, 2).reshape({B, N, C});

    x_attended = proj(x_attended);
    x_attended = proj_drop(x_attended);

    return x_attended;
  }
};

TORCH_MODULE(MultiHeadAttention);

// 2. MLP (Feed-Forward) Block
// -----------------------------
// A simple two-layer fully connected network with GELU activation.
struct MLPImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::nn::Dropout dropout;

  MLPImpl(int in_features, int hidden_features, double dropout_p = 0.0)
      : dropout(dropout_p) {
    fc1 =
        register_module("fc1", torch::nn::Linear(in_features, hidden_features));
    fc2 =
        register_module("fc2", torch::nn::Linear(hidden_features, in_features));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = fc1(x);
    x = torch::gelu(x, "tanh");  // GELU activation is standard in ViT
    x = dropout(x);
    x = fc2(x);
    x = dropout(x);
    return x;
  }
};

TORCH_MODULE(MLP);

// 3. Transformer Encoder Block
// ------------------------------
// This combines MultiHeadAttention and MLP with Layer Normalization and
// residual connections. We use Pre-LayerNorm for better stability.
struct EncoderBlockImpl : torch::nn::Module {
  torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  MultiHeadAttention attn{nullptr};
  MLP mlp{nullptr};

  EncoderBlockImpl(int embed_dim, int num_heads, int mlp_dim,
                   double dropout_p = 0.0) {
    norm1 = register_module(
        "norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    norm2 = register_module(
        "norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    attn = register_module("attn",
                           MultiHeadAttention(embed_dim, num_heads, dropout_p));
    mlp = register_module("mlp", MLP(embed_dim, mlp_dim, dropout_p));
  }

  torch::Tensor forward(torch::Tensor x) {
    // Pre-LN: x + Attention(LayerNorm(x))
    auto norm_x = norm1(x);
    auto attn_output = attn(norm_x);
    x = x + attn_output;

    // Pre-LN: x + MLP(LayerNorm(x))
    auto norm_x_2 = norm2(x);
    auto mlp_output = mlp(norm_x_2);
    x = x + mlp_output;

    return x;
  }
};

TORCH_MODULE(EncoderBlock);

// 4. Patch Embedding
// --------------------
// Converts an image [B, C, H, W] into a sequence of patch embeddings [B, N, D]
// This is achieved with a Conv2d layer with stride equal to kernel size.
struct PatchEmbeddingImpl : torch::nn::Module {
  torch::nn::Conv2d proj{nullptr};
  int num_patches;

  PatchEmbeddingImpl(int img_size, int patch_size, int in_channels,
                     int embed_dim) {
    TORCH_CHECK(img_size % patch_size == 0,
                "Image dimensions must be divisible by the patch size.");
    num_patches = (img_size / patch_size) * (img_size / patch_size);

    proj = register_module(
        "proj", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, embed_dim, patch_size)
                        .stride(patch_size)));
  }

  torch::Tensor forward(torch::Tensor x) {
    // x: [B, C, H, W] -> proj: [B, D, H/P, W/P]
    x = proj(x);

    // Flatten the spatial dimensions: [B, D, H/P, W/P] -> [B, D, N]
    x = x.flatten(2);

    // Transpose to get [B, N, D] which is the standard sequence format for transformers
    x = x.transpose(1, 2);

    return x;
  }
};

TORCH_MODULE(PatchEmbedding);

// 5. Vision Transformer (ViT)
// -----------------------------
// The main model that assembles all the building blocks.
struct VisionTransformerImpl : torch::nn::Module {
  PatchEmbedding patch_embed{nullptr};
  torch::Tensor cls_token;
  torch::Tensor pos_embed;
  torch::nn::Dropout pos_drop;
  torch::nn::ModuleList blocks;
  torch::nn::LayerNorm norm{nullptr};
  torch::nn::Linear head{nullptr};

  VisionTransformerImpl(int img_size, int patch_size, int in_channels,
                        int num_classes, int embed_dim, int depth,
                        int num_heads, int mlp_dim, double dropout_p = 0.1)
      : pos_drop(dropout_p) {

    // 1. Patch Embedding
    patch_embed = register_module(
        "patch_embed",
        PatchEmbedding(img_size, patch_size, in_channels, embed_dim));
    int num_patches = patch_embed->num_patches;

    // 2. CLS Token
    // A learnable parameter that will be prepended to the sequence of patch embeddings.
    // It's used to aggregate information for the final classification.
    cls_token =
        register_parameter("cls_token", torch::randn({1, 1, embed_dim}));

    // 3. Positional Embedding
    // Learnable parameters to encode spatial information for each patch and the CLS token.
    pos_embed = register_parameter(
        "pos_embed", torch::randn({1, num_patches + 1, embed_dim}));

    // 4. Transformer Encoder Blocks
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int i = 0; i < depth; ++i) {
      blocks->push_back(EncoderBlock(embed_dim, num_heads, mlp_dim, dropout_p));
    }

    // 5. Final LayerNorm and Classification Head
    norm = register_module(
        "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    head = register_module("head", torch::nn::Linear(embed_dim, num_classes));
  }

  torch::Tensor forward(torch::Tensor x) {
    int64_t B = x.size(0);  // Batch size

    // Convert image to patch embeddings: [B, C, H, W] -> [B, N, D]
    x = patch_embed->forward(x);

    // Expand and prepend CLS token: [B, N, D] -> [B, N+1, D]
    auto cls_tokens = cls_token.expand({B, -1, -1});
    x = torch::cat({cls_tokens, x}, 1);

    // Add positional embedding
    x = x + pos_embed;
    x = pos_drop(x);

    // Pass through encoder blocks
    for (const auto& block : *blocks) {
      x = block->as<EncoderBlock>()->forward(x);
    }

    // Apply final layer norm
    x = norm(x);

    // Extract the CLS token output for classification
    auto cls_output = x.select(1, 0);  // or x[:, 0]

    // Pass through the classification head
    return head(cls_output);
  }
};

TORCH_MODULE(VisionTransformer);

int main() {
  std::cout << "Vision Transformer in LibTorch\n\n";

  // Setup device (CUDA if available, otherwise CPU)
  Options opts;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Using GPU." << std::endl;
    opts.device = torch::kCUDA;
  } else {
    std::cout << "Using CPU." << std::endl;
    opts.device = torch::kCPU;
  }

  // --- Model Configuration (ViT-Base-like) ---
  int img_size = 224;
  int patch_size = 16;
  int in_channels = 3;
  int num_classes = 1000;
  int embed_dim = 768;
  int depth = 12;      // Number of Transformer Encoder blocks
  int num_heads = 12;  // Number of attention heads
  int mlp_dim = 3072;  // Dimension of the MLP's hidden layer
  double dropout_p = 0.1;

  // Create the Vision Transformer model
  auto model =
      VisionTransformer(img_size, patch_size, in_channels, num_classes,
                        embed_dim, depth, num_heads, mlp_dim, dropout_p);
  model->to(opts.device);

  long num_params = 0;
  for (const auto& p : model->parameters()) {
    num_params += p.numel();
  }
  std::cout << "Model created successfully." << std::endl;
  std::cout << "Total number of parameters: " << (num_params / 1.0e6) << "M\n"
            << std::endl;

  // --- Create Dummy Data Once ---
  int batch_size = 4;
  auto dummy_input =
      torch::randn({batch_size, in_channels, img_size, img_size}, opts.device);

  // --- Dummy Inference Example (in its own scope) ---
  {  // <--- START OF SCOPE
    std::cout << "--- Running a dummy inference pass ---" << std::endl;
    model->eval();
    torch::NoGradGuard no_grad;  // This guard is now local to this scope

    auto output = model->forward(dummy_input);

    std::cout << "Input shape:  " << dummy_input.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    TORCH_CHECK(output.size(0) == batch_size && output.size(1) == num_classes,
                "Output shape is incorrect!");
    std::cout << "Inference pass successful.\n" << std::endl;
  }  // <--- END OF SCOPE. The `no_grad` guard is destroyed here, and gradients are re-enabled.

  // --- Dummy Training Loop Example ---
  {  // <--- Scoping the training block is also good practice
    std::cout << "--- Running a dummy training loop for one step ---"
              << std::endl;
    model->train();

    auto dummy_labels = torch::randint(
        0, num_classes, {batch_size},
        torch::TensorOptions().dtype(torch::kLong).device(opts.device));

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(1e-4).weight_decay(1e-5));

    // Forward pass will now correctly track gradients
    auto train_output = model->forward(dummy_input);

    auto loss =
        torch::nn::functional::cross_entropy(train_output, dummy_labels);

    optimizer.zero_grad();
    loss.backward();  // This will now work correctly!
    optimizer.step();

    std::cout << "Loss: " << loss.item<float>() << std::endl;
    std::cout << "Training step successful.\n" << std::endl;
  }

  return 0;
}
