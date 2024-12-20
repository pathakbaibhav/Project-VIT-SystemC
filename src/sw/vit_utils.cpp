#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "vit_utils.h"
//#include "ggml/ggml.h"
//#include "ggml/ggml-alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // stb image load

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>
#include <algorithm>
#include <iostream>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// default ViT-B hparams
// vit_base_patch8_224.augreg2_in21k_ft_in1k from timm
int32_t vit_hparams::n_enc_head_dim() const
{
    return hidden_size / num_attention_heads;
}

int32_t vit_hparams::n_img_size() const
{
    return img_size;
}

int32_t vit_hparams::n_patch_size() const
{
    return patch_size;
}

int32_t vit_hparams::n_img_embd() const
{
    return n_img_size() / n_patch_size();
}

//
// Helpers
//

void print_t_f32(const char *title, struct ggml_tensor *t, int n = 10)
{
    printf("%s\n", title);
    float *data = (float *)t->data;
    printf("dims: % " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("First & Last %d elements:\n", n);
    for (int i = 0; i < std::min((int)(t->ne[0] * t->ne[1]), n); i++)
    {
        printf("%.5f ", data[i]);
        if (i != 0 && i % t->ne[0] == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 0; i < std::min((int)(t->ne[0] * t->ne[1]), n); i++)
    {
        printf("%.5f ", data[ggml_nelements(t) - n + i]);
        if ((ggml_nelements(t) - n + i) % t->ne[0] == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    double sum = 0.0;
    for (int i = 0; i < ggml_nelements(t); i++)
    {
        sum += data[i];
    }
    printf("sum:  %f\n\n", sum);
}

static void ggml_disconnect_node_from_graph(ggml_tensor *t)
{
    t->op = GGML_OP_NONE;
    for (int i = 0; i < GGML_MAX_SRC; i++)
    {
        t->src[i] = NULL;
    }
}

void ggml_graph_compute_helper(std::vector<uint8_t> &buf, ggml_cgraph *graph, int n_threads)
{
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0)
    {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

// load image from a file(uses stbi_load)
bool load_image_from_file(const std::string &fname, image_u8 &img)
{
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data)
    {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

// preprocess input image : bilinear resize + normalize
bool vit_image_preprocess_bilinear(const image_u8 &img, image_f32 &res, const vit_hparams &params)
{
    const int nx = img.nx;
    const int ny = img.ny;

    const int target_size = params.n_img_size();

    res.nx = target_size;
    res.ny = target_size;
    res.data.resize(3 * target_size * target_size);

    const float x_scale = nx / (float)target_size;
    const float y_scale = ny / (float)target_size;

    fprintf(stderr, "%s: x_scale = %f, y_scale = %f\n", __func__, x_scale, y_scale);

    const int nx3 = int(nx / x_scale + 0.5f);
    const int ny3 = int(ny / y_scale + 0.5f);

    const float m3[3] = {123.675f, 116.280f, 103.530f};
    const float s3[3] = {58.395f, 57.120f, 57.375f};

    // #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < ny3; y++)
    {
        for (int x = 0; x < nx3; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                // linear interpolation
                const float sx = (x + 0.5f) * x_scale - 0.5f;
                const float sy = (y + 0.5f) * y_scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = img.data[j00];
                const float v01 = img.data[j01];
                const float v10 = img.data[j10];
                const float v11 = img.data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res.data[i] = (float(v2) - m3[c]) / s3[c];
            }
        }
    }
    return true;
}

float clip(float x, float lower, float upper)
{
    return std::max(lower, std::min(x, upper));
}

// preprocess input image : bicubic resize + normalize
bool vit_image_preprocess_bicubic(const image_u8 &img, image_f32 &res, const vit_hparams &params)
{
    const int nx = img.nx;
    const int ny = img.ny;

    const int newWidth = params.n_img_size();
    const int newHeight = params.n_img_size();
    res.nx = newWidth;
    res.ny = newHeight;
    res.data.resize(3 * newWidth * newHeight);

    int a, b, c, d, index;
    float Ca, Cb, Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, ii, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)newWidth;
    ty = (float)ny / (float)newHeight;
    printf("newWidth, newHeight = %d, %d\n", newWidth, newHeight);
    printf("tx, ty = %f, %f\n", tx, ty);
    printf("nx, ny = %d, %d\n", nx, ny);

    float scale = std::max(tx, ty);
    fprintf(stderr, "%s: scale = %f\n", __func__, scale);

    const float m3[3] = {123.675f, 116.280f, 103.530f};
    const float s3[3] = {58.395f, 57.120f, 57.375f};

    // Bicubic interpolation; inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    // #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < newHeight; i++)
    {
        for (j = 0; j < newWidth; j++)
        {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            index = (y * nx + x) * 3;
            a = (y * nx + (x + 1)) * 3;
            b = ((y + 1) * nx + x) * 3;
            c = ((y + 1) * nx + (x + 1)) * 3;

            for (k = 0; k < 3; k++)
            {
                for (jj = 0; jj <= 3; jj++)
                {
                    d0 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    res.data[(i * newWidth + j) * 3 + k] = (float(Cc2) - m3[k]) / s3[k];
                }
            }
        }
    }

    return true;
}


/* bool vit_image_preprocess(const image_u8 &img, image_f32 &res, const vit_hparams &params)
{
    const std::string mode = params.interpolation.c_str();
    if (mode == "bilinear")
    {
        return vit_image_preprocess_bilinear(img, res, params);
    }
    else if (mode == "bicubic")
    {
        return vit_image_preprocess_bicubic(img, res, params);
    }
    else
    {
        std::cout << "Interpolation mode '" << mode << "' is not supported; returning 'false'...";
        return false;
    }
} */


// Skipping the diff options of pre processing in the above implementation and sticking to standard processing below 

// preprocess the image
bool vit_image_preprocess(const image_u8 &img, image_f32 &res, const vit_hparams &params)
{
    // convert to grayscale
    std::vector<uint8_t> grayscale(img.nx * img.ny);
    for (int i = 0; i < img.nx * img.ny; ++i)
    {
        grayscale[i] = rgb_to_grayscale(img.data[3 * i], img.data[3 * i + 1], img.data[3 * i + 2]);
    }

    // resize using linear interpolation
    const int target_size = params.n_img_size();
    res.nx = target_size;
    res.ny = target_size;
    res.data.resize(target_size * target_size, 0);

    const float x_scale = static_cast<float>(img.nx) / target_size;
    const float y_scale = static_cast<float>(img.ny) / target_size;

    for (int y = 0; y < target_size; ++y)
    {
        for (int x = 0; x < target_size; ++x)
        {
            float gx = x * x_scale;
            float gy = y * y_scale;
            int gxi = static_cast<int>(gx);
            int gyi = static_cast<int>(gy);
            float u = gx - gxi;
            float v = gy - gyi;
            int px0 = std::max(0, std::min(gxi, img.nx - 2));
            int py0 = std::max(0, std::min(gyi, img.ny - 2));
            int px1 = px0 + 1;
            int py1 = py0 + 1;

            // linear interpolation
            float val = (1 - u) * (1 - v) * grayscale[py0 * img.nx + px0] +
                        u * (1 - v) * grayscale[py0 * img.nx + px1] +
                        (1 - u) * v * grayscale[py1 * img.nx + px0] +
                        u * v * grayscale[py1 * img.nx + px1];

            // normalize pixels to [-1, 1]
            val = (val / 255.0f - 0.5f) * 2.0f;

            res.data[(y * target_size + x)] = val;
        }
    }

    // Padding
    // width (right padding)
    for (int y = 0; y < target_size; ++y)
    {
        for (int x = res.nx; x < target_size; ++x)
        {
            res.data[(y * target_size + x)] = -1.0f;
        }
    }

    // height (bottom padding)
    for (int y = res.ny; y < target_size; ++y)
    {
        for (int x = 0; x < target_size; ++x)
        {
            res.data[(y * target_size + x)] = -1.0f;
        }
    }

    return true;
}

//
// ViT Encoder
//

struct ggml_cgraph *vit_encode_image(
    const vit_model &model,
    vit_state &state,
    const image_f32 &img)
{

    const auto &hparams = model.hparams;
    const auto &enc = model.enc_img;
    const auto &classifier = model.classifier;

    const int32_t hidden_size = hparams.hidden_size;
    const int32_t num_hidden_layers = hparams.num_hidden_layers;
    const int32_t num_attention_heads = hparams.num_attention_heads;
    const int32_t num_classes = hparams.num_classes;
    const int32_t n_img_size = hparams.img_size;
    const int32_t n_enc_head_dim = hparams.n_enc_head_dim();

    const int32_t n_img_embd = hparams.n_img_embd();
    const int32_t n_patch_size = hparams.n_patch_size();

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/state.buf_compute_img_enc.size(),
        /*.mem_buffer =*/state.buf_compute_img_enc.data(),
        /*.no_alloc   =*/true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
    ggml_allocr_alloc(state.allocr, inp);
    if (!ggml_allocr_is_measure(state.allocr))
    {
        float *data = (float *)ggml_get_data(inp);

        const int nx = img.nx;
        const int ny = img.ny;
        const int n = nx * ny;

        GGML_ASSERT(nx == n_img_size && ny == n_img_size);

        for (int k = 0; k < 3; k++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[k * n + y * nx + x] = img.data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    // patch embedding
    struct ggml_tensor *cur = ggml_conv_2d_sk_p0(ctx0, enc.proj_w, inp);
    cur = ggml_add_inplace(ctx0,
                           cur,
                           ggml_repeat(ctx0, enc.proj_b, cur));

    // keep in F32
    cur = ggml_cont(ctx0,
                    ggml_permute(ctx0, cur, 1, 2, 0, 3));

    // convert to F16
    // cur = ggml_cpy(ctx0,
    //                ggml_permute(ctx0, cur, 1, 2, 0, 3),
    //                ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hidden_size, n_img_embd, n_img_embd));

    // add positional embedding
    // cur dim     : 768  28  28  1
    // enc.pe dim  : 768  785  1  1

    // reshape patch embeddings from (768  28  28  1) to (768  784  1  1)
    cur = ggml_reshape_4d(ctx0, cur, hidden_size, n_img_embd * n_img_embd, 1, 1);

    // concat class embeddings(cls_token) : (768  1  1  1) with positional embeddings (pos_embed = cur) : (768  784  1  1)
    cur = ggml_permute(ctx0, ggml_concat(ctx0, enc.cls_token, ggml_permute(ctx0, cur, 0, 2, 1, 3)),
                       0, 2, 1, 3); // 768  785  1  1

    cur = ggml_add_inplace(ctx0, cur, enc.pe);

    struct ggml_tensor *inpL = cur;

    // loop over layers
    for (int il = 0; il < num_hidden_layers; ++il)
    {
        const auto &layer = enc.layers[il];

        // norm 1
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = w * cur + b
            cur = ggml_mul(ctx0, cur, layer.norm1_w);
            cur = ggml_add_inplace(ctx0, cur, layer.norm1_b);
        }

        const int64_t W = cur->ne[1];
        const int64_t H = cur->ne[2];

        // self-attention
        {
            cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.qkv_b);

            // split qkv into separate tensors
            const int B = cur->ne[3];

            cur = ggml_reshape_4d(ctx0, cur, hidden_size, 3, W * H, B);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 3, 1, 2));

            struct ggml_tensor *Q;
            struct ggml_tensor *K;
            struct ggml_tensor *V;

            Q = ggml_view_3d(ctx0, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 0 * cur->nb[3]);
            Q = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, num_attention_heads, W * H, B);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, n_enc_head_dim, W * H, B * num_attention_heads);

            K = ggml_view_3d(ctx0, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
            K = ggml_reshape_4d(ctx0, K, n_enc_head_dim, num_attention_heads, W * H, B);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, n_enc_head_dim, W * H, B * num_attention_heads);

            V = ggml_view_3d(ctx0, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
            V = ggml_reshape_4d(ctx0, V, n_enc_head_dim, num_attention_heads, W * H, B);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
            V = ggml_reshape_3d(ctx0, V, W * H, n_enc_head_dim, B * num_attention_heads);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            // attention weights
            struct ggml_tensor *KQ_scaled =
                ggml_scale_inplace(ctx0,
                                   KQ,
                                   ggml_new_f32(ctx0, 1.0f / sqrtf(n_enc_head_dim)));

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            cur =
                ggml_reshape_4d(ctx0,
                                ggml_cont(ctx0,
                                          ggml_permute(ctx0,
                                                       ggml_reshape_4d(ctx0, KQV, n_enc_head_dim, W * H, num_attention_heads, B),
                                                       0, 2, 1, 3)),
                                hidden_size, W, H, B);

            cur = ggml_mul_mat(ctx0, layer.proj_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.proj_b);
        }

        // add skip connection
        cur = ggml_add_inplace(ctx0, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm 2
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = w * cur + b
                cur = ggml_mul(ctx0, cur, layer.norm2_w);
                cur = ggml_add_inplace(ctx0, cur, layer.norm2_b);
            }

            // fully connected layer
            cur = ggml_mul_mat(ctx0, layer.mlp_lin1_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin1_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0, layer.mlp_lin2_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin2_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    //
    // pooling
    //

    // get the output of cls token at index 0
    struct ggml_tensor *cls = ggml_new_i32(ctx0, 0);
    cur = ggml_get_rows(ctx0, cur, cls);

    // layer normalization
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = w * cur + b
        cur = ggml_mul(ctx0, cur, classifier.norm_w);
        cur = ggml_add_inplace(ctx0, cur, classifier.norm_b);
    }

    //
    // classification head
    //

    // projection
    cur = ggml_mul_mat(ctx0, classifier.head_w, cur);
    cur = ggml_add_inplace(ctx0, cur, classifier.head_b);

    // softmax
    ggml_tensor *probs = ggml_soft_max(ctx0, cur);

    probs = ggml_cpy(ctx0, probs, state.prediction);

    ggml_build_forward_expand(gf, probs);
    ggml_disconnect_node_from_graph(state.prediction);

    ggml_free(ctx0);

    return gf;
}

void print_usage(int argc, char **argv, const vit_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help              show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model       model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp         input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -t N, --threads         number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -k N, --topk            top k classes to print (default: %d)\n", params.topk);
    fprintf(stderr, "  -s SEED, --seed         RNG seed (default: -1)\n");
    fprintf(stderr, "  -e FLOAT, --epsilon     epsilon constant in Layer Norm layers (default: %f)\n", params.eps);
    fprintf(stderr, "\n");
}

bool vit_params_parse(int argc, char **argv, vit_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed")
        {
            params.seed = std::stoi(argv[++i]);
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-i" || arg == "--inp")
        {
            params.fname_inp = argv[++i];
        }
        else if (arg == "-k" || arg == "--topk")
        {
            params.topk = std::stoi(argv[++i]);
        }
        else if (arg == "-e" || arg == "--epsilon")
        {
            params.eps = std::stof(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argc, argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

int vit_predict(const vit_model &model, vit_state &state, const image_f32 img1, const vit_params &params, std::vector<std::pair<float, int>> &predictions)
{
    static const size_t tensor_alignment = 32;

    // first build the graph and record memory requirements
    state.buf_compute_img_enc.resize(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
    state.allocr = ggml_allocr_new_measure(tensor_alignment);
    struct ggml_cgraph *gf_measure = vit_encode_image(model, state, img1);
    if (!gf_measure)
    {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return 1;
    }

    size_t alloc_size = ggml_allocr_alloc_graph(state.allocr, gf_measure) + tensor_alignment;
    ggml_allocr_free(state.allocr);

    // recreate allocator with exact memory requirements
    state.buf_alloc_img_enc.resize(alloc_size);
    state.allocr = ggml_allocr_new(state.buf_alloc_img_enc.data(), state.buf_alloc_img_enc.size(), tensor_alignment);

    // compute the graph with the measured exact memory requirements from above
    ggml_allocr_reset(state.allocr);

    struct ggml_cgraph *gf = vit_encode_image(model, state, img1);
    if (!gf)
    {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return 1;
    }

    ggml_allocr_alloc_graph(state.allocr, gf);
    ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

    // print_t_f32("after probs", state.prediction);

    const float *probs_data = ggml_get_data_f32(state.prediction);

    // clear previous predictions, used for topk
    predictions.clear();
    // std::vector<std::pair<float, int>> predictions;

    // store probability and index
    for (int i = 0; i < model.hparams.num_classes; ++i)
    {
        predictions.push_back(std::make_pair(probs_data[i], i));
    }

    // sort in descending order
    std::sort(predictions.begin(), predictions.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b)
              {
                  return a.first > b.first;
              });

    fprintf(stderr, "\n");

    // top k predictions
    for (int i = 0; i < params.topk && i < predictions.size(); ++i)
    {
        printf(" > %s : %.2f\n",
               model.hparams.id2label.at(predictions[i].second).c_str(),
               predictions[i].first);
    }

    // free memory
    ggml_allocr_free(state.allocr);
    state.allocr = NULL;
    state.work_buffer.clear();

    return 0;
}