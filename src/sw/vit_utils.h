#pragma once

#include "stb_image.h"

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

struct vit_hparams
{
    int32_t hidden_size = 768;
    int32_t num_hidden_layers = 12;
    int32_t num_attention_heads = 12;
    int32_t num_classes = 1000;
    int32_t patch_size = 8;
    int32_t img_size = 224;
    int32_t ftype = 1;
    float eps = 1e-6f;
    std::string interpolation = "bicubic";
    std::map<int, std::string> id2label;

    int32_t n_enc_head_dim() const;
    int32_t n_img_size() const;
    int32_t n_patch_size() const;
    int32_t n_img_embd() const;
};
struct vit_model
{
    vit_hparams hparams; // already non ggml 
    // non ggml image encoder equ 
    // non ggml MLP 

};

struct image_u8
{
    int nx;
    int ny;
    std::vector<uint8_t> data;
};

struct image_f32
{
    int nx;
    int ny;
    std::vector<float> data;
};

struct vit_params
{
    int32_t seed = -1;
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t topk = 5;
    std::string model = "../ggml-model-f16.gguf";  // model path - load specifici arrays from weights.cpp 
    std::string fname_inp = "../images/demo_1.png"; // image path  - This /images directory or /assets in vit cpp ? 
    float eps = 1e-6f;                             // epsilon used in LN
};

//void print_t_f32(const char *title, struct ggml_tensor *t, int n);
bool load_image_from_file(const std::string &fname, image_u8 &img);
bool vit_image_preprocess(const image_u8 &img, image_f32 &res, const vit_hparams &params);
//bool vit_model_load(const std::string &fname, vit_model &model);
void print_usage(int argc, char **argv, const vit_params &params);
