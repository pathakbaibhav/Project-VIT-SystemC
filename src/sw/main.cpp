#define _CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

//#include "vitstr.h"
#include "stb_image.h" // stb image load
#include "ImagePatchEmbedding.h"
#include "vit_utils.h"

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
#include <chrono> // cpp lib for calculating time 


#if defined(_MSC_VER) // related to stb_image.h - 
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// main function
int main(int argc, char **argv)
{
    
    // Capture start time
    const auto t_main_start = std::chrono::high_resolution_clock::now();

    vit_params params;

    image_u8 img0;
    image_f32 img1;

    //vit_model model;

    vit_state state;
    std::vector<std::pair<float, int>> predictions;

    int64_t t_load_us = 0;

    if (vit_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    if (params.seed < 0)
    {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);
    fprintf(stderr, "%s: n_threads = %d / %d\n", __func__, params.n_threads, (int32_t)std::thread::hardware_concurrency());

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!vit_model_load(params.model.c_str(), model))
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // load the image
    if (!load_image_from_file(params.fname_inp.c_str(), img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // preprocess the image to f32
    if (vit_image_preprocess(img0, img1, model.hparams))
    {
        fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    }

    // prepare for graph computation, memory allocation and results processing
    {
        static size_t buf_size = 3u * 1024 * 1024;

        struct ggml_init_params ggml_params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,
        };

        state.ctx = ggml_init(ggml_params);
        state.prediction = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, model.hparams.num_classes, 25);
    }

    {
        // run prediction on img1
        //vit_predict(model, state, img1, params, predictions);
    }

    // Measure elapsed time
    const auto t_main_end = std::chrono::high_resolution_clock::now();
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_main_end - t_main_start).count();

    std::cout << "Elapsed time: " << elapsed_us << " microseconds" << std::endl;


    // Implement an EQU for freeing memory like ggml_free(model.ctx)

    return 0;
}