#include <stdio.h>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"
#include "ncnn/datareader.h"

#include "mnist-1-bin.h"
#include "fs.h"

extern "C" void app_main(void)
{
    ncnn::Net net;

    printf("Loading ncnn mnist model...");
    net.load_param(mnist_1_opt_param_bin);
    net.load_model(mnist_1_opt_bin);
    net.opt.lightmode = true;
    printf("Done.\n");

    printf("Preparing input...");

    ncnn::Mat in = ncnn::Mat::from_pixels(IN_IMG, ncnn::Mat::PIXEL_GRAY, IMAGE_W, IMAGE_H);
    ncnn::Mat out;

    printf("[28, 28, 1].\n");

    printf("Start Mesuring!\n");

    double total_latency = 0;
    float max = -1;
    float min = 10000000000000000;

    for(int i=0; i<10; i++){
        vTaskDelay(20 / portTICK_PERIOD_MS);

        long start = esp_timer_get_time();

        ncnn::Extractor ex = net.create_extractor();
        ex.input(0, in);
        ex.extract(17, out);
//         {
//             int j = 17;
// //         for(int j=0; j<=17; j++){
//             ex.extract(j, out);
//             const float* ptr = out.channel(0);
//             printf("%d: ", j);
//             for(int k=0; k<out.w * out.h; k++){
//                 printf("%.2f, ", ptr[k]);
//             }
//             printf("\n");
//         }

        long end = esp_timer_get_time();

        float lat = (end - start)/1000.0;
        total_latency += lat;
        if(lat > max){
            max = lat;
        }
        if(lat < min){
            min = lat;
        }
    }

    printf("Done!\n");

    const float* ptr = out.channel(0);
    int gussed = -1;
    float guss_exp = -10000000;
    for(int i=0; i<out.w * out.h; i++){
        printf("%d: %.2f\n", i, ptr[i]);
        if(guss_exp < ptr[i]){
            gussed = i;
            guss_exp = ptr[i];
        }
    }

    printf("I think it is number %d!\n", gussed);

    printf("Latency, avg: %.2fms, max: %.2f, min: %.2f. Avg Flops: %.2fMFlops\n",total_latency / 10.0, max, min, 0.78 / (total_latency / 10.0 / 1000.0));
}
