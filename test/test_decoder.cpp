#include "Channel.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <unistd.h>

#include<filesystem>
using namespace std::filesystem;


void print_help() {
    printf("Usage: test_decode -i input [-o output] [-r [start]:[end]] -v \n");
    printf("-i input file, .h264 or .aqms\n");
    printf("-o output. video should be suffixed with mp4. folder means extracting frame images\n");
    printf("-v view the frames. If -o is missing or invalid, -v will be default option\n");
    printf("-r output frame range, from [start] to [end], they have default value of beginning and ending of video\n");
}


int main(int argc, char *argv[])
{
    bool b_visual = false;
    bool b_save_video = false;
    bool b_out = false;
    std::string input, output, range;
    int start = 0, end = -1;
    char o;
    while ((o = getopt(argc, argv, "hvi:o:r:")) != -1) {
        switch (o)
        {
        case 'h':
            print_help();
            return 0;
        case 'v':
            b_visual = true;
            break;
        case 'i':
            input = optarg;
            break;
        case 'o':
            output = optarg;
            break;
        case 'r':
            range = optarg;
            break;
        
        default:
            break;
        }
    }

    path in(input);
    std::string suffix = in.extension();
    if (!exists(in) || !(suffix == ".h264" || suffix == ".aqms")) {
        printf("Invalid input file: %s\n", input.c_str());
        return -1;
    }

    path out(output);
    if (out.extension() == ".mp4") {
        b_save_video = true;
        b_out = true;
    } else if (out.extension() != "") {
        printf("Invalid file extention: %s. No output.\n", out.extension().c_str());
        b_out = false;
        b_visual = true;
    } else if (exists(out)) {
        b_save_video = false;
        b_out = true;
    } else {
        printf("Invalid output path: %s, No output.\n", out.c_str());
        b_out = false;
        b_visual = true;
    }
    if (!range.empty()) {
        auto pos_col = range.find_first_of(':');
        if (pos_col == 0) {
            start = 0;
        }else {
            start = std::stoi(range.substr(0, pos_col));
        }
        if (pos_col == std::string::npos) {
            end = -1;
        } else {
            end = std::stoi(range.substr(pos_col + 1));
        }
    }
    if (!CCUDACtxMan::Init())
    {
        return -1;
    }

    std::shared_ptr<PtrQueue<Frames>> q(new PtrQueue<Frames>);
    q->setMaxSize(100);

    q->startQueue();

    Channel c0("0000", q);


    c0.Open(input, start);
    u_char* d_img;
    cv::Mat h_img(2160, 3840, CV_8UC3);
    cudaMalloc(&d_img, 3840* 2160 * 3);
    cudaStream_t s;
    cudaStreamCreate(&s);
    int i = start;
    cv::VideoWriter writer;
    if (b_out && b_save_video) {
        writer.open(output, cv::VideoWriter::fourcc('X', '2', '6', '4'), 30, {3840, 2160}, true);
    }
    while (i++ != end) {
        std::shared_ptr<Frames> f;
        f = q->pop();
        cudaMemcpyAsync(h_img.data, f->data, 3840* 2160 *3, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        if (b_out) {
            if (b_save_video) {
                writer.write(h_img);
            } else {
                auto filename = out;
                std::stringstream ss;
	            ss << setw(6) << setfill('0') << i ;
                filename /= ss.str() + ".png";
                cv::imwrite(filename, h_img);

            }
        }
        if (b_visual) {
            time_t ts = f->ts / 1000;
            std::string date{ctime(&ts)};
            cv::putText(h_img, date, {10, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, {150, 150, 150}, 2);
            cv::imshow("aa", h_img);
            if (cv::waitKey(1) == 27) break;
        }
        if ((i - start)%5 ==0) {
            std::string back("\b\b\b\b\b\b");
            std::cout << back;
            std::cout << "=" << setw(6) << i << std::flush;
        }
    
    }
    writer.release();
    printf("\n%s saved successfully\n", output.c_str());
    q->stopQueue();
    return 0;

}