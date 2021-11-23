// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/script.h>

// C++ STD
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <thread>

using namespace std;

size_t ITER_NUM = 1000;
const size_t WARM_UPS = 10;

typedef  unordered_map<string, c10::IValue> KWARG;

void move_to_cuda_if_available(KWARG &kwargs) {

    if (torch::cuda::is_available()) {
         torch::Device device = torch::kCUDA;

         for(auto &p : kwargs)
            p.second = p.second.toTensor().to(device);
    }
}

class Worker {
    public:
    Worker(size_t id, size_t& queue, mutex& mtx)
    :id_(id), queue_(queue), mtx_(mtx) {
        kwargs_["input_ids"] = torch::tensor(std::vector<int64_t>{
            101,  1109,  1419, 20164, 10932,  2271,  7954,  1110,  1359,  1107,
            1203,  1365,  1392,   102,  7302,  1116,  1132,  2108,  2213,  1111,
            1240,  2332,   102}).unsqueeze(0);
        kwargs_["token_type_ids"] = torch::tensor(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0);
        kwargs_["attention_mask"] = torch::tensor(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0);

        move_to_cuda_if_available(kwargs_);

    }

    void do_work() {
        {
            lock_guard<mutex> lock(mtx_);
            cout << "Thread " << id_ << endl;
        }
        
        while(true){
            {
                lock_guard<mutex> lock(mtx_);
                if(queue_ == 0)
                    return;
                if(queue_ % (ITER_NUM/10) == 0)
                    cout << "Queue length: " << queue_ << endl;
                queue_--;
            }
            process_payload();
        }
    }

    protected:

    virtual void process_payload() = 0;

    size_t& queue_;
    mutex& mtx_;
    size_t id_;
    std::unordered_map<std::string, c10::IValue> kwargs_;
};

class TorchScriptWorker: public Worker {
    public:
    TorchScriptWorker(size_t id, size_t& queue, mutex& mtx, torch::jit::Module &model)
    :model_(model), Worker(id, queue, mtx) {
        // std::unordered_map<std::string, c10::IValue> local_copy;

        local_copy["input_ids"] = torch::tensor(std::vector<int64_t>{
            101,  1109,  1419, 20164, 10932,  2271,  7954,  1110,  1359,  1107,
            1203,  1365,  1392,   102,  7302,  1116,  1132,  2108,  2213,  1111,
            1240,  2332,   102}).unsqueeze(0);
        local_copy["token_type_ids"] = torch::tensor(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0);
        local_copy["attention_mask"] = torch::tensor(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0);
        move_to_cuda_if_available(local_copy);

    }
    protected:

    void process_payload() {
        // auto ret = model_.forward({}, kwargs_).toIValue().toTuple()->elements()[0];
        auto ret = model_.forward({}, local_copy).toIValue().toTuple()->elements()[0];
    }

    torch::jit::Module &model_;
    KWARG local_copy;
};


int main(const int argc, const char* const argv[]) {
    if (argc != 3) {
        std::cout << "Usage: benchmark [QUEUE_LENGTH] [THREADS]" << std::endl
                << "Example: benchmark 1000 4" << std::endl;
        return EXIT_FAILURE;
    }

    ITER_NUM = std::stoul(argv[1]);
    const size_t THREAD_NUM = std::stoul(argv[2]);

    at::set_num_interop_threads(1);
    at::set_num_threads(1); 

    cout << "Inter-op threads:" << at::get_num_interop_threads() << endl;
    cout << "Intra-op threads:" << at::get_num_threads() << endl; 


    std::unordered_map<std::string, c10::IValue> kwargs;

    kwargs["input_ids"] = torch::tensor(std::vector<int64_t>{
        101,  1109,  1419, 20164, 10932,  2271,  7954,  1110,  1359,  1107,
        1203,  1365,  1392,   102,  7302,  1116,  1132,  2108,  2213,  1111,
        1240,  2332,   102}).unsqueeze(0);
    kwargs["token_type_ids"] = torch::tensor(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0);
    kwargs["attention_mask"] = torch::tensor(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0);

    move_to_cuda_if_available(kwargs);

    // Torch script
    torch::jit::Module traced  = torch::jit::load("../models/bert_model_only_traced.pt");

    traced.eval();

    chrono::steady_clock::time_point begin;

    mutex mtx;
    size_t queue = ITER_NUM;

    // vector<Worker> worker;
    vector<thread> worker_threads;

    auto ret = traced.forward({}, kwargs).toIValue().toTuple()->elements()[0];
    float paraphrased_percent = 100.0 * torch::softmax(ret.toTensor(),1)[0][1].item<float>();
    cout << round(paraphrased_percent) << "% paraphrase" << endl;

    // Warm-up
    for(size_t i=0; i<WARM_UPS; ++i){
        auto ret = traced.forward({}, kwargs).toIValue().toTuple()->elements()[0];
    }

    begin = chrono::steady_clock::now();
    for(int i=0; i<THREAD_NUM; ++i)
        worker_threads.emplace_back(&Worker::do_work, TorchScriptWorker(i, queue, mtx, traced));
    
    for(auto &t : worker_threads)
        t.join();

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Mean ModelTime (ms): " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() / float(ITER_NUM) << endl;
    cout << "Throughput: " << float(ITER_NUM) / (chrono::duration_cast<chrono::milliseconds>(end - begin).count()/1000.f)  << endl;
}
