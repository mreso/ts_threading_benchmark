# Repro steps

0 - (optional) Create conda environment

    $ conda create --name ts_threading_benchmark
    $ conda activate ts_threading_benchmark

1 - Checkout source 
    
    $ git clone git@github.com:mreso/ts_threading_benchmark.git

2 - Install dependencies

    $ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    $ pip install transformers==4.12.0


3 - Export model

    $ mkdir models
    $ python export_model.py

4 - Compile and run benchmark
    
    $ mkdir build
    $ cd build

    $ cmake -DCMAKE_PREFIX_PATH=/home/ubuntu/anaconda3/envs/ts_threading_benchmark/lib/python3.9/site-packages/torch/ ..

    $ make

    $ OMP_NUM_THREADS=1 ./benchmark 10000 1
    $ OMP_NUM_THREADS=1 ./benchmark 10000 2





