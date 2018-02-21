#include <mxnet/c_predict_api.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "MXNetWrapper.h"

using namespace std;

#ifdef DEBUG
#define D(x) {x}
#else
#define D(x)
#endif

namespace net_classes {
//-------------------------------------BufferFile class---------------------------------------
BufferFile::BufferFile(std::string file_path) : file_path_(file_path)
{
    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        D(std::cerr << "Can't open the file. Please check " << file_path << ". \n";)
        length_ = 0;
        buffer_ = NULL;
        return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    D(std::cout << "Loading " << file_path.c_str() << "(" << length_ << " bytes) ... ";)

    buffer_ = new char[sizeof(char) * length_];
    ifs.read(buffer_, length_);
    ifs.close();
    D(std::cout << "Done\n";)
}

int BufferFile::GetLength()
{
    return length_;
}

char* BufferFile::GetBuffer()
{
    return buffer_;
}

BufferFile::~BufferFile()
{
    if (buffer_) {
        delete[] buffer_;
        buffer_ = NULL;
    }
}
//-------------------------------------------------------------------------------------------

//-------------------------------------Network class-----------------------------------------
MXNetWrapper::MXNetWrapper(
    std::string net_name_str,
    const char* input_key[],
    const mx_uint input_shape_indptr[],
    const mx_uint input_shape_data[]):
    dev_type(1),
    dev_id(0),
    num_input_nodes(1),
    net(0)
{
    //input_keys = input_key;

    json_file = net_name_str+".json";
    param_file = net_name_str+".params";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Create Predictor
    MXPredCreate((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_key,
                 input_shape_indptr,
                 input_shape_data,
                 &net);
    assert(net);
    D(std::cout << "Network Created: " << net_name_str << "\n";)
}

std::vector<mx_float> MXNetWrapper::fordward(std::vector<mx_float> input)
{
    D(
        float* r = input.data();
        cout << "Data(" << input.size() << "): [" << r[0];
        for(int i=1; i<input.size(); i++) {
            cout  << ',' << r[i];
        }
        cout << "]\n";
    )
    // Set Input
//            std::vector<mx_float> inputT = {1.2,2.0,3.1};
    MXPredSetInput(net, "Input", input.data(), input.size());

    // Do Predict Forward
    MXPredForward(net);

    // Get Output
    mx_uint output_index = 0;
    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result Dimensions
    MXPredGetOutputShape(net, output_index, &shape, &shape_len);
    D(
        cout << "Output index: " << output_index << "\n";
        cout << "Shape len of the output: " << shape_len << '\n';
        cout << "Shape dim: [" << shape[0];
        for (int i = 1; i < shape_len; ++i){
            cout << ',' << shape[i];
        }
        cout << "]\n";
    )

    // Create the output data vector
    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
    std::vector<mx_float> out_vett(size);
    MXPredGetOutput(net, output_index, &(out_vett[0]), size);
    D(
            float* r = out_vett.data();
            cout << "Data("<< out_vett.size() <<"): [" << r[0];
            for(int i = 1; i < out_vett.size(); i++) {
                cout  << ',' << r[i];
            }
            cout << "]\n";
    )
    return out_vett;
}

void MXNetWrapper::free()
{
    // Release Predictor
    MXPredFree(net);
}

//-------------------------------------------------------------------------------------------
}
