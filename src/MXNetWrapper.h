#include <mxnet/c_predict_api.h>
#include <string>
#include <vector>

#ifndef MXNETWRAPPER_H
#define MXNETWRAPPER_H

namespace net_classes {
//-------------------------------------BufferFile class---------------------------------------
    class BufferFile {
    public:
        std::string file_path_;
        int length_;
        char *buffer_;

        explicit BufferFile(std::string file_path);

        int GetLength();

        char *GetBuffer();

        ~BufferFile();
    };
//-------------------------------------------------------------------------------------------

//-------------------------------------Network class-----------------------------------------
    class MXNetWrapper {
        // Parameters
        int dev_type;  // 1: cpu, 2: gpu
        int dev_id;  // arbitrary.
        mx_uint num_input_nodes;  // 1 for feedforward
        const char** input_keys;

        // Files of params and structure
        std::string json_file;
        std::string param_file;

        // Handle
        PredictorHandle net;

    public:
        MXNetWrapper(std::string net_name_str, const char* input_key[], const mx_uint input_shape_indptr[], const mx_uint input_shape_data[]);

        void fordward(std::vector<mx_float> input);

        void free();
    };
//-------------------------------------------------------------------------------------------
}


#endif //MXNETWRAPPER_H
