
#include "MXNetWrapper.h"


int main(int argc, char* argv[]){
//------------------------------------------------------------------------------
//-------------------------Example with Inception Network-----------------------
//------------------------------------------------------------------------------
// int width = 224;
// int height = 224;
// int channels = 3;
// int dim = width*height*channels;
// const mx_uint input_shape_indptr[] = {
//                                  0, // The index in the vector input_shape_data that represents the first dimension of the first input
//                                  4  // Dimensions of the vector input_shape_data
//                                };
// const mx_uint input_shape_data[] = {
//                                       1,                              // Number of Examples
//                                       static_cast<mx_uint>(channels), // First dim of the first input
//                                       static_cast<mx_uint>(height),   // Second dim of the first input
//                                       static_cast<mx_uint>(width)     // Third dim of the first input
//                                    };
// mx_uint num_input_nodes_in = 1;      // Number of inputs
// const char* input_key[1] = {"data"}; // Vector of strings of the input labels
//
// std::vector<mx_float> vett_float = std::vector<mx_float>(dim);
// for (int i = 0 ; i < dim ; i++){
//     vett_float[i] = 0.0;
// }
//
// net_classes::MXNetWrapper net = net_classes::MXNetWrapper((std::string)"networks/Inception-BN", num_input_nodes_in, input_key, input_shape_indptr, input_shape_data);
// net.forward(vett_float);

//------------------------------------------------------------------------------
//---------------------- First net 1 Input 1 Output ----------------------------
//------------------------------------------------------------------------------

    //----------------------------------------------------------
    mx_uint num_input_nodes_in1 = 1;
    const char* input_key1[1] = {"myIn"};
    const mx_uint input_shape_indptr1[] = { 0, 2 };
    const mx_uint input_shape_data1[] = { 1, 5 };
    std::vector<std::vector<mx_float>> vett_In = {{1.0,2.0,3.0,4.0,5.0}};
    //----------------------------------------------------------

    //!!!!!!!!!!!!!!Error
    //net_classes::MXNetWrapper net1_112_standard = net_classes::MXNetWrapper((std::string)"networks/net1.Standard11.2Export", num_input_nodes_in, input_key, input_shape_indptr, input_shape_data);
    //net1_112_standard.forward(vett_in);
    //!!!!!!!!!!!!!!Error

    net_classes::MXNetWrapper net1_112_fixed = net_classes::MXNetWrapper((std::string)"networks/net1.Fixed11.2Export", num_input_nodes_in1, input_key1, input_shape_indptr1, input_shape_data1);
    net1_112_fixed.forward(vett_In);

    //!!!!!!!!!!!!!!Error
    //net_classes::MXNetWrapper net1_113_standard = net_classes::MXNetWrapper((std::string)"networks/net1.Standard11.3Export", num_input_nodes_in, input_key, input_shape_indptr, input_shape_data);
    //net1_113_standard.forward(vett_in);
    //!!!!!!!!!!!!!!Error

    const char* input_key11[1] = {".Inputs.myIn"};

    net_classes::MXNetWrapper net1_113_fixed = net_classes::MXNetWrapper((std::string)"networks/net1.Fixed11.3Export", num_input_nodes_in1, input_key11, input_shape_indptr1, input_shape_data1);
    net1_113_fixed.forward(vett_In);

//------------------------------------------------------------------------------
//---------------------- First net 2 Input 2 Output ----------------------------
//------------------------------------------------------------------------------


    //----------------------------------------------------------
    mx_uint num_input_nodes_in2 = 2;
    const char* input_key2[] = {"myIn1","myIn2"};
    const mx_uint input_shape_indptr2[] = {
        0, //Index of the first dimension numbers of the first input in the vector input_shape_data2
        2, //Index of the first dimension numbers of the second input in the vector input_shape_data2
        4  //Size of the vector input_shape_data2
    };
    const mx_uint input_shape_data2[] = {
        1, // Number of Examples of first input
        4, // First dimensions of the first input
        1, // Number of Examples of second input
        4 // First dimensions of the second input
    };
    std::vector<mx_float> vett_In1 = {1.0,2.0,3.0,4.0};
    std::vector<mx_float> vett_In2 = {1.0,2.0,3.0,4.0};
    std::vector<std::vector<mx_float>> vett_In12= {vett_In1,vett_In2};
    //----------------------------------------------------------

    //!!!!!!!!!!!!!!Error
    //net_classes::MXNetWrapper net2_112_standard = net_classes::MXNetWrapper((std::string)"networks/net2.Standard11.2Export", num_input_nodes_in2, input_key2, input_shape_indptr2, input_shape_data2);
    //net2_112_standard.forward(vett_In12,0);
    //net2_112_standard.forward(vett_In12,1);
    //!!!!!!!!!!!!!!Error

    net_classes::MXNetWrapper net2_112_fixed = net_classes::MXNetWrapper((std::string)"networks/net2.Fixed11.2Export", num_input_nodes_in2, input_key2, input_shape_indptr2, input_shape_data2);
    net2_112_fixed.forward(vett_In12,0);
    net2_112_fixed.forward(vett_In12,1);

    //!!!!!!!!!!!!!!Error
    //net_classes::MXNetWrapper net2_113_standard = net_classes::MXNetWrapper((std::string)"networks/net2.Standard11.3Export", num_input_nodes_in2, input_key2, input_shape_indptr2, input_shape_data2);
    //net2_113_standard.forward(vett_In12,0);
    //net2_113_standard.forward(vett_In12,1);
    //!!!!!!!!!!!!!!Error

    const char* input_key21[] = {".Inputs.myIn1",".Inputs.myIn2"};
    net_classes::MXNetWrapper net2_113_fixed = net_classes::MXNetWrapper((std::string)"networks/net2.Fixed11.3Export", num_input_nodes_in2, input_key21, input_shape_indptr2, input_shape_data2);
    net2_113_fixed.forward(vett_In12,0);
    net2_113_fixed.forward(vett_In12,1);

}
