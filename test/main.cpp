
#include "MXNetWrapper.h"


int main(int argc, char* argv[]){
//------------------------------------------------------------------------------
//-------------------------Example with Inception Network-----------------------
//------------------------------------------------------------------------------
// int width = 224;
// int height = 224;
// int channels = 3;
// int dim = width*height*channels;
// const mx_uint input_shape_indptr[] = { 0, 4 };
// const mx_uint input_shape_data[] = { 1,
//                                       static_cast<mx_uint>(channels),
//                                       static_cast<mx_uint>(height),
//                                       static_cast<mx_uint>(width)};
// const char* input_key[1];
// input_key[0] = {"data"};
//
// std::vector<mx_float> vett_float = std::vector<mx_float>(dim);
// for (int i = 0 ; i < dim ; i++){
//     vett_float[i] = 0.0;
// }
//
// net_classes::MXNetWrapper net = net_classes::MXNetWrapper((std::string)"networks/Inception-BN",  input_key, input_shape_indptr, input_shape_data);
// net.fordward(vett_float);

//------------------------------------------------------------------------------
//-----------------------Example with my Graph Wrong Export---------------------
//------------------------------------------------------------------------------

  //----------------------------------------------------------
//    const mx_uint input_shape_indptr2[] = { 0, 2 };
//    const mx_uint input_shape_data2[] = { 1, 2 };
//    const char* input_key2[1];
//    input_key2[0] = {"Input"};
  //----------------------------------------------------------

    //net_classes::MXNetWrapper net2 = net_classes::MXNetWrapper((std::string)"networks/MathematicaNet_Wrong", input_key2, input_shape_indptr2, input_shape_data2);
//    std::vector<mx_float> vett_float2 = {1.2,1.0};
    //net2.fordward(vett_float2);

    //net_classes::MXNetWrapper net3 = net_classes::MXNetWrapper((std::string)"networks/MathematicaNet_Wrong", input_key2, input_shape_indptr2, input_shape_data2);
    //net3.fordward(vett_float2);

//------------------------------------------------------------------------------
//-----------------------Example with my Graph Right Export---------------------
//------------------------------------------------------------------------------

/*    net_classes::MXNetWrapper net4 = net_classes::MXNetWrapper((std::string)"networks/MathematicaNet", input_key2, input_shape_indptr2, input_shape_data2);
    net4.fordward(vett_float2);

    net_classes::MXNetWrapper net5 = net_classes::MXNetWrapper((std::string)"networks/MathematicaNet", input_key2, input_shape_indptr2, input_shape_data2);
    net5.fordward(vett_float2);
*/
/*
    const mx_uint input_shape_indptr1[] = { 0, 2, 2 };
    const mx_uint input_shape_data1[] = { 1, 5, 1, 5 };
    const char* input_key1[2] = {"throttle","brake"};
    std::vector<mx_float> vett_float1 = {1.0,2.0,3.0,4.0,5.0};

    net_classes::MXNetWrapper net6 = net_classes::MXNetWrapper((std::string)"networks/GAS", input_key1, input_shape_indptr1, input_shape_data1);
    net6.fordward(vett_float1);
*/

/*
    Test 1
*/

    const mx_uint input_shape_indptr1[] = { 0, 2 };
    const mx_uint input_shape_data1[] = { 2 /* Number of Examples */ , 5 /* First dimensions of the input */ };
    const char* input_key1[1] = {"throttle"};
    std::vector<std::vector<mx_float>> vett_float1 = {{1.0,2.0,3.0,4.0,5.0,1.0,2.0,3.0,4.0,6.0}};

    net_classes::MXNetWrapper netTest1 = net_classes::MXNetWrapper((std::string)"networks/netTest1", 1, input_key1, input_shape_indptr1, input_shape_data1);
    netTest1.fordward(vett_float1);

/*
    Test 2
*/

    const mx_uint input_shape_indptr2[] = { 0, 2, 4 };
    const mx_uint input_shape_data2[] = { 1 /* Number of Examples */ , 5 /* First dimensions of the input */, 1, 5};
    const char* input_key2[] = {"throttle","brake"};
    std::vector<std::vector<mx_float>> vett_float2 = {{1.0,2.0,3.0,4.0,5.0},{1.0,2.0,3.0,4.0,5.0}};

    net_classes::MXNetWrapper netTest2 = net_classes::MXNetWrapper((std::string)"networks/netTest2", 2, input_key2, input_shape_indptr2, input_shape_data2);
    netTest2.fordward(vett_float2);

/*
    Test 3
*/

    const mx_uint input_shape_indptr3[] = {
        0, //Index of the first dimension numbers of the first input in the vector input_shape_data2
        2, //Index of the first dimension numbers of the second input in the vector input_shape_data2
        4  //Size of the vector input_shape_data2
    };
    const mx_uint input_shape_data3[] = {
        1, // Number of Examples of first input
        5, // First dimensions of the first input
        1, // Number of Examples of second input
        5 // First dimensions of the second input
    };
    const char* input_key3[] = {"throttle","brake"};
    std::vector<mx_float> vett_throttle = {1.0,2.0,3.0,4.0,5.0};
    std::vector<mx_float> vett_brake = {1.0,2.0,3.0,4.0,5.0};
    std::vector<std::vector<mx_float>> vett_input = {vett_throttle,vett_brake};

    net_classes::MXNetWrapper netTest3 = net_classes::MXNetWrapper((std::string)"networks/netTest3", 2, input_key3, input_shape_indptr3, input_shape_data3);
    netTest3.fordward(vett_input,0);
    netTest3.fordward(vett_input,1);

/*
    int windowSpeed = 50;
    int windowAcc   = 50;
    int dimInput = windowSpeed + windowAcc;

    const mx_uint input_shape_indptrInvNet[] = {0, 2};
    const mx_uint input_shape_dataInvNet[] = {1, static_cast<mx_uint>(dimInput)};
    const char* input_keyInvNet[1] = {"/state_in1"};

    std::vector<mx_float> invNetInput = std::vector<mx_float>(dimInput);
    for (int i = 0 ; i < dimInput ; i++){
        invNetInput[i] = 0.3;
    }

    net_classes::MXNetWrapper invNet = net_classes::MXNetWrapper((std::string)"networks/invNet0", 1, input_keyInvNet, input_shape_indptrInvNet, input_shape_dataInvNet);
    invNet.fordward(invNetInput);

    for (int i = 0 ; i < dimInput ; i++){
        invNetInput[i] = 0.1;
    }
    invNet.fordward(invNetInput);

    for (int i = 0 ; i < dimInput ; i++){
        invNetInput[i] = 0.3;
    }

    invNet.fordward(invNetInput);
    invNet.fordward(invNetInput);
    invNet.fordward(invNetInput);
    invNet.fordward(invNetInput);
    invNet.fordward(invNetInput);
    invNet.fordward(invNetInput);
*/
}
