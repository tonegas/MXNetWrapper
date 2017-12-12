
#include "MXNetWrapper.cc"


int main(int argc, char* argv[]){
//------------------------------------------------------------------------------
//-------------------------Example with Inception Network-----------------------
//------------------------------------------------------------------------------
  int width = 224;
  int height = 224;
  int channels = 3;
  int dim = width*height*channels;
  const mx_uint input_shape_indptr[] = { 0, 4 };
  const mx_uint input_shape_data[] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(height),
                                        static_cast<mx_uint>(width)};
  const char* input_key[1];
  input_key[0] = {"data"};

  std::vector<mx_float> vett_float = std::vector<mx_float>(dim);
  for (int i = 0 ; i < dim ; i++){
      vett_float[i] = 0.0;
  }

  net_classes::MXNetWrapper net = net_classes::MXNetWrapper((std::string)"../networks/Inception-BN",  input_key, input_shape_indptr, input_shape_data);
  net.fordward(vett_float);

//------------------------------------------------------------------------------
//----------------------------Example with my Graph-----------------------------
//------------------------------------------------------------------------------

  //
  // W is a matrix 2 row 3 column
  // b is a vector of 2
  // input is a vetor of 3
  // output is a vetor of 2
  // output = W * input + b
  //
  // Are these inputs right?
  //----------------------------------------------------------
    const mx_uint input_shape_indptr2[] = { 0, 2 };
    const mx_uint input_shape_data2[] = { 1, 3 };
    const char* input_key2[1];
    input_key2[0] = {"Input"};
  //----------------------------------------------------------

    net_classes::MXNetWrapper net2 = net_classes::MXNetWrapper((std::string)"../networks/netTest", input_key2, input_shape_indptr2, input_shape_data2);
    std::vector<mx_float> vett_float2 = {1.2,1.0,2.2};
    net2.fordward(vett_float2);

    net_classes::MXNetWrapper net3 = net_classes::MXNetWrapper((std::string)"../networks/netTest", input_key2, input_shape_indptr2, input_shape_data2);
    net3.fordward(vett_float2);

}
