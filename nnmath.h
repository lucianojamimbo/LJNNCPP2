#include <vector>
#include <iostream>
#include <cmath>




//==========multiplication==========
//dot product of two vectors
template <typename T>
float dot(const std::vector<T>& a,
	  const std::vector<T>& b,
	  T& product){ //product will be modified by this function
  if (a.size() != b.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "b.size(): " << b.size() << std::endl;
    throw std::runtime_error("dot product received vectors of different sizes");
  }
  product = 0;
  for (int i = 0; i < a.size(); i++){
    product += a[i] * b[i];
  }
  return product;
}
//hadamard product for single vectors
template <typename T>
std::vector<T> hadamard(const std::vector<T>& a,
			const std::vector<T>& b,
			std::vector<T>& product //all 3 of these vectors must have identical sizes.
			){
  if (a.size() != b.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "b.size(): " << b.size() << std::endl;
    throw std::runtime_error("hadamard received inputs of different sizes");
  }
  if (a.size() !=  product.size()){
    std::cout << "WARN: hardamard 'product' input size != 'a' input or 'b' input size" << std::endl;
  }
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] * b[i];
  }
  return product;
}
//multiply all items in vector<float> a by float b
std::vector<float> vectbyscalarmultiply(const std::vector<float>& a,
				  const float& b,
				  std::vector<float>& product){
  if (a.size() != product.size()){
    std::cout << "WARN: scalarmultiply received vectors of different size for 'a' and 'product'" << std::endl;
  }
  for (int item = 0; item < a.size(); item++){
    product[item] = (a[item] * b);
  }
  return product;
}



//==========addition/subtraction=========
//adds two vectors of the same size
template <typename T>
std::vector<T> vectadd(const std::vector<T>& a,
		       const std::vector<T>& b,
		       std::vector<T>& product){ //product will be modified by this function
  if (a.size() != b.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "b.size(): " << b.size() << std::endl;
    throw std::runtime_error("vect addition received inputs of different sizes");
  }
  if (a.size() !=  product.size()){
    std::cout << "WARN: vectadd product input size != a input or b input size" << std::endl;
  }
  for (int i = 0; i < a.size(); i ++){
    product[i] = a[i] + b[i];
  }
  return product;
}
//sets product to vector a - vector b
template <typename T>
std::vector<T> vectsub(const std::vector<T>& a,
		       const std::vector<T>& b,
		       std::vector<T>& product){ //product will be modified by this function
  if (a.size() != b.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "b.size(): " << b.size() << std::endl;
    throw std::runtime_error("vect subtraction received inputs of different sizes!");
  }
  if (a.size() !=  product.size()){
    std::cout << "WARN: vectsub product input size != a input or b input size" << std::endl;
  }
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] - b[i];
  }
  return product;
}




//==========other==========
//transpose vector of vectors
template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& a,
				      std::vector<std::vector<T>>& transposed){ //transposed will be modified by this function
  if (&a == &transposed){
    throw std::runtime_error("do not pass the same variable for both inputs to transpose");
  }
  transposed.resize(a[0].size()); //resizing could be maybe possibly moved out of function but probably wouldnt speed things up much
  for (int i = 0; i < transposed.size(); i++){
    transposed[i].resize(a.size());
  }
  for (std::vector<int>::size_type i = 0; i < a[0].size(); i++){
    for (std::vector<int>::size_type j = 0; j < a.size(); j++){
      transposed[i][j] = a[j][i];
    }
  }
  return transposed;
}
//sigmoid function
float sigmoid(const float& a,
	     float& tomod){ //tomod will be modified by this function
  tomod = 1/(1+exp(-a));
  return tomod;
}
//sigmoid prime
float sigmoidprime(const float& a,
		  float& tomod){ //tomod will be modified by this function
  sigmoid(a, tomod);
  tomod = tomod*(1-tomod);
  return tomod;
}
//sigmoid function for vectors
std::vector<float> vectsigmoid(const std::vector<float>& a,
			       std::vector<float>& vecttomod){ //vecttomod will be modified by this function
  if (a.size() != vecttomod.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "vecttomod.size(): " << vecttomod.size() << std::endl;
    throw std::runtime_error("vectsigmoids received inputs were of different size");
  }
  for (int i = 0; i < a.size(); i++){
    sigmoid(a[i], vecttomod[i]);
  }
  return vecttomod;
}
//sigmoid prime for vectors
std::vector<float> vectsigmoidprime(const std::vector<float>& a,
				    std::vector<float>& vecttomod){ //vecttomod will be modified by this function
  if (a.size() != vecttomod.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "vecttomod.size(): " << vecttomod.size() << std::endl;
    throw std::runtime_error("vectsigmoidprimes received inputs were of different size");
  }
  for (int i = 0; i < a.size(); i++){
    sigmoidprime(a[i], vecttomod[i]);
  }
  return vecttomod;
}
