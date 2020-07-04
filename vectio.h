#include <vector>
#include <iostream>


//print out each item of a vector
template <typename T> 
void printV(const std::vector<T>& a){
  for (auto& i : a){
    std::cout << i << ", ";
  }
  std::cout << std::endl;
}
//print out each item of a vector of vectors
template <typename T>
void printVV(const std::vector<std::vector<T>>& a){
  for (auto& i : a){
    for (auto& i2 : i){
      std::cout << i2 << ", ";
    }
    std::cout << std::endl;
  }
}
//print out each item of a vector of vectors of vectors
template <typename T>
void printVVV(const std::vector<std::vector<std::vector<T>>>& a){
  for (auto& i : a){
    for (auto& i2 : i){
      for (auto& i3 : i2){
	std::cout << i3 << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "===" << std::endl;
  }
}
