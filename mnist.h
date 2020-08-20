#include <vector>
#include <fstream>
#include <iostream>

int chars_to_int (const char* buf){
  int rtn = (buf[3]&0xff) | (buf[2]&0xff)<<8 | (buf[1]&0xff)<<16 | (buf[0]&0xff)<<24;
  return rtn;
}

std::vector<int> loadlabels(){
  std::ifstream t_lab;
  t_lab.open("train-labels-idx1-ubyte", std::ios::in | std::ios::binary);
  //process file data
  char buff[4];
  t_lab.read(buff, 4);
  int magic_labs = chars_to_int (buff);
  t_lab.read(buff, 4);
  int n_labs = chars_to_int (buff);
  std::cout << "n_labs " << n_labs << std::endl;
  //create and resize labels vector
  std::vector<int> labels;
  labels.resize(n_labs);
  //read values into labels vector
  char lbuff[1];
  for (int label = 0; label < n_labs; label++){
    t_lab.read(lbuff, 1);
    unsigned char uc = lbuff[0];
    labels[label] = uc;
  }
  return labels;
}

std::vector<std::vector<float>> loadimages(){
  std::ifstream t_img;
  t_img.open("train-images-idx3-ubyte", std::ios::in | std::ios::binary);
  //process file data
  char buff[4];
  t_img.read(buff, 4);
  int magic_imgs = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_imags = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_rows = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_cols = chars_to_int (buff);
  std::cout << "n_imags " << n_imags << std::endl;
  //create and resize images vector
  std::vector<std::vector<float>> images;
  images.resize(n_imags);
  for (int i = 0; i < n_imags; i++){
    images[i].resize(n_cols*n_rows);
  }
  //read values into images vector
  char pbuff[1];
  for (int image = 0; image < n_imags; image++){
    for (int pixel = 0; pixel < (n_rows*n_cols); pixel++){
      t_img.read(pbuff, 1);
      unsigned char uc = pbuff[0];
      images[image][pixel] = uc;
    }
  }
  return images;
}

std::vector<int> loadtestlabels(){
  std::ifstream t_lab;
  t_lab.open("t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);
  //process file data
  char buff[4];
  t_lab.read(buff, 4);
  int magic_labs = chars_to_int (buff);
  t_lab.read(buff, 4);
  int n_labs = chars_to_int (buff);
  std::cout << "n_labs " << n_labs << std::endl;
  //create and resize labels vector
  std::vector<int> labels;
  labels.resize(n_labs);
  //read values into labels vector
  char lbuff[1];
  for (int label = 0; label < n_labs; label++){
    t_lab.read(lbuff, 1);
    unsigned char uc = lbuff[0];
    labels[label] = uc;
  }
  return labels;
}

std::vector<std::vector<float>> loadtestimages(){
  std::ifstream t_img;
  t_img.open("t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
  //process file data
  char buff[4];
  t_img.read(buff, 4);
  int magic_imgs = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_imags = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_rows = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_cols = chars_to_int (buff);
  std::cout << "n_imags " << n_imags << std::endl;
  //create and resize images vector
  std::vector<std::vector<float>> images;
  images.resize(n_imags);
  for (int i = 0; i < n_imags; i++){
    images[i].resize(n_cols*n_rows);
  }
  //read values into images vector
  char pbuff[1];
  for (int image = 0; image < n_imags; image++){
    for (int pixel = 0; pixel < (n_rows*n_cols); pixel++){
      t_img.read(pbuff, 1);
      unsigned char uc = pbuff[0];
      images[image][pixel] = uc;
    }
  }
  return images;
}
