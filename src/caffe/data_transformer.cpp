#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  const bool rotate = param_.rotate();	
  const int resize = param_.resize();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }
  Dtype* temp;
  CHECK(resize) << "resize must be given";
  temp= new Dtype [channels*resize*resize];
  cv::Point2f pt(width/2., height/2.);
  double angle=0;
  if (rotate) {
	  angle=Rand() % 360;
  }
  double rangle=angle/180.*M_PI;
  int height1=ceil(height*abs(cos(rangle))+width*abs(sin(rangle)));
  int width1=ceil(height*abs(sin(rangle))+width*abs(cos(rangle)));
	    
  double resize_scale=double(resize)/std::max(height1,width1);
  if (resize_scale<1.0) {
	  resize_scale=1.0;
  } 
  cv::Mat r( 2, 3,  cv::DataType<float>::type );
  r = cv::getRotationMatrix2D(pt, angle, resize_scale);
  r.at<float>(0,2) += (resize- width)/2;
  r.at<float>(1,2) += (resize- height)/2;	    
  for (int c = 0; c < channels; ++c) {
	  cv::Mat dst(resize,resize,cv::DataType<Dtype>::type);

	  cv::Mat src(height, width,cv::DataType<Dtype>::type);
	  for (int h = 0; h < crop_size; ++h) {
	    for (int w = 0; w < crop_size; ++w) {
		int data_index = (c * height + h) * width + w;
		Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
		src.at<Dtype>(h,w)=datum_element;
	    }
	  }	  
	  cv::warpAffine(src, dst, r, dst.size());  
	  for (int j = 0; j < resize*resize; ++j) {
		temp[j+c*resize*resize] = dst.at<Dtype>(j);
	  }	
  }
  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
	//std::cout<<"mean: "<< mean[j]<< " "<<datum_element<< " "<<transformed_data[j+batch_item_id*size]<<std::endl;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
  delete[] temp;
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size() || param_.rotate());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
