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

  // SR: fix for  bug https://github.com/BVLC/caffe/issues/1430
  Caffe::Phase phase_ = Caffe::phase(); 
  // 
 
  bool train_flag= (phase_ == Caffe::TRAIN);
  const string& data = datum.data();
  const int channels = datum.channels();
  int height = datum.height();
  int width = datum.width();
  int size = datum.channels() * datum.height() * datum.width();
  int actual_size = datum.actual_size();
  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  const bool rotate = param_.rotate();	
  const int resize = param_.resize();
  const bool randsize = param_.randsize();	

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }
  Dtype* temp;
  CHECK(resize) << "resize must be given";
  temp= new Dtype [channels*resize*resize];
  cv::Point2f pt(width/2., height/2.);
  double angle=0;
  if (train_flag && rotate) {
	  angle=Rand() % 360;
  }
	    
  if (actual_size> std::max(height,width)) {
	  actual_size=std::max(height,width);
  }
  double resize_scale=double(resize)/double(actual_size);
 
  if (train_flag && randsize ) {
	  resize_scale=resize_scale*(1.0+((Rand() % 200)-100.0)/1000.0);
  }
/*
  if (resize_scale>1.0) {
	  resize_scale=1.0;
  }
  */ 
  cv::Mat r( 2, 3,  cv::DataType<double>::type );
  r = cv::getRotationMatrix2D(pt, angle, resize_scale);


  r.at<double>(0,2) -= double(width-resize)/2.0;
  r.at<double>(1,2) -= double(height-resize)/2.0;
  
  //std::cout<<r<<std::endl;
  
  if (resize*resize!=data.size() || angle != 0 || resize_scale!=1.0) {	
	  for (int c = 0; c < channels; ++c) {
		  cv::Mat dst(resize,resize,cv::DataType<Dtype>::type);

		  cv::Mat src(height, width,cv::DataType<Dtype>::type);
		  for (int h = 0; h < height; ++h) {
		    for (int w = 0; w < width; ++w) {
			int data_index = (c * height + h) * width + w;
			Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
			src.at<Dtype>(h,w)=datum_element;
		    }
		  }	  
		   
		  cv::warpAffine(src, dst, r, dst.size()); 
		  //std::cout<<dst<<std::endl; 
		  for (int j = 0; j < resize*resize; ++j) {
			temp[j+c*resize*resize] = dst.at<Dtype>(j);
		  }	
	  }
	  height=resize;
	  width=resize;
	  size=channels*resize*resize;
  }
  else {
  	for (int c = 0; c < channels; ++c) {
	  for (int h = 0; h < height; ++h) {
	    for (int w = 0; w < width; ++w) {
		int data_index = (c * height + h) * width + w;
		Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
		temp[data_index] = datum_element;
	    }	
	  }
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
            Dtype datum_element = temp[data_index];
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
            Dtype datum_element = temp[data_index];
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
        Dtype datum_element = temp[j];
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
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
