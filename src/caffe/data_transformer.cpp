#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
namespace caffe {


template<typename Dtype>
DataTransformer<Dtype>::DataTransformer (const TransformationParameter& param) 
    : param_(param) {
    phase_ = Caffe::phase();
}

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
  const bool distort = param_.distort();

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
 
  double angle1=0;
  double angle2=0;
  double alpha=1; // aspect ratio
  double shear=0;
  double iscale=1; // pixel intensity scaling
  if (train_flag && randsize ) {
	  resize_scale=resize_scale*(1.0+((Rand() % 400)-200.0)/1000.0);
      angle1=Rand() % 360;
      angle2=Rand()% 360;
      alpha=1+(double((Rand()%400)-200.0)/1000.0);
      iscale=1+(double((Rand()%400-200.0))/1000.0);  
      shear=double(Rand()%400-200.0)/1000.0;
  }
/*
  if (resize_scale>1.0) {
	  resize_scale=1.0;
  }
  */ 
  cv::Mat r( 2, 3,  cv::DataType<double>::type );
 
  cv::Mat r1( 2, 3,  cv::DataType<double>::type );
  cv::Mat row = (cv::Mat_<double>(1,3) << 0,0,1);
  r1 = cv::getRotationMatrix2D(pt, angle1,1);
  //change aspect ratio
  cv::Mat C = (cv::Mat_<double>(3,3) << alpha, 0, (1-alpha)*pt.x, 0,1, 0, 0,0,1); 
  r1.push_back(row);


  // Shear
  cv::Mat r2(2,3,cv::DataType<double>::type);
  r2=cv::getRotationMatrix2D(pt,angle2,1);
  r2.push_back(row);
  cv::Mat S = (cv::Mat_<double>(3,3) << 1, shear, -shear*pt.y, 0,1, 0, 0,0,1); 
  

  r = cv::getRotationMatrix2D(pt, angle,resize_scale);
  r.push_back(row);

  r=r*S*r2*C*r1;  

  cv::Mat m=r(cv::Range(0,2),cv::Range::all());

  m.at<double>(0,2) -= double(width-resize)/2.0;
  m.at<double>(1,2) -= double(height-resize)/2.0;
  
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

          if (distort && (Rand() % 5)) {

            cv::Mat dst1(height,width,cv::DataType<Dtype>::type);  
            cv::Mat mapx(height,width,CV_32FC1);
            cv::Mat mapy(height,width,CV_32FC1);
	    double norm=64*64/10;
	    double axx= (Rand()% 200-100.0)/100.0/norm;
	    double axy= (Rand()% 200-100.0)/100.0/norm;
	    double ayy= (Rand()% 200-100.0)/100.0/norm;

	    double bxx= (Rand()% 200-100.0)/100.0/norm;
	    double bxy= (Rand()% 200-100.0)/100.0/norm;
	    double byy= (Rand()% 200-100.0)/100.0/norm;
	    	
            for (int i=0;i<height;i++) { //rows
                for (int j=0;j<width;j++) { //cols
		    double jc=j-128;
	            double ic=i-128;	    
                    mapx.at<float>(i,j)=j+(axx*jc*jc+2*axy*jc*ic+ayy*ic*ic);
                    mapy.at<float>(i,j)=i+(bxx*ic*ic+2*bxy*jc*ic+byy*jc*jc);
                }
            }
            cv::remap(src,dst1,mapx,mapy,cv::INTER_CUBIC);
	        cv::warpAffine(dst1, dst, m, dst.size(),cv::INTER_CUBIC);
          }

          else {
    		  cv::warpAffine(src, dst, m, dst.size(),cv::INTER_CUBIC); 
          }

		  //std::cout<<dst<<std::endl; 
		  for (int j = 0; j < resize*resize; ++j) {
            double v=iscale*dst.at<Dtype>(j);
            if (v>255) {
                v=255;
            }
			temp[j+c*resize*resize] = int(v);
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
