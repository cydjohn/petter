//
//  LFLivePreview.h
//  LFLiveKit
//
//  Created by 倾慕 on 16/5/2.
//  Copyright © 2016年 live Interactive. All rights reserved.
//

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

#import <LFLiveKit/LFLiveKit.h>

#import <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

@interface LFLivePreview : UIView {
    NSMutableDictionary* oldPredictionValues;
    NSMutableArray* labelLayers;
    NSTimer* timer;
    
    std::vector<std::string> labels;
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    double total_latency;
    int total_count;
}
@property(strong, nonatomic) CATextLayer* predictionTextLayer;

@end
