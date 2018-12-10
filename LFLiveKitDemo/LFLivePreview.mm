//
//  LFLivePreview.m
//  LFLiveKit
//
//  Created by 倾慕 on 16/5/2.
//  Copyright © 2016年 live Interactive. All rights reserved.
//

#import "LFLivePreview.h"
#import "UIControl+YYAdd.h"
#import "UIView+YYAdd.h"
#import "LFLiveKit.h"

#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"

#define LOG(x) std::cerr

namespace {
    
    // If you have your own model, modify this to the file name, and make sure
    // you've added the file to your app resources too.
    NSString* model_file_name = @"converted_model";
    NSString* model_file_type = @"tflite";
    // If you have your own model, point this to the labels file.
    NSString* labels_file_name = @"label1";
    NSString* labels_file_type = @"txt";
    
    // These dimensions need to match those the model was trained with.
    const int wanted_input_width = 100;
    const int wanted_input_height = 100;
    const int wanted_input_channels = 3;
    const float input_mean = 128;
    const float input_std = 128;
    const std::string input_layer_name = "input";
    const std::string output_layer_name = "softmax1";
    
    NSString* FilePathForResourceName(NSString* name, NSString* extension) {
        NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
        if (file_path == NULL) {
            LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
            << "' in bundle.";
        }
        return file_path;
    }
    
    void LoadLabels(NSString* file_name, NSString* file_type, std::vector<std::string>* label_strings) {
        NSString* labels_path = FilePathForResourceName(file_name, file_type);
        if (!labels_path) {
            LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
            << [file_type UTF8String];
        }
        std::ifstream t;
        t.open([labels_path UTF8String]);
        std::string line;
        while (t) {
            std::getline(t, line);
            label_strings->push_back(line);
        }
        t.close();
    }
    
    // Returns the top N confidence values over threshold in the provided vector,
    // sorted by confidence in descending order.
    void GetTopN(
                 const float* prediction, const int prediction_size, const int num_results,
                 const float threshold, std::vector<std::pair<float, int> >* top_results) {
        // Will contain top N results in ascending order.
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
        std::greater<std::pair<float, int> > >
        top_result_pq;
        
        const long count = prediction_size;
        for (int i = 0; i < count; ++i) {
            const float value = prediction[i];
            // Only add it if it beats the threshold and has a chance at being in
            // the top N.
            if (value < threshold) {
                continue;
            }
            
            top_result_pq.push(std::pair<float, int>(value, i));
            
            // If at capacity, kick the smallest value out.
            if (top_result_pq.size() > num_results) {
                top_result_pq.pop();
            }
        }
        
        // Copy to output vector and reverse into descending order.
        while (!top_result_pq.empty()) {
            top_results->push_back(top_result_pq.top());
            NSLog(@"%i", top_result_pq.top().second);
            top_result_pq.pop();
        }
        std::reverse(top_results->begin(), top_results->end());
    }
    
    // Preprocess the input image and feed the TFLite interpreter buffer for a float model.
    void ProcessInputWithFloatModel(
                                    uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
        for (int y = 0; y < wanted_input_height; ++y) {
            float* out_row = buffer + (y * wanted_input_width * wanted_input_channels);
            for (int x = 0; x < wanted_input_width; ++x) {
                const int in_x = (y * image_width) / wanted_input_width;
                const int in_y = (x * image_height) / wanted_input_height;
                uint8_t* input_pixel =
                input + (in_y * image_width * image_channels) + (in_x * image_channels);
                float* out_pixel = out_row + (x * wanted_input_channels);
                for (int c = 0; c < wanted_input_channels; ++c) {
                    out_pixel[c] = (input_pixel[c] - input_mean) / input_std;
                }
            }
        }
    }
    
    // Preprocess the input image and feed the TFLite interpreter buffer for a quantized model.
    void ProcessInputWithQuantizedModel(
                                        uint8_t* input, uint8_t* output, int image_width, int image_height, int image_channels) {
        for (int y = 0; y < wanted_input_height; ++y) {
            uint8_t* out_row = output + (y * wanted_input_width * wanted_input_channels);
            for (int x = 0; x < wanted_input_width; ++x) {
                const int in_x = (y * image_width) / wanted_input_width;
                const int in_y = (x * image_height) / wanted_input_height;
                uint8_t* in_pixel = input + (in_y * image_width * image_channels) + (in_x * image_channels);
                uint8_t* out_pixel = out_row + (x * wanted_input_channels);
                for (int c = 0; c < wanted_input_channels; ++c) {
                    out_pixel[c] = in_pixel[c];
                }
            }
        }
    }
    
}  // namespace

inline static NSString *formatedSpeed(float bytes, float elapsed_milli) {
    if (elapsed_milli <= 0) {
        return @"N/A";
    }

    if (bytes <= 0) {
        return @"0 KB/s";
    }

    float bytes_per_sec = ((float)bytes) * 1000.f /  elapsed_milli;
    if (bytes_per_sec >= 1000 * 1000) {
        return [NSString stringWithFormat:@"%.2f MB/s", ((float)bytes_per_sec) / 1000 / 1000];
    } else if (bytes_per_sec >= 1000) {
        return [NSString stringWithFormat:@"%.1f KB/s", ((float)bytes_per_sec) / 1000];
    } else {
        return [NSString stringWithFormat:@"%ld B/s", (long)bytes_per_sec];
    }
}

@interface LFLivePreview ()<LFLiveSessionDelegate>

@property (nonatomic, strong) UIButton *beautyButton;
@property (nonatomic, strong) UIButton *cameraButton;
@property (nonatomic, strong) UIButton *closeButton;
@property (nonatomic, strong) UIButton *startLiveButton;
@property (nonatomic, strong) UIView *containerView;
@property (nonatomic, strong) LFLiveDebug *debugInfo;
@property (nonatomic, strong) LFLiveSession *session;
@property (nonatomic, strong) UILabel *stateLabel;

@end

@implementation LFLivePreview

- (void)runModelOnFrame:(CVPixelBufferRef)pixelBuffer {
    assert(pixelBuffer != NULL);
    
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
           sourcePixelFormat == kCVPixelFormatType_32BGRA);
    
    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
    int image_height;
    unsigned char* sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    const int image_channels = 4;
    assert(image_channels >= wanted_input_channels);
    uint8_t* in = sourceStartAddr;
    
    int input = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input);
    
    bool is_quantized;
    switch (input_tensor->type) {
        case kTfLiteFloat32:
            is_quantized = false;
            break;
        case kTfLiteUInt8:
            is_quantized = true;
            break;
        default:
            NSLog(@"Input data type is not supported by this demo app.");
            return;
    }
    
    if (is_quantized) {
        uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
        ProcessInputWithQuantizedModel(in, out, image_width, image_height, image_channels);
    } else {
        float* out = interpreter->typed_tensor<float>(input);
        ProcessInputWithFloatModel(in, out, image_width, image_height, image_channels);
    }
    
    double start = [[NSDate new] timeIntervalSince1970];
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    double end = [[NSDate new] timeIntervalSince1970];
    total_latency += (end - start);
    total_count += 1;
    NSLog(@"Time: %.4lf, avg: %.4lf, count: %d", end - start, total_latency / total_count,
          total_count);
    
    const int output_size = 10;
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    
    std::vector<std::pair<float, int> > top_results;
    
    if (is_quantized) {
        uint8_t* quantized_output = interpreter->typed_output_tensor<uint8_t>(0);
        int32_t zero_point = input_tensor->params.zero_point;
        float scale = input_tensor->params.scale;
        float output[output_size];
        for (int i = 0; i < output_size; ++i) {
            output[i] = (quantized_output[i] - zero_point) * scale;
        }
        GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    } else {
        float* output = interpreter->typed_output_tensor<float>(0);
        GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    }
    
    NSMutableDictionary* newValues = [NSMutableDictionary dictionary];
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        NSString* labelObject = [NSString stringWithUTF8String:labels[index].c_str()];
        NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
        [newValues setObject:valueObject forKey:labelObject];
    }
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        [self setPredictionValues:newValues];
    });
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)setPredictionValues:(NSDictionary*)newValues {
    const float decayValue = 0.75f;
    const float updateValue = 0.25f;
    const float minimumThreshold = 0.01f;
    
    NSMutableDictionary* decayedPredictionValues = [[NSMutableDictionary alloc] init];
    for (NSString* label in oldPredictionValues) {
        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        const float decayedPredictionValue = (oldPredictionValue * decayValue);
        if (decayedPredictionValue > minimumThreshold) {
            NSNumber* decayedPredictionValueObject = [NSNumber numberWithFloat:decayedPredictionValue];
            [decayedPredictionValues setObject:decayedPredictionValueObject forKey:label];
        }
    }
    oldPredictionValues = decayedPredictionValues;
    
    for (NSString* label in newValues) {
        NSNumber* newPredictionValueObject = [newValues objectForKey:label];
        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
        if (!oldPredictionValueObject) {
            oldPredictionValueObject = [NSNumber numberWithFloat:0.0f];
        }
        const float newPredictionValue = [newPredictionValueObject floatValue];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        const float updatedPredictionValue = (oldPredictionValue + (newPredictionValue * updateValue));
        NSNumber* updatedPredictionValueObject = [NSNumber numberWithFloat:updatedPredictionValue];
        [oldPredictionValues setObject:updatedPredictionValueObject forKey:label];
    }
    NSArray* candidateLabels = [NSMutableArray array];
    for (NSString* label in oldPredictionValues) {
        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        if (oldPredictionValue > 0.05f) {
            NSDictionary* entry = @{@"label" : label, @"value" : oldPredictionValueObject};
            candidateLabels = [candidateLabels arrayByAddingObject:entry];
        }
    }
    NSSortDescriptor* sort = [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
    NSArray* sortedLabels =
    [candidateLabels sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];
    
    const float leftMargin = 10.0f;
    const float topMargin = 10.0f;
    
    const float valueWidth = 48.0f;
    const float valueHeight = 18.0f;
    
    const float labelWidth = 246.0f;
    const float labelHeight = 18.0f;
    
    const float labelMarginX = 5.0f;
    const float labelMarginY = 5.0f;
    
    [self removeAllLabelLayers];
    
    int labelCount = 0;
    for (NSDictionary* entry in sortedLabels) {
        NSString* label = [entry objectForKey:@"label"];
        NSNumber* valueObject = [entry objectForKey:@"value"];
        const float value = [valueObject floatValue];
        const float originY = topMargin + ((labelHeight + labelMarginY) * labelCount);
        const int valuePercentage = (int)roundf(value * 100.0f);
        
        const float valueOriginX = leftMargin;
        NSString* valueText = [NSString stringWithFormat:@"%d%%", valuePercentage];
        
        [self addLabelLayerWithText:valueText
                            originX:valueOriginX
                            originY:originY
                              width:valueWidth
                             height:valueHeight
                          alignment:kCAAlignmentRight];
        
        const float labelOriginX = (leftMargin + valueWidth + labelMarginX);
        
        [self addLabelLayerWithText:[label capitalizedString]
                            originX:labelOriginX
                            originY:originY
                              width:labelWidth
                             height:labelHeight
                          alignment:kCAAlignmentLeft];
        
        labelCount += 1;
        if (labelCount > 4) {
            break;
        }
    }
    self.stateLabel.text = [sortedLabels[0] objectForKey:@"label"];
    self.session.warterMarkView = self.stateLabel;
}

- (void)removeAllLabelLayers {
    for (CATextLayer* layer in labelLayers) {
        [layer removeFromSuperlayer];
    }
    [labelLayers removeAllObjects];
}

- (void)addLabelLayerWithText:(NSString*)text
                      originX:(float)originX
                      originY:(float)originY
                        width:(float)width
                       height:(float)height
                    alignment:(NSString*)alignment {
    CFTypeRef font = (CFTypeRef) @"Menlo-Regular";
    const float fontSize = 12.0;
    const float marginSizeX = 5.0f;
    const float marginSizeY = 2.0f;
    
    const CGRect backgroundBounds = CGRectMake(originX, originY, width, height);
    const CGRect textBounds = CGRectMake((originX + marginSizeX), (originY + marginSizeY),
                                         (width - (marginSizeX * 2)), (height - (marginSizeY * 2)));
    
    CATextLayer* background = [CATextLayer layer];
    [background setBackgroundColor:[UIColor blackColor].CGColor];
    [background setOpacity:0.5f];
    [background setFrame:backgroundBounds];
    background.cornerRadius = 5.0f;
    
    [[self layer] addSublayer:background];
    [labelLayers addObject:background];
    
    CATextLayer* layer = [CATextLayer layer];
    [layer setForegroundColor:[UIColor whiteColor].CGColor];
    [layer setFrame:textBounds];
    [layer setAlignmentMode:alignment];
    [layer setWrapped:YES];
    [layer setFont:font];
    [layer setFontSize:fontSize];
    layer.contentsScale = [[UIScreen mainScreen] scale];
    [layer setString:text];
    
    [[self layer] addSublayer:layer];
    [labelLayers addObject:layer];
}

- (instancetype)initWithFrame:(CGRect)frame {
    if (self = [super initWithFrame:frame]) {
        self.backgroundColor = [UIColor clearColor];
        [self requestAccessForVideo];
        [self requestAccessForAudio];
        [self addSubview:self.containerView];
//        [self.containerView addSubview:self.stateLabel];
        self.session.warterMarkView = self.stateLabel;
//        [self.containerView addSubview:self.closeButton];
        [self.containerView addSubview:self.cameraButton];
//        [self.containerView addSubview:self.beautyButton];
        [self.containerView addSubview:self.startLiveButton];
        
        labelLayers = [[NSMutableArray alloc] init];
        oldPredictionValues = [[NSMutableDictionary alloc] init];
        
        NSString* graph_path = FilePathForResourceName(model_file_name, model_file_type);
        model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
        if (!model) {
            LOG(FATAL) << "Failed to mmap model " << graph_path;
        }
        LOG(INFO) << "Loaded model " << graph_path;
        model->error_reporter();
        LOG(INFO) << "resolved reporter";
        
        tflite::ops::builtin::BuiltinOpResolver resolver;
        LoadLabels(labels_file_name, labels_file_type, &labels);
        
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        // Explicitly resize the input tensor.
        {
            int input = interpreter->inputs()[0];
            std::vector<int> sizes = {1, 100, 100, 3};
            interpreter->ResizeInputTensor(input, sizes);
        }
        if (!interpreter) {
            LOG(FATAL) << "Failed to construct interpreter";
        }
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            LOG(FATAL) << "Failed to allocate tensors!";
        }
        timer = [NSTimer scheduledTimerWithTimeInterval:0.1 target:self selector:@selector(Timered:) userInfo:nil repeats:YES];
    }
    return self;
}

- (void)Timered:(NSTimer*)timer {
    CVPixelBufferRef pixelBuffer = [self CVPixelBufferRefFromUiImage:self.session.currentImage];
    [self runModelOnFrame:pixelBuffer];
}

#pragma mark -- Public Method
// 请求摄像许可
- (void)requestAccessForVideo {
    __weak typeof(self) _self = self;
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    switch (status) {
    case AVAuthorizationStatusNotDetermined: {
        // 许可对话没有出现，发起授权许可
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
                if (granted) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [_self.session setRunning:YES];
                    });
                }
            }];
        break;
    }
    case AVAuthorizationStatusAuthorized: {
        // 已经开启授权，可继续
        dispatch_async(dispatch_get_main_queue(), ^{
            [_self.session setRunning:YES];
        });
        break;
    }
    case AVAuthorizationStatusDenied:
    case AVAuthorizationStatusRestricted:
        // 用户明确地拒绝授权，或者相机设备无法访问

        break;
    default:
        break;
    }
}

// 请求录音许可
- (void)requestAccessForAudio {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    switch (status) {
    case AVAuthorizationStatusNotDetermined: {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
            }];
        break;
    }
    case AVAuthorizationStatusAuthorized: {
        break;
    }
    case AVAuthorizationStatusDenied:
    case AVAuthorizationStatusRestricted:
        break;
    default:
        break;
    }
}

#pragma mark -- LFStreamingSessionDelegate
/** live status changed will callback */
- (void)liveSession:(nullable LFLiveSession *)session liveStateDidChange:(LFLiveState)state {
    NSLog(@"liveStateDidChange: %ld", state);
    switch (state) {
    case LFLiveReady:
        _stateLabel.text = @"未连接";
        break;
    case LFLivePending:
        _stateLabel.text = @"连接中";
        break;
    case LFLiveStart:
        _stateLabel.text = @"已连接";
        break;
    case LFLiveError:
        _stateLabel.text = @"连接错误";
        break;
    case LFLiveStop:
        _stateLabel.text = @"未连接";
        break;
    default:
        break;
    }
}

/** live debug info callback */
- (void)liveSession:(nullable LFLiveSession *)session debugInfo:(nullable LFLiveDebug *)debugInfo {
    NSLog(@"debugInfo uploadSpeed: %@", formatedSpeed(debugInfo.currentBandwidth, debugInfo.elapsedMilli));
}

/** callback socket errorcode */
- (void)liveSession:(nullable LFLiveSession *)session errorCode:(LFLiveSocketErrorCode)errorCode {
    NSLog(@"errorCode: %ld", errorCode);
}

#pragma mark -- Getter Setter
- (LFLiveSession *)session {
    if (!_session) {
        /**      发现大家有不会用横屏的请注意啦，横屏需要在ViewController  supportedInterfaceOrientations修改方向  默认竖屏  ****/
        /**      发现大家有不会用横屏的请注意啦，横屏需要在ViewController  supportedInterfaceOrientations修改方向  默认竖屏  ****/
        /**      发现大家有不会用横屏的请注意啦，横屏需要在ViewController  supportedInterfaceOrientations修改方向  默认竖屏  ****/


        /***   默认分辨率368 ＊ 640  音频：44.1 iphone6以上48  双声道  方向竖屏 ***/
        LFLiveVideoConfiguration *videoConfiguration = [LFLiveVideoConfiguration new];
        videoConfiguration.videoSize = CGSizeMake(360, 640);
        videoConfiguration.videoBitRate = 800*1024;
        videoConfiguration.videoMaxBitRate = 1000*1024;
        videoConfiguration.videoMinBitRate = 500*1024;
        videoConfiguration.videoFrameRate = 24;
        videoConfiguration.videoMaxKeyframeInterval = 48;
        videoConfiguration.outputImageOrientation = UIInterfaceOrientationPortrait;
        videoConfiguration.autorotate = NO;
        videoConfiguration.sessionPreset = LFCaptureSessionPreset720x1280;
        _session = [[LFLiveSession alloc] initWithAudioConfiguration:[LFLiveAudioConfiguration defaultConfiguration] videoConfiguration:videoConfiguration captureType:LFLiveCaptureDefaultMask];
        _session.captureDevicePosition = AVCaptureDevicePositionBack;

        /**    自己定制单声道  */
        /*
           LFLiveAudioConfiguration *audioConfiguration = [LFLiveAudioConfiguration new];
           audioConfiguration.numberOfChannels = 1;
           audioConfiguration.audioBitrate = LFLiveAudioBitRate_64Kbps;
           audioConfiguration.audioSampleRate = LFLiveAudioSampleRate_44100Hz;
           _session = [[LFLiveSession alloc] initWithAudioConfiguration:audioConfiguration videoConfiguration:[LFLiveVideoConfiguration defaultConfiguration]];
         */

        /**    自己定制高质量音频96K */
        /*
           LFLiveAudioConfiguration *audioConfiguration = [LFLiveAudioConfiguration new];
           audioConfiguration.numberOfChannels = 2;
           audioConfiguration.audioBitrate = LFLiveAudioBitRate_96Kbps;
           audioConfiguration.audioSampleRate = LFLiveAudioSampleRate_44100Hz;
           _session = [[LFLiveSession alloc] initWithAudioConfiguration:audioConfiguration videoConfiguration:[LFLiveVideoConfiguration defaultConfiguration]];
         */

        /**    自己定制高质量音频96K 分辨率设置为540*960 方向竖屏 */

        /*
           LFLiveAudioConfiguration *audioConfiguration = [LFLiveAudioConfiguration new];
           audioConfiguration.numberOfChannels = 2;
           audioConfiguration.audioBitrate = LFLiveAudioBitRate_96Kbps;
           audioConfiguration.audioSampleRate = LFLiveAudioSampleRate_44100Hz;

           LFLiveVideoConfiguration *videoConfiguration = [LFLiveVideoConfiguration new];
           videoConfiguration.videoSize = CGSizeMake(540, 960);
           videoConfiguration.videoBitRate = 800*1024;
           videoConfiguration.videoMaxBitRate = 1000*1024;
           videoConfiguration.videoMinBitRate = 500*1024;
           videoConfiguration.videoFrameRate = 24;
           videoConfiguration.videoMaxKeyframeInterval = 48;
           videoConfiguration.orientation = UIInterfaceOrientationPortrait;
           videoConfiguration.sessionPreset = LFCaptureSessionPreset540x960;

           _session = [[LFLiveSession alloc] initWithAudioConfiguration:audioConfiguration videoConfiguration:videoConfiguration];
         */


        /**    自己定制高质量音频128K 分辨率设置为720*1280 方向竖屏 */

        /*
           LFLiveAudioConfiguration *audioConfiguration = [LFLiveAudioConfiguration new];
           audioConfiguration.numberOfChannels = 2;
           audioConfiguration.audioBitrate = LFLiveAudioBitRate_128Kbps;
           audioConfiguration.audioSampleRate = LFLiveAudioSampleRate_44100Hz;

           LFLiveVideoConfiguration *videoConfiguration = [LFLiveVideoConfiguration new];
           videoConfiguration.videoSize = CGSizeMake(720, 1280);
           videoConfiguration.videoBitRate = 800*1024;
           videoConfiguration.videoMaxBitRate = 1000*1024;
           videoConfiguration.videoMinBitRate = 500*1024;
           videoConfiguration.videoFrameRate = 15;
           videoConfiguration.videoMaxKeyframeInterval = 30;
           videoConfiguration.landscape = NO;
           videoConfiguration.sessionPreset = LFCaptureSessionPreset360x640;

           _session = [[LFLiveSession alloc] initWithAudioConfiguration:audioConfiguration videoConfiguration:videoConfiguration];
         */


        /**    自己定制高质量音频128K 分辨率设置为720*1280 方向横屏  */

        /*
           LFLiveAudioConfiguration *audioConfiguration = [LFLiveAudioConfiguration new];
           audioConfiguration.numberOfChannels = 2;
           audioConfiguration.audioBitrate = LFLiveAudioBitRate_128Kbps;
           audioConfiguration.audioSampleRate = LFLiveAudioSampleRate_44100Hz;

           LFLiveVideoConfiguration *videoConfiguration = [LFLiveVideoConfiguration new];
           videoConfiguration.videoSize = CGSizeMake(1280, 720);
           videoConfiguration.videoBitRate = 800*1024;
           videoConfiguration.videoMaxBitRate = 1000*1024;
           videoConfiguration.videoMinBitRate = 500*1024;
           videoConfiguration.videoFrameRate = 15;
           videoConfiguration.videoMaxKeyframeInterval = 30;
           videoConfiguration.landscape = YES;
           videoConfiguration.sessionPreset = LFCaptureSessionPreset720x1280;

           _session = [[LFLiveSession alloc] initWithAudioConfiguration:audioConfiguration videoConfiguration:videoConfiguration];
        */

        _session.delegate = self;
        _session.showDebugInfo = NO;
        // 这里和本身View绑定
        _session.preView = self;
        
        
        /*本地存储*/
//        _session.saveLocalVideo = YES;
//        NSString *pathToMovie = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents/Movie.mp4"];
//        unlink([pathToMovie UTF8String]); // If a file already exists, AVAssetWriter won't let you record new frames, so delete the old movie
//        NSURL *movieURL = [NSURL fileURLWithPath:pathToMovie];
//        _session.saveLocalVideoPath = movieURL;
        
        /*
        UIImageView *imageView = [[UIImageView alloc] init];
        imageView.alpha = 0.8;
        imageView.frame = CGRectMake(100, 100, 29, 29);
        imageView.image = [UIImage imageNamed:@"ios-29x29"];
        _session.warterMarkView = imageView;*/
        
    }
    return _session;
}

// 包含控件的UIView
- (UIView *)containerView {
    if (!_containerView) {
        _containerView = [UIView new];
        _containerView.frame = self.bounds;
        _containerView.backgroundColor = [UIColor clearColor];
        _containerView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    }
    return _containerView;
}

// 状态标签
- (UILabel *)stateLabel {
    if (!_stateLabel) {
        _stateLabel = [[UILabel alloc] initWithFrame:CGRectMake(20, 500, 80, 40)];
        _stateLabel.text = @"未连接";
        _stateLabel.textColor = [UIColor whiteColor];
        _stateLabel.font = [UIFont boldSystemFontOfSize:14.f];
    }
    return _stateLabel;
}

// 未设置任何功能的关闭按钮
- (UIButton *)closeButton {
    if (!_closeButton) {
        _closeButton = [UIButton new];
        _closeButton.size = CGSizeMake(44, 44);
        _closeButton.left = self.width - 10 - _closeButton.width;
        _closeButton.top = 20;
        [_closeButton setImage:[UIImage imageNamed:@"close_preview"] forState:UIControlStateNormal];
        _closeButton.exclusiveTouch = YES;
        __weak typeof(self) ws = self;
        [_closeButton addBlockForControlEvents:UIControlEventTouchUpInside block:^(id sender) {
            if([timer isValid]){
                [timer invalidate];
                timer = nil;
                [ws removeAllLabelLayers];
            }else {
                timer = [NSTimer scheduledTimerWithTimeInterval:1 target:self selector:@selector(Timered:) userInfo:nil repeats:YES];
            }
        }];
    }
    return _closeButton;
}

// 切换前置摄像头和后置摄像头的按钮
- (UIButton *)cameraButton {
    if (!_cameraButton) {
        _cameraButton = [UIButton new];
        _cameraButton.size = CGSizeMake(44, 44);
        _cameraButton.origin = CGPointMake(_closeButton.left - 10 - _cameraButton.width, 20);
        [_cameraButton setImage:[UIImage imageNamed:@"camra_preview"] forState:UIControlStateNormal];
        _cameraButton.exclusiveTouch = YES;
        __weak typeof(self) _self = self;
        // 添加点击事件
        [_cameraButton addBlockForControlEvents:UIControlEventTouchUpInside block:^(id sender) {
            AVCaptureDevicePosition devicePositon = _self.session.captureDevicePosition;
            _self.session.captureDevicePosition = (devicePositon == AVCaptureDevicePositionBack) ? AVCaptureDevicePositionFront : AVCaptureDevicePositionBack;
        }];
    }
    return _cameraButton;
}

// 开关美颜的
- (UIButton *)beautyButton {
    if (!_beautyButton) {
        _beautyButton = [UIButton new];
        _beautyButton.size = CGSizeMake(44, 44);
        _beautyButton.origin = CGPointMake(_cameraButton.left - 10 - _beautyButton.width, 20);
        [_beautyButton setImage:[UIImage imageNamed:@"camra_beauty"] forState:UIControlStateNormal];
        [_beautyButton setImage:[UIImage imageNamed:@"camra_beauty_close"] forState:UIControlStateSelected];
        _beautyButton.exclusiveTouch = YES;
        __weak typeof(self) _self = self;
        [_beautyButton addBlockForControlEvents:UIControlEventTouchUpInside block:^(id sender) {
            _self.session.beautyFace = !_self.session.beautyFace;
            _self.beautyButton.selected = !_self.session.beautyFace;
        }];
    }
    return _beautyButton;
}

// 开始直播的按钮
- (UIButton *)startLiveButton {
    if (!_startLiveButton) {
        _startLiveButton = [UIButton new];
        _startLiveButton.size = CGSizeMake(self.width - 60, 44);
        _startLiveButton.left = 30;
        _startLiveButton.bottom = self.height - 50;
        _startLiveButton.layer.cornerRadius = _startLiveButton.height/2;
        [_startLiveButton setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
        [_startLiveButton.titleLabel setFont:[UIFont systemFontOfSize:16]];
        [_startLiveButton setTitle:@"开始直播" forState:UIControlStateNormal];
        [_startLiveButton setBackgroundColor:[UIColor colorWithRed:50 green:32 blue:245 alpha:1]];
        _startLiveButton.exclusiveTouch = YES;
        __weak typeof(self) _self = self;
        [_startLiveButton addBlockForControlEvents:UIControlEventTouchUpInside block:^(id sender) {
            _self.startLiveButton.selected = !_self.startLiveButton.selected;
            if (_self.startLiveButton.selected) {
                [_self.startLiveButton setTitle:@"结束直播" forState:UIControlStateNormal];
                LFLiveStreamInfo *stream = [LFLiveStreamInfo new];
                stream.url = @"rtmp://39.108.210.48:1935/live/cai";
                [_self.session startLive:stream];
            } else {
                [_self.startLiveButton setTitle:@"开始直播" forState:UIControlStateNormal];
                [_self.session stopLive];
            }
        }];
    }
    return _startLiveButton;
}


- (CVPixelBufferRef)CVPixelBufferRefFromUiImage:(UIImage *)img {
    CGSize size = img.size;
    CGImageRef image = [img CGImage];
    
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGBitmapContextCompatibilityKey, nil];
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, size.width, size.height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef) options, &pxbuffer);
    
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    NSParameterAssert(pxdata != NULL);
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    
    //CGBitmapInfo的设置
    //uint32_t bitmapInfo = CGImageAlphaInfo | CGBitmapInfo;
    
    //当inputPixelFormat=kCVPixelFormatType_32BGRA CGBitmapInfo的正确的设置
    //uint32_t bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Host;
    //uint32_t bitmapInfo = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Host;
    
    //当inputPixelFormat=kCVPixelFormatType_32ARGB CGBitmapInfo的正确的设置
    //uint32_t bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Big;
    //uint32_t bitmapInfo = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Big;
    
    uint32_t bitmapInfo = bitmapInfoWithPixelFormatType(kCVPixelFormatType_32BGRA);
    
    CGContextRef context = CGBitmapContextCreate(pxdata, size.width, size.height, 8, 4*size.width, rgbColorSpace, bitmapInfo);
    NSParameterAssert(context);
    
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image), CGImageGetHeight(image)), image);
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    
    return pxbuffer;
}

static uint32_t bitmapInfoWithPixelFormatType(OSType inputPixelFormat){
    /*
     CGBitmapInfo的设置
     uint32_t bitmapInfo = CGImageAlphaInfo | CGBitmapInfo;
     
     当inputPixelFormat=kCVPixelFormatType_32BGRA CGBitmapInfo的正确的设置 只有如下两种正确设置
     uint32_t bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Host;
     uint32_t bitmapInfo = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Host;
     
     typedef CF_ENUM(uint32_t, CGImageAlphaInfo) {
     kCGImageAlphaNone,                For example, RGB.
     kCGImageAlphaPremultipliedLast,   For example, premultiplied RGBA
     kCGImageAlphaPremultipliedFirst,  For example, premultiplied ARGB
     kCGImageAlphaLast,                For example, non-premultiplied RGBA
     kCGImageAlphaFirst,               For example, non-premultiplied ARGB
     kCGImageAlphaNoneSkipLast,        For example, RBGX.
     kCGImageAlphaNoneSkipFirst,       For example, XRGB.
     kCGImageAlphaOnly                 No color data, alpha data only
     };
     
     当inputPixelFormat=kCVPixelFormatType_32ARGB CGBitmapInfo的正确的设置 只有如下两种正确设置
     uint32_t bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Big;
     uint32_t bitmapInfo = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Big;
     */
    if (inputPixelFormat == kCVPixelFormatType_32BGRA) {
        //uint32_t bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Host;
        //此格式也可以
        uint32_t bitmapInfo = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Host;
        return bitmapInfo;
    }else if (inputPixelFormat == kCVPixelFormatType_32ARGB){
        uint32_t bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Big;
        //此格式也可以
        //uint32_t bitmapInfo = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Big;
        return bitmapInfo;
    }else{
        NSLog(@"不支持此格式");
        return 0;
    }
}


@end

