#pragma once

#include "kalman_filter.h"
#include "tracker.h"


void AssociateDetectionsToTrackers(const std::vector<cv::Rect>& detection,
        std::map<int, Tracker>& tracks,
        std::map<int, cv::Rect>& matched,
        std::vector<cv::Rect>& unmatched_det,
        float iou_threshold = 0.3);


void HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,   size_t nrows, size_t ncols, std::vector<std::vector<float>>& association);

class Sort{

public:
    Sort(){}
    Sort(int max_age=30,int min_hits=10, float iou_threshold=0.3);
    std::map<int,cv::Rect> update(const std::vector<cv::Rect>& detections);

private:
    // Hash-map between ID and corresponding tracker
    std::map<int, Tracker> tracks;
    // Assigned ID for each bounding box
    int current_ID = 0;
    int kMinHits = 3;
    int _max_age = 1;
    float _iou_threshold = 0.3;
    int frame_index = 0;
};