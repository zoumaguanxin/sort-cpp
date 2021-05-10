
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/program_options.hpp>


#include "sort.h"
#include "matrix.h"
#include "munkres.h"


using namespace cv;

float CalculateIou(const cv::Rect& det, const Tracker& track) {
    auto trk = track.GetStateAsBbox();
    // get min/max points
    auto xx1 = std::max(det.tl().x, trk.tl().x);
    auto yy1 = std::max(det.tl().y, trk.tl().y);
    auto xx2 = std::min(det.br().x, trk.br().x);
    auto yy2 = std::min(det.br().y, trk.br().y);
    auto w = std::max(0, xx2 - xx1);
    auto h = std::max(0, yy2 - yy1);

    // calculate area of intersection and union
    float det_area = det.area();
    float trk_area = trk.area();
    auto intersection_area = w * h;
    float union_area = det_area + trk_area - intersection_area;
    auto iou = intersection_area / union_area;
    return iou;
}


void HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,
        size_t nrows, size_t ncols,
        std::vector<std::vector<float>>& association) {
    Matrix<float> matrix(nrows, ncols);
    // Initialize matrix with IOU values
    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            // Multiply by -1 to find max cost
            if (iou_matrix[i][j] != 0) {
                matrix(i, j) = -iou_matrix[i][j];
            }
            else {
                // TODO: figure out why we have to assign value to get correct result
                matrix(i, j) = 1.0f;
            }
        }
    }

//    // Display begin matrix state.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(10);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;


    // Apply Kuhn-Munkres algorithm to matrix.
    Munkres<float> m;
    m.solve(matrix);

//    // Display solved matrix.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(2);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            association[i][j] = matrix(i, j);
        }
    }
}


/**
 * Assigns detections to tracked object (both represented as bounding boxes)
 * Returns 2 lists of matches, unmatched_detections
 * @param detection
 * @param tracks
 * @param matched
 * @param unmatched_det
 * @param iou_threshold
 */
void AssociateDetectionsToTrackers(const std::vector<cv::Rect>& detection,
        std::map<int, Tracker>& tracks,
        std::map<int, cv::Rect>& matched,
        std::vector<cv::Rect>& unmatched_det,
        float iou_threshold) {

    // Set all detection as unmatched if no tracks existing
    if (tracks.empty()) {
        for (const auto& det : detection) {
            unmatched_det.push_back(det);
        }
        return;
    }

    std::vector<std::vector<float>> iou_matrix;
    // resize IOU matrix based on number of detection and tracks
    iou_matrix.resize(detection.size(), std::vector<float>(tracks.size()));

    std::vector<std::vector<float>> association;
    // resize association matrix based on number of detection and tracks
    association.resize(detection.size(), std::vector<float>(tracks.size()));


    // row - detection, column - tracks
    for (size_t i = 0; i < detection.size(); i++) {
        size_t j = 0;
        for (const auto& trk : tracks) {
            iou_matrix[i][j] = CalculateIou(detection[i], trk.second);
            j++;
        }
    }

    // Find association
    HungarianMatching(iou_matrix, detection.size(), tracks.size(), association);

    for (size_t i = 0; i < detection.size(); i++) {
        bool matched_flag = false;
        size_t j = 0;
        for (const auto& trk : tracks) {
            if (0 == association[i][j]) {
                // Filter out matched with low IOU
                if (iou_matrix[i][j] >= iou_threshold) {
                    matched[trk.first] = detection[i];
                    matched_flag = true;
                }
                // It builds 1 to 1 association, so we can break from here
                break;
            }
            j++;
        }
        // if detection cannot match with any tracks
        if (!matched_flag) {
            unmatched_det.push_back(detection[i]);
        }
    }
}


Sort::Sort(int max_age ,int min_hits, float iou_threshold){

    _max_age = max_age;
    kMinHits = min_hits;
    _iou_threshold = iou_threshold;

}


std::map<int,cv::Rect> Sort::update(const std::vector<cv::Rect>& detections){

    frame_index += 1;
    //std::cout << "************* NEW FRAME ************* " << std::endl;
    /*** Predict internal tracks from previous frame ***/
    for (auto &track : tracks) {
        track.second.Predict();
    }

    /*** Build association ***/

//     std::cout << "Raw detections:" << std::endl;
//     for (const auto &det : detections) {
//         std::cout << frame_index << "," << "-1" << "," << det.tl().x << "," << det.tl().y
//                     << "," << det.width << "," << det.height << std::endl;
//     }
    
	// Hash-map between track ID and associated detection bounding box
    std::map<int, cv::Rect> matched;
    // vector of unassociated detections
    std::vector<cv::Rect> unmatched_det;

    // return values - matched, unmatched_det
    AssociateDetectionsToTrackers(detections, tracks, matched, unmatched_det,_iou_threshold);

    /*** Update tracks with associated bbox ***/
    for (const auto &match : matched) {
        const auto &ID = match.first;
        tracks[ID].Update(match.second);
    }

    /*** Create new tracks for unmatched detections ***/
    for (const auto &det : unmatched_det) {
        Tracker tracker;
        tracker.Init(det);
        // Create new track and generate new ID
        tracks[current_ID++] = tracker;
    }

    /*** Delete lose tracked tracks ***/
    for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->second.coast_cycles_ > _max_age) {
            it = tracks.erase(it);
        } else {
            it++;
        }
    }

    std::map<int, cv::Rect> results;
    for (auto &trk : tracks) {
        cv::Rect bbox = trk.second.GetStateAsBbox();
        int id = trk.first;
        // Note that we will not export coasted tracks
        // If we export coasted tracks, the total number of false negative will decrease (and maybe ID switch)
        // However, the total number of false positive will increase more (from experiments),
        // which leads to MOTA decrease
        // Developer can export coasted cycles if false negative tracks is critical in the system
        if (trk.second.coast_cycles_ < 1 && (trk.second.hit_streak_ >= kMinHits || frame_index < kMinHits)) {
            // Print to terminal for debugging

//             std::cout << frame_index << "," << trk.first << "," << bbox.tl().x << "," << bbox.tl().y
//                         << "," << bbox.width << "," << bbox.height << ",1,-1,-1,-1"
//                         << " Hit Streak = " << trk.second.hit_streak_
//                         << " Coast Cycles = " << trk.second.coast_cycles_ << std::endl;

            // Export to text file for metrics evaluation

            // output_file << frame_index << "," << trk.first << "," << bbox.tl().x << "," << bbox.tl().y
            //             << "," << bbox.width << "," << bbox.height << ",1,-1,-1,-1\n";
            results[id] = bbox;
            //output_file_NIS << trk.second.GetNIS() << "\n";
        }
    }
    return results;
     


}