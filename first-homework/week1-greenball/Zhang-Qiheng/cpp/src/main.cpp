#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <optional>

#include "json.hpp"
using json = nlohmann::json;

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// 单帧轮廓候选信息
struct BallCandidate
{
    cv::Point2f center;
    double radius = 0.0;
    double area = 0.0;
    double circularity = 0.0;
    double aspect_ratio = 0.0;
    cv::Rect bounding_box;
};

// 跟踪时序状态
struct BallTrackState
{
    cv::Point2f center;
    cv::Point2f velocity;
    double radius = 0.0;
    double speed = 0.0;
    cv::Rect bounding_box;
    int lost_frames = 0;
    bool initialized = false;
};

constexpr int kMaxLostFrames = 8;
constexpr double kBorderMargin = 8.0;

// ===================== 配置结构体 =====================
struct Config
{
    string input_video;
    string output_video;

    Mat camera_matrix;
    Mat dist_coeffs;
    double ball_radius;

    Scalar lower_green;
    Scalar upper_green;
    int morph_kernel_size;
    double min_contour_area;

    double kf_process_noise;
    double kf_measure_noise;
    int max_lost_frame;

    double speed_scale;
    bool show_track_line;
    int track_line_length;

    double margin_w_ratio;
    double margin_h_ratio;
};

// ===================== 卡尔曼滤波 =====================
class KalmanFilter2D
{
public:
    KalmanFilter2D(double dt, double proc_noise, double meas_noise)
    {
        kf.init(4, 2, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1);

        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(proc_noise));
        setIdentity(kf.measurementNoiseCov, Scalar::all(meas_noise));
        setIdentity(kf.errorCovPost, Scalar::all(1000));
    }

    Point2f predict()
    {
        Mat pred = kf.predict();
        return Point2f(pred.at<float>(0), pred.at<float>(1));
    }

    Point2f update(Point2f pt)
    {
        Mat meas = (Mat_<float>(2, 1) << pt.x, pt.y);
        Mat corr = kf.correct(meas);
        return Point2f(corr.at<float>(0), corr.at<float>(1));
    }

private:
    KalmanFilter kf;
};

// 前置声明绘图函数
void drawTrackOverlay(
    cv::Mat & frame, const BallTrackState & current_track, const BallTrackState & previous_track,
    double fps, int frame_index, bool is_predicted);

// ===================== 工具函数 =====================
bool isNearFrameBorder(const cv::Rect& box, const cv::Size& frame_size)
{
    return box.x <= kBorderMargin ||
           box.y <= kBorderMargin ||
           box.x + box.width >= frame_size.width - kBorderMargin ||
           box.y + box.height >= frame_size.height - kBorderMargin;
}

Point2f clampPointToFrame(const cv::Point2f& pt, const cv::Size& frame_size)
{
    double x = std::clamp(static_cast<double>(pt.x), 0.0, static_cast<double>(frame_size.width - 1));
    double y = std::clamp(static_cast<double>(pt.y), 0.0, static_cast<double>(frame_size.height - 1));
    return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

std::vector<BallCandidate> detectBallCandidates(const cv::Mat& frame, const Config& cfg)
{
    std::vector<BallCandidate> candidates;
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cfg.lower_green, cfg.upper_green, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(cfg.morph_kernel_size, cfg.morph_kernel_size));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& cnt : contours)
    {
        double area = cv::contourArea(cnt);
        if (area < cfg.min_contour_area || area > 32000.0)
            continue;

        double perimeter = cv::arcLength(cnt, true);
        if (perimeter <= 1e-6) continue;

        cv::Rect box = cv::boundingRect(cnt);
        double aspect = static_cast<double>(box.width) / box.height;
        double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        bool near_border = isNearFrameBorder(box, frame.size());

        bool pass_filter = false;
        if (!near_border)
        {
            pass_filter = (circularity >= 0.68 && aspect >= 0.85 && aspect <= 1.20);
        }
        else
        {
            pass_filter = (circularity >= 0.45 && aspect >= 0.55 && aspect <= 1.45);
        }
        if (!pass_filter) continue;

        cv::Point2f center;
        float radius_f;
        cv::minEnclosingCircle(cnt, center, radius_f);
        double radius = static_cast<double>(radius_f);
        candidates.push_back({center, radius, area, circularity, aspect, box});
    }
    return candidates;
}

double calcShapeScore(const BallCandidate& cand)
{
    double circle_score = std::clamp((cand.circularity - 0.68) / 0.12, 0.0, 1.0);
    double aspect_score = std::clamp(1.0 - fabs(cand.aspect_ratio - 1.0) / 0.20, 0.0, 1.0);
    double area_score = std::clamp(1.0 - fabs(cand.area - 12000.0) / 10000.0, 0.0, 1.0);
    return 0.45 * circle_score + 0.35 * aspect_score + 0.20 * area_score;
}

std::optional<BallCandidate> selectBestCandidate(
    const std::vector<BallCandidate>& candidates, const BallTrackState& prev_track)
{
    if (candidates.empty()) return std::nullopt;
    double best_score = -1e9;
    const BallCandidate* best = nullptr;

    for (auto& cand : candidates)
    {
        double score = calcShapeScore(cand);
        if (prev_track.initialized)
        {
            cv::Point2f pred_pos = prev_track.center + prev_track.velocity;
            double dist = cv::norm(cand.center - pred_pos);
            double r_val = prev_track.radius * 1.8;
            double s_val = prev_track.speed * 3.5 + 35.0;
            double max_dist = std::max(r_val, s_val);

            if (dist > max_dist) continue;
            double radius_diff = fabs(cand.radius - prev_track.radius);
            if (radius_diff > prev_track.radius * 0.45) continue;
            score -= dist * 0.01;
            score -= radius_diff * 0.02;
        }
        if (score > best_score)
        {
            best_score = score;
            best = &cand;
        }
    }
    if (!best) return std::nullopt;
    return *best;
}

void calcSpeedDir(Point2f prev, Point2f curr, double& speed, double& angle_deg)
{
    if (prev.x < 0 || curr.x < 0)
    {
        speed = 0.0;
        angle_deg = 0.0;
        return;
    }
    double dx = curr.x - prev.x;
    double dy = curr.y - prev.y;
    speed = hypot(dx, dy);
    double rad = atan2(dy, dx);
    angle_deg = rad * 180.0 / CV_PI;
    if (angle_deg < 0) angle_deg += 360.0;
}

// ===================== 【改造点1】新增路径转换工具函数 =====================
// 将相对路径转为基于项目根目录的绝对路径
string rel2abs(const fs::path& root, const string& rel_path)
{
    fs::path p(rel_path);
    // 如果是绝对路径直接返回
    if (p.is_absolute())
        return p.string();
    // 拼接项目根目录
    return (root / p).string();
}

// 加载JSON配置（新增根目录参数，自动转换相对路径）
bool loadConfig(Config& cfg, const string& cfg_path, const fs::path& project_root)
{
    if (!fs::exists(cfg_path))
    {
        cerr << "Config file not found: " << cfg_path << endl;
        return false;
    }
    ifstream f(cfg_path);
    json j;
    f >> j;

    // 读取JSON原始相对路径，自动转为绝对路径
    string raw_in = j["path"]["input_video"];
    string raw_out_vid = j["path"]["output_video"];
    string raw_out_data = j["path"]["output_data"];

    cfg.input_video  = rel2abs(project_root, raw_in);
    cfg.output_video = rel2abs(project_root, raw_out_vid);

    auto cm = j["camera"]["camera_matrix"];
    cfg.camera_matrix = Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            cfg.camera_matrix.at<double>(i, k) = cm[i][k];

    auto dc = j["camera"]["dist_coeffs"];
    cfg.dist_coeffs = Mat(4, 1, CV_64F);
    for (int i = 0; i < 4; i++)
        cfg.dist_coeffs.at<double>(i, 0) = dc[i];
    cfg.ball_radius = j["camera"]["ball_radius"];

    auto lg = j["color_detect"]["lower_green"];
    auto ug = j["color_detect"]["upper_green"];
    cfg.lower_green  = Scalar(lg[0], lg[1], lg[2]);
    cfg.upper_green  = Scalar(ug[0], ug[1], ug[2]);
    cfg.morph_kernel_size = j["color_detect"]["morph_kernel_size"];
    cfg.min_contour_area  = j["color_detect"]["min_contour_area"];

    cfg.kf_process_noise   = j["kalman"]["process_noise"];
    cfg.kf_measure_noise    = j["kalman"]["measurement_noise"];
    cfg.max_lost_frame      = j["kalman"]["max_lost_frame"];

    cfg.speed_scale    = j["visual"]["speed_scale"];
    cfg.show_track_line= j["visual"]["show_track_line"];
    cfg.track_line_length = j["visual"]["track_line_length"];

    cfg.margin_w_ratio = j["roi_margin"]["margin_w_ratio"];
    cfg.margin_h_ratio = j["roi_margin"]["margin_h_ratio"];

    return true;
}

// 绘图实现：识别框、中心点、速度箭头、文字
void drawTrackOverlay(
    cv::Mat & frame, const BallTrackState & current_track, const BallTrackState & previous_track,
    double fps, int frame_index, bool is_predicted)
{
    cv::circle(frame, current_track.center, static_cast<int>(std::round(current_track.radius)),
        cv::Scalar(0, 255, 255), 2);
    cv::circle(frame, current_track.center, 4, cv::Scalar(0, 0, 255), -1);
    // 识别包围框
    cv::rectangle(frame, current_track.bounding_box, Scalar(0, 255, 0), 2);

    const double speed_pixels_per_second = current_track.speed * fps;
    const double previous_speed_pixels_per_second = previous_track.speed * fps;
    const double speed_delta_pixels_per_second =
        speed_pixels_per_second - previous_speed_pixels_per_second;

    cv::Point2f arrow_vector = current_track.velocity;
    if (cv::norm(arrow_vector) < 1.0) {
        arrow_vector = cv::Point2f(30.0F, 0.0F);
    }

    const double arrow_scale = std::clamp(current_track.speed * 4.5, 25.0, 140.0);
    const double velocity_norm = cv::norm(arrow_vector);
    const cv::Point2f arrow_direction = arrow_vector * static_cast<float>(arrow_scale / velocity_norm);
    const cv::Point2f arrow_end = current_track.center + arrow_direction;

    cv::arrowedLine(
        frame, current_track.center, arrow_end,
        cv::Scalar(255, 64, 64), 3, cv::LINE_AA, 0, 0.28);

    const cv::Point info_anchor(
        std::max(20, current_track.bounding_box.x - 20),
        std::max(40, current_track.bounding_box.y - 16));

    string label = is_predicted ? "green ball predicted" : "green ball";
    cv::putText(
        frame, label, info_anchor,
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    ostringstream stream;
    stream << fixed << setprecision(1) << "speed: " << speed_pixels_per_second << " px/s";
    if (fabs(speed_delta_pixels_per_second) < 1e-3) {
        stream << "  stable";
    } else if (speed_delta_pixels_per_second > 0.0) {
        stream << "  accelerating";
    } else {
        stream << "  decelerating";
    }
    cv::putText(
        frame, stream.str(),
        info_anchor + cv::Point(0, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(
        frame, "frame: " + to_string(frame_index), cv::Point(20, 36),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

int main()
{
    // ===================== 【改造点2】基于源码文件推导项目根目录 =====================
    // __FILE__ 是当前main.cpp源码绝对路径
    fs::path src_file = __FILE__;
    // src文件夹路径
    fs::path src_dir = src_file.parent_path();
    // 项目根目录 = src上一级文件夹（你的作业根目录）
    fs::path project_root = src_dir.parent_path();
    // config.json固定在项目根目录
    fs::path cfg_file = project_root / "config.json";
    string cfg_path = cfg_file.string();

    cout << "Project Root Dir: " << project_root << endl;
    cout << "Config full path: " << cfg_path << endl;

    Config cfg;
    // 传入项目根目录用于路径转换
    if (!loadConfig(cfg, cfg_path, project_root))
        return -1;
    cout << "Absolute input video path: " << cfg.input_video << endl;

    // 自动创建输出视频文件夹
    fs::create_directories(fs::path(cfg.output_video).parent_path());
    // CSV输出路径固定在项目根/output_data
    fs::path csv_dir = project_root / "output_data";
    fs::create_directories(csv_dir);
    string csv_path = (csv_dir / "frame_speed.csv").string();

    VideoCapture cap(cfg.input_video, CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cerr << "Open video FAILED! Path: " << cfg.input_video << endl;
        cap.open(cfg.input_video);
        if (!cap.isOpened())
        {
            cerr << "All methods failed to open video!" << endl;
            return -1;
        }
    }
    cout << "Open video SUCCESS" << endl;

    double fps = cap.get(CAP_PROP_FPS);
    int w = cap.get(CAP_PROP_FRAME_WIDTH);
    int h = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "FPS: " << fps << ", W: " << w << ", H: " << h << endl;

    VideoWriter out_video(cfg.output_video, VideoWriter::fourcc('m','p','4','v'), fps, Size(w,h));
    KalmanFilter2D kf(1.0/fps, cfg.kf_process_noise, cfg.kf_measure_noise);

    ofstream csv_f(csv_path, ios::out);
    csv_f << "frame_idx,time_s,speed_pixel,direction_deg\n";

    int frame_idx = 0;
    BallTrackState prev_track;
    Mat frame;

    while (cap.read(frame))
    {
        double time_s = static_cast<double>(frame_idx) / fps;
        auto candidates = detectBallCandidates(frame, cfg);
        auto best_cand = selectBestCandidate(candidates, prev_track);

        BallTrackState curr_track;
        bool use_prediction = false;
        double curr_speed = 0.0;
        double curr_angle = 0.0;
        Point2f curr_pt(-1,-1);
        Point2f prev_pt(-1,-1);

        if (best_cand.has_value())
        {
            auto& cand = best_cand.value();
            curr_track.center = cand.center;
            curr_track.radius = cand.radius;
            curr_track.bounding_box = cand.bounding_box;
            curr_track.lost_frames = 0;
            curr_track.initialized = true;

            curr_pt = cand.center;
            if (prev_track.initialized)
            {
                prev_pt = prev_track.center;
                curr_track.velocity = curr_track.center - prev_track.center;
                curr_track.speed = cv::norm(curr_track.velocity);
            }
            else
            {
                curr_track.velocity = cv::Point2f(0,0);
                curr_track.speed = 0.0;
            }
            drawTrackOverlay(frame, curr_track, prev_track, fps, frame_idx, false);
            prev_track = curr_track;
        }
        else if (prev_track.initialized && prev_track.lost_frames < kMaxLostFrames)
        {
            BallTrackState pred_track = prev_track;
            pred_track.lost_frames += 1;
            pred_track.center = clampPointToFrame(prev_track.center + prev_track.velocity, frame.size());
            pred_track.bounding_box.x = static_cast<int>(std::round(pred_track.center.x - pred_track.radius));
            pred_track.bounding_box.y = static_cast<int>(std::round(pred_track.center.y - pred_track.radius));
            pred_track.bounding_box.width = static_cast<int>(std::round(pred_track.radius * 2.0));
            pred_track.bounding_box.height = static_cast<int>(std::round(pred_track.radius * 2.0));
            use_prediction = true;
            drawTrackOverlay(frame, pred_track, prev_track, fps, frame_idx, true);
            prev_track = pred_track;
            curr_pt = pred_track.center;
            prev_pt = prev_track.center - prev_track.velocity;
        }
        else
        {
            cv::putText(frame, "green ball lost", cv::Point(20,36), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
            prev_track = BallTrackState{};
        }

        // CSV 写入速度、角度
        calcSpeedDir(prev_pt, curr_pt, curr_speed, curr_angle);
        csv_f << frame_idx << ","
              << fixed << setprecision(3) << time_s << ","
              << fixed << setprecision(2) << curr_speed << ","
              << fixed << setprecision(2) << curr_angle << "\n";

        out_video << frame;
        imshow("track", frame);
        if (waitKey(1) == 'q') break;

        frame_idx++;
    }

    cap.release();
    out_video.release();
    csv_f.close();
    destroyAllWindows();

    cout << "Track Finished!" << endl;
    cout << "Video output: " << cfg.output_video << endl;
    cout << "CSV data: " << csv_path << endl;
    return 0;
}
