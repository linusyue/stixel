#include <iostream>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "semi_global_matching.h"
#include "stixel_world.h"

#include <dirent.h>

using namespace cv;
using namespace std;

void usage(char* argv[])
{
    //Folder struct
    //-directory/
    //     --image2/
    //     --image3/
    std::cout << "usage: " << argv[0] << "stixel [directory] [camera in/ex para]" << std::endl;

    cout<<"Foler structure"<<endl;
    cout<<"-[directory]/"<<endl;
    cout<<"      --image_2/"<<endl;
    cout<<"      --image_3/"<<endl;
}

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        usage(argv);
        return -1;
    }

    string directory = argv[1];

    DIR *dp, *dp1;
    struct dirent *ep;

    string image2_dir;
    image2_dir = directory + "/" + "image_2";
    dp = opendir(image2_dir.c_str());

    string image3_dir;
    image3_dir = directory + "/" + "image_3";
    dp1 = opendir(image3_dir.c_str());

    if (dp == NULL || dp1 ==  NULL) {
        std::cerr << "Invalid folder structure under: " << directory << std::endl;
        usage(argv);
        exit(EXIT_FAILURE);
    }

    string I0_path;
    string I1_path;

    SemiGlobalMatching::Parameters param;
    SemiGlobalMatching sgm(param);
    cv::Mat D0, D1, draw;

    while ((ep = readdir(dp)) != NULL) 
    {
        // Skip directories
        if (!strcmp (ep->d_name, "."))
            continue;
        if (!strcmp (ep->d_name, ".."))
            continue;

        //        string postfix = "_10.png";
        //        string::size_type idx;

        string image_name = ep->d_name;

#if 0
        //Only _10 has groundtruth
        idx = image_name.find(postfix);

        if(idx == string::npos )
            continue;  
#endif

        I0_path = directory + "/" + "image_2" + "/" + image_name;
        I1_path = directory + "/" + "image_3" + "/" + image_name;

        cout<<"I0: "<<I0_path<<endl;
        cout<<"I1: "<<I1_path<<endl;

        Mat I0 = imread(I0_path);
        Mat I1 = imread(I1_path);

        if (I0.empty() || I1.empty())
        {
            std::cerr << "failed to read any image." << std::endl;
            break;
        }

        CV_Assert(I0.size() == I1.size() && I0.type() == I1.type());

        //convert to gray
        Mat I0_Gray, I1_Gray;
        cvtColor(I0, I0_Gray, cv::COLOR_BGR2GRAY);
        cvtColor(I1, I1_Gray, cv::COLOR_BGR2GRAY);

        imshow("I0", I0);
        imshow("I1", I1);

        const auto t1 = std::chrono::system_clock::now();

        sgm.compute(I0_Gray, I1_Gray, D0, D1);

        const auto t2 = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;

        D0.convertTo(draw, CV_8U, 255. / (SemiGlobalMatching::DISP_SCALE * param.numDisparities));
        cv::applyColorMap(draw, draw, cv::COLORMAP_JET);
        draw.setTo(0, D0 == SemiGlobalMatching::DISP_INV);

        cv::imshow("disparity", draw);

        Mat fdisp;

        D0.convertTo(fdisp, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
        //        Mat D0_16u(D0.size(), CV_16U);

        // calculate stixels
        std::vector<Stixel> stixels;
        const auto t3 = std::chrono::system_clock::now();

        StixelWorld::Parameters stix_param;

	    const cv::FileStorage cvfs(argv[2], FileStorage::READ);

        bool opened = cvfs.isOpened();
	    
//        const cv::FileNode node = cvfs.getFirstTopLevelNode();
        const cv::FileNode node = cvfs.root();
      
        stix_param.camera.fu = node["FocalLengthX"];
        stix_param.camera.fv = node["FocalLengthY"];
        stix_param.camera.u0 = node["CenterX"];
        stix_param.camera.v0 = node["CenterY"];
        stix_param.camera.baseline = node["BaseLine"];
        stix_param.camera.height = node["Height"];
        stix_param.camera.tilt = node["Tilt"];
        stix_param.minDisparity = -1;
        stix_param.maxDisparity = param.numDisparities;

        StixelWorld stixelWorld(stix_param);

        stixelWorld.compute(fdisp, stixels);
        const auto t4 = std::chrono::system_clock::now();
        const auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        std::cout << "stixel computation time: " << 1e-3 * duration << "[msec]" << std::endl;

        // draw stixels
        cv::Mat showStixel = I0.clone();
        cv::Mat stixelImg = cv::Mat::zeros(I0.size(), showStixel.type());

        for (const auto& stixel : stixels)
            drawStixel(stixelImg, stixel, dispToColor(stixel.disp, (float)param.numDisparities));

		showStixel = showStixel + 0.5 * stixelImg;
		cv::imshow("stixels", showStixel);

        waitKey();
    }

    return 0; 
}
