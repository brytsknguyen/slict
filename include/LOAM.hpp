#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utility.h"

/* All needed for filter of custom point type----------*/
#include <pcl/pcl_base.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/impl/crop_box.hpp>
/* All needed for filter of custom point type----------*/

// All about gaussian process
#include "GaussianProcess.hpp"


using NodeHandlePtr = boost::shared_ptr<ros::NodeHandle>;
class LOAM
{
private:

    NodeHandlePtr nh_ptr;
    
    // Index for distinguishing between clouds
    int LIDX;

    // Feature to map association parameters
    double min_planarity = 0.2;
    double max_plane_dis = 0.3;

    // Initial pose of the lidars
    SE3d T_W_Li0;

    // Gaussian Process for the trajectory of each lidar
    GaussianProcessPtr traj;

    // Knot length
    double deltaT = 0.1;
    double mpSigGa = 10;
    double mpSigNu = 10;
    
    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;

    // Buffer for the pointcloud segments
    mutex cloud_seg_buf_mtx;
    deque<CloudXYZITPtr> cloud_seg_buf;

    // Publisher
    ros::Publisher trajPub;
    ros::Publisher swTrajPub;
    ros::Publisher assocCloudPub;
    ros::Publisher deskewedCloudPub;

public:

    // Destructor
   ~LOAM() {};

    // LOAM(NodeHandlePtr &nh_ptr_, mutex & nh_mtx, const SE3d &T_W_Li0_, double t0, int &LIDX_)
    LOAM(const SE3d &T_W_Li0_, double t0, int &LIDX_)
        : T_W_Li0(T_W_Li0_), LIDX(LIDX_)
    {
        // lock_guard<mutex> lg(nh_mtx);

        // // Trajectory estimate
        // nh_ptr->getParam("deltaT", deltaT);

        // // Weight for the motion prior
        // nh_ptr->getParam("mpSigGa", mpSigGa);
        // nh_ptr->getParam("mpSigNu", mpSigNu);

        // // Association params
        // nh_ptr->getParam("min_planarity", min_planarity);
        // nh_ptr->getParam("max_plane_dis", max_plane_dis);
        // nh_ptr->getParam("knnSize", knnSize);

        // trajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/gp_traj", LIDX), 1);
        // swTrajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
        // assocCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
        // deskewedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/cloud_inW", LIDX), 1);

        Matrix3d SigGa = Vector3d(mpSigGa, mpSigGa, mpSigGa).asDiagonal();
        Matrix3d SigNu = Vector3d(mpSigNu, mpSigNu, mpSigNu).asDiagonal();

        traj = GaussianProcessPtr(new GaussianProcess(deltaT, SigGa, SigNu, true));
        traj->setStartTime(t0);
        traj->setKnot(0, GPState(t0, T_W_Li0));
    }

    void Associate(GaussianProcessPtr &traj, const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZITPtr &cloudRaw, const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   vector<LidarCoef> &Coef)
    {
        ROS_ASSERT_MSG(cloudRaw->size() == cloudInB->size(),
                       "cloudRaw: %d. cloudInB: %d", cloudRaw->size(), cloudInB->size());

        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                double tpoint = cloudRaw->points[pidx].t;
                PointXYZIT pointRaw = cloudRaw->points[pidx];
                PointXYZI  pointInB = cloudInB->points[pidx];
                PointXYZI  pointInW = cloudInW->points[pidx];

                Coef_[pidx].n = Vector4d(0, 0, 0, 0);
                Coef_[pidx].t = -1;

                if(!Util::PointIsValid(pointInB))
                {
                    // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                    pointInB.x = 0; pointInB.y = 0; pointInB.z = 0; pointInB.intensity = 0;
                    continue;
                }

                if(!Util::PointIsValid(pointInW))
                    continue;

                if (!traj->TimeInInterval(tpoint, 1e-6))
                    continue;

                vector<int> knn_idx(knnSize, 0); vector<float> knn_sq_dis(knnSize, 0);
                kdtreeMap->nearestKSearch(pointInW, knnSize, knn_idx, knn_sq_dis);

                vector<PointXYZI> nbrPoints;
                if (knn_sq_dis.back() < minKnnSqDis)
                    for(auto &idx : knn_idx)
                        nbrPoints.push_back(priormap->points[idx]);
                else
                    continue;

                // Fit the plane
                if(Util::fitPlane(nbrPoints, min_planarity, max_plane_dis, Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    // ROS_ASSERT(tpoint >= 0);
                    Coef_[pidx].t = tpoint;
                    Coef_[pidx].f = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
                    Coef_[pidx].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
                    Coef_[pidx].fdsk = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                }
            }

            // Copy the coefficients to the buffer
            Coef.clear();
            int totalFeature = 0;
            for(int pidx = 0; pidx < pointsCount; pidx++)
            {
                LidarCoef &coef = Coef_[pidx];
                if (coef.t >= 0)
                {
                    Coef.push_back(coef);
                    Coef.back().ptIdx = totalFeature;
                    totalFeature++;
                }
            }
        }
    }

    void Deskew(GaussianProcessPtr &traj, CloudXYZITPtr &cloudRaw, CloudXYZIPtr &cloudDeskewedInB)
    {
        int Npoints = cloudRaw->size();

        if (Npoints == 0)
            return;

        cloudDeskewedInB->resize(Npoints);

        SE3d T_Be_W = traj->pose(cloudRaw->points.back().t).inverse();
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int pidx = 0; pidx < Npoints; pidx++)
        {
            PointXYZIT &pi = cloudRaw->points[pidx];
            PointXYZI  &po = cloudDeskewedInB->points[pidx];

            double ts = pi.t;
            SE3d T_Be_Bs = T_Be_W*traj->pose(ts);

            Vector3d pinBs(pi.x, pi.y, pi.z);
            Vector3d pinBe = T_Be_Bs*pinBs;

            po.x = pinBe.x();
            po.y = pinBe.y();
            po.z = pinBe.z();
            // po.t = pi.t;
            po.intensity = pi.intensity;
        }
    }

    // void Visualize(double tmin, double tmax, deque<vector<LidarCoef>> &swCloudCoef, CloudXYZIPtr &cloudUndiInW, bool publish_full_traj=false)
    // {
    //     if (publish_full_traj)
    //     {
    //         CloudPosePtr trajCP = CloudPosePtr(new CloudPose());
    //         for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
    //         {
    //             trajCP->points.push_back(myTf(traj->getKnotPose(kidx)).Pose6D(traj->getKnotTime(kidx)));
    //             trajCP->points.back().intensity = (tmax - trajCP->points.back().t) < 0.1 ? 1.0 : 0.0;
    //         }

    //         // Publish global trajectory
    //         Util::publishCloud(trajPub, *trajCP, ros::Time::now(), "world");
    //     }

    //     // Sample and publish the slinding window trajectory
    //     CloudPosePtr poseSampled = CloudPosePtr(new CloudPose());
    //     for(double ts = tmin; ts < tmax; ts += traj->getDt()/5)
    //         if(traj->TimeInInterval(ts))
    //             poseSampled->points.push_back(myTf(traj->pose(ts)).Pose6D(ts));

    //     // static ros::Publisher swTrajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
    //     Util::publishCloud(swTrajPub, *poseSampled, ros::Time::now(), "world");

    //     CloudXYZIPtr assoc_cloud(new CloudXYZI());
    //     for (int widx = 0; widx < swCloudCoef.size(); widx++)
    //     {
    //         for(auto &coef : swCloudCoef[widx])
    //             {
    //                 PointXYZI p;
    //                 p.x = coef.finW.x();
    //                 p.y = coef.finW.y();
    //                 p.z = coef.finW.z();
    //                 p.intensity = widx;
    //                 assoc_cloud->push_back(p);
    //             }
    //     }
        
    //     // static ros::Publisher assocCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
    //     if (assoc_cloud->size() != 0)
    //         Util::publishCloud(assocCloudPub, *assoc_cloud, ros::Time::now(), "world");

    //     // Publish the deskewed pointCloud
    //     Util::publishCloud(deskewedCloudPub, *cloudUndiInW, ros::Time::now(), "world");
    // }

    GaussianProcessPtr &GetTraj()
    {
        return traj;
    }

};
