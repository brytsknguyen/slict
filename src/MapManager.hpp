#include "utility.h"

class MapManager
{

private:

    // Node handle
    ros::NodeHandlePtr nh_ptr;

    // Publishing the prior map for visualization
    CloudXYZITPtr  pmSurfGlobal;
    CloudXYZITPtr  pmEdgeGlobal;
    CloudXYZITPtr  priorMap;
    ros::Publisher priorMapPub;
    
    using ufoSurfelMap = ufo::map::SurfelMap;
    ufoSurfelMap surfelMapSurf;
    ufoSurfelMap surfelMapEdge;

public:

   ~MapManager();
    MapManager(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        pmSurfGlobal = CloudXYZITPtr(new CloudXYZIT());
        pmEdgeGlobal = CloudXYZITPtr(new CloudXYZIT());
        priorMap     = CloudXYZITPtr(new CloudXYZIT());

        // Initializing priormap
        priorMapPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap", 10);

        string prior_map_dir = "";
        nh_ptr->param("/prior_map_dir", prior_map_dir, string(""));

        // Read the pose log of priormap
        string pmPose_ = prior_map_dir + "/kfpose6d.pcd";
        CloudPosePtr pmPose(new CloudPose());
        pcl::io::loadPCDFile<PointPose>(pmPose_, *pmPose);

        int PM_KF_COUNT = pmPose->size();
        printf("Prior map path %s. Num scans: %d\n", pmPose_.c_str(), pmPose->size());

        // Reading the surf feature from log
        deque<CloudXYZITPtr> pmSurf(PM_KF_COUNT);
        deque<CloudXYZITPtr> pmEdge(PM_KF_COUNT);
        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < PM_KF_COUNT; i++)
        {
            pmSurf[i] = CloudXYZITPtr(new CloudXYZIT());
            string pmSurf_ = prior_map_dir + "/pointclouds/" + "KfSurfPcl_" + to_string(i) + ".pcd";
            pcl::io::loadPCDFile<PointXYZIT>(pmSurf_, *pmSurf[i]);

            pmEdge[i] = CloudXYZITPtr(new CloudXYZIT());
            string pmEdge_ = prior_map_dir + "/pointclouds/" + "KfEdgePcl_" + to_string(i) + ".pcd";
            pcl::io::loadPCDFile<PointXYZIT>(pmEdge_, *pmEdge[i]);

            // printf("Reading scan %d.\n", i);
        }

        printf("Merging the scans.\n");

        // Merge the scans
        for (int i = 0; i < PM_KF_COUNT; i++)
        {
            *pmSurfGlobal += *pmSurf[i];
            *pmEdgeGlobal += *pmEdge[i];
        }

        // Publish the prior map for visualization
        *priorMap = *pmSurfGlobal + *pmEdgeGlobal;
        {
            pcl::UniformSampling<PointXYZIT> downsampler;
            downsampler.setRadiusSearch(0.1);
            downsampler.setInputCloud(priorMap);
            downsampler.filter(*priorMap);
        }

        Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "world");

        printf(KYEL "Surfelizing the scans.\n" RESET);

        double leaf_size = 0.1; // To-do: change this to param
        // Surfelize the surf map
        surfelMapSurf = ufoSurfelMap(leaf_size);
        insertCloudToSurfelMap(surfelMapSurf, *pmSurfGlobal);
        // Surfelize the edge map
        surfelMapEdge = ufoSurfelMap(leaf_size);
        insertCloudToSurfelMap(surfelMapEdge, *pmEdgeGlobal);

        printf(KGRN "Done. Surfmap: %d. Edgemap: %d\n" RESET, surfelMapSurf.size(), surfelMapEdge.size());
    }
};
