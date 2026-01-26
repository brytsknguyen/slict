#ifndef POINT_TO_MAP_ASSOC_H
#define POINT_TO_MAP_ASSOC_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Utils
#include "utility.h"

// Association
// #include "PointToMapAssoc.h"

// Shorthands for ufomap
namespace ufopred     = ufo::map::predicate;
using ufoSurfelMap    = ufo::map::SurfelMap;
using ufoSurfelMapPtr = std::shared_ptr<ufoSurfelMap>;
using ufoNode         = ufo::map::NodeBV;
using ufoSphere       = ufo::geometry::Sphere;
using ufoPoint3       = ufo::map::Point3;

class PointToMapAssoc
{
public:

    // Constructor
    PointToMapAssoc(RosNodeHandlePtr &nh_);

    // Destructor
   ~PointToMapAssoc();

   //Associate function
   void AssociatePointWithMap(PointXYZIT &fRaw, Vector3d &finB, Vector3d &finW, ufoSurfelMap &Map, LidarCoef &coef);

private:

    // Node handler
    RosNodeHandlePtr nh;

    // Surfel params
    int    surfel_map_depth;
    int    surfel_min_point;
    int    surfel_min_depth;
    int    surfel_query_depth;
    double surfel_intsect_rad;
    double surfel_min_plnrty;
    double dis_to_surfel_max;
    double score_min;

};

#endif