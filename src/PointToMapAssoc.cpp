#include "PointToMapAssoc.h"

// Construct the object
PointToMapAssoc::PointToMapAssoc(ros::NodeHandlePtr &nh_) : nh(nh_)
{
    nh->getParam("/surfel_map_depth",   surfel_map_depth);
    nh->getParam("/surfel_min_point",   surfel_min_point);
    nh->getParam("/surfel_min_depth",   surfel_min_depth);
    nh->getParam("/surfel_query_depth", surfel_query_depth);
    nh->getParam("/surfel_intsect_rad", surfel_intsect_rad);
    nh->getParam("/surfel_min_plnrty",  surfel_min_plnrty);

    nh->getParam("/dis_to_surfel_max",  dis_to_surfel_max);
    nh->getParam("/score_min",          score_min);
}

// Destructor
PointToMapAssoc::~PointToMapAssoc(){};

// Associate the points
void PointToMapAssoc::AssociatePointWithMap(PointXYZIT &fRaw, Vector3d &finB, Vector3d &finW, ufoSurfelMap &Map, LidarCoef &coef)
{
    static auto commonPred = ufopred::HasSurfel()
                          && ufopred::DepthMin(surfel_min_depth)
                          && ufopred::DepthMax(surfel_query_depth - 1)
                          && ufopred::NumSurfelPointsMin(surfel_min_point);
                        //   && ufopred::SurfelPlanarityMin(surfel_min_plnrty);
    auto pred = commonPred
             && ufopred::Intersects(ufoSphere(ufoPoint3(finW(0), finW(1), finW(2)), surfel_intsect_rad));

    // Reset the flag for association
    coef.t = -1;

    vector<ufoNode> mapNodes;
    for (const ufoNode &node : Map.queryBV(pred))
        mapNodes.push_back(node);

    struct ClosestSurf
    {
        uint32_t depth = -1;
        uint32_t count = -1;
        double   d2pln = -1;
        double   plnrt = -1;
        Vector3d eigen = Vector3d(0, 0, 0);
        Vector4d plane = Vector4d(0, 0, 0, 0);
    };
    vector<ClosestSurf> closestSurf(surfel_query_depth);

    // If there is associable node, evaluate further
    for (const ufoNode &node : mapNodes)
    {
        auto const& surfel = Map.getSurfel(node);

        int depth = node.depth();
        int numPoint = surfel.getNumPoints();

        std::array<double, 6> cov = surfel.getSymmetricCovariance();
        double planarity = 0;
        Vector3d norm(0, 0, 0);

        auto AccessCov = [](const std::array<double, 6> &cov, double &planarity, Vector3d &norm) -> void
        {
            double a = cov[0]; double b = cov[3]; double c = cov[5]; double d = cov[1]; double e = cov[4]; double f = cov[2];

            if(isnan(a) || isnan(b) || isnan(c) || isnan(d) || isnan(e) || isnan(f))
            {
                printf("Node has NaN Cov");
                planarity = 0;
                norm = Vector3d(0, 0, 0);
            }

            if(isnan(a) || isnan(b) || isnan(c) || isnan(d) || isnan(e) || isnan(f))
            {
                printf("Node has inf Cov");
                planarity = 0;
                norm = Vector3d(0, 0, 0);
            }

            ROS_ASSERT_MSG(!isnan(a) && !isnan(b) && !isnan(c) && !isnan(d) && !isnan(e) && !isnan(f)
                           && isfinite(a) && isfinite(b) && isfinite(c) && isfinite(d) && isfinite(e) && isfinite(f),
                           "%f, %f, %f, %f, %f, %f\n", a, b, c, d, e, f);

            double const x_1 = a * a + b * b + c * c - a * b - a * c - b * c + 3 * (d * d + f * f + e * e);

            double const x_2 = -(2 * a - b - c) * (2 * b - a - c) * (2 * c - a - b)
                               + 9 * ((2 * c - a - b) * (d * d) + (2 * b - a - c) * (f * f) + (2 * a - b - c) * (e * e))
                               - 54 * (d * e * f);

            double const phi = 0 < x_2 ? std::atan(std::sqrt(4 * x_1 * x_1 * x_1 - x_2 * x_2) / x_2)
                                       : (0 > x_2 ? std::atan(std::sqrt(4 * x_1 * x_1 * x_1 - x_2 * x_2) / x_2) + M_PI
                                                  : M_PI_2);
            assert(!isnan(phi) && isfinite(phi));

            // Find egein values
            double l_1 = (a + b + c - 2 * std::sqrt(x_1) * std::cos(phi / 3)) / 3;
            double l_2 = (a + b + c + 2 * std::sqrt(x_1) * std::cos((phi + M_PI) / 3)) / 3;
            double l_3 = (a + b + c + 2 * std::sqrt(x_1) * std::cos((phi - M_PI) / 3)) / 3;

            f = 0 == f ? std::numeric_limits<float>::epsilon() : f;
            double const m_1 = (d * (c - l_1) - e * f) / (f * (b - l_1) - d * e);            
            planarity = 2 * (l_2 - l_1) / (l_1 + l_2 + l_3);
            norm = Vector3d((l_1 - c - e * m_1) / f, m_1, 1);
            norm.normalize();
        };

        AccessCov(cov, planarity, norm);

        // double planarity = surfel.getPlanarity();
        // Vector3d norm = ufo::math::toEigen(surfel.getNormal());
        Vector3d mean = ufo::math::toEigen(surfel.getMean());

        if(    isnan(mean(0)) || isnan(mean(1)) || isnan(mean(2))
            || isnan(norm(0)) || isnan(norm(1)) || isnan(norm(2))
            || planarity < surfel_min_plnrty || planarity > 1.0
          )
        {       
            // printf(KYEL "Node with anomaly surf: %d. m: %f, %f, %f. n: %f, %f, %f. rho: %f\n" RESET,
            //        node, mean(0), mean(1), mean(2), norm(0), norm(1), norm(2), planarity);
            continue;
        }

        // ROS_ASSERT_MSG(planarity >= 0 && planarity <= 1.0, "plnrty: %f\n", planarity);
        double d2pln = fabs(norm.dot(finW - mean));

        if (closestSurf[depth].d2pln == -1 || d2pln < closestSurf[depth].d2pln)
        {
            closestSurf[depth].depth = depth;
            closestSurf[depth].count = numPoint;
            closestSurf[depth].d2pln = d2pln;
            closestSurf[depth].plnrt = planarity;
            // closestSurf[depth].eigen = eig;
            closestSurf[depth].plane << norm, -norm.dot(mean);
        }
    }

    bool point_associated = false;
    for (int depth = 0; depth < surfel_query_depth; depth++)
    {
        // Write down the d2p for the original point
        // if (depth == surfel_min_depth)
        //     CloudCoef[coeff_idx + depth].d2P = closestSurf[depth].d2pln;

        if (closestSurf[depth].d2pln > dis_to_surfel_max || closestSurf[depth].d2pln == -1 || point_associated)
            continue;

        double score = (1 - 0.9 * closestSurf[depth].d2pln / finB.norm())*closestSurf[depth].plnrt;
        // double score = closestSurf[depth].plnrt;

        // Weightage based on how close the point is to the plane
        if (score > score_min)
        {
            // LidarCoef &coef = CloudCoef[coeff_idx];

            coef.t      = fRaw.t;
            // coef.ptIdx  = point_idx;
            coef.n      = closestSurf[depth].plane;
            coef.scale  = depth;
            coef.surfNp = closestSurf[depth].count;
            coef.plnrty = score;
            coef.d2P    = closestSurf[depth].d2pln;
            coef.f      = Vector3d(fRaw.x, fRaw.y, fRaw.z);
            coef.fdsk   = finB;
            coef.finW   = finW;

            // Set the spline coefficients
            // auto us = traj.computeTIndex(coef.t);
            // coef.dt = traj.getDt();
            // coef.u  = us.first;
            // coef.s  = us.second;

            point_associated = true;
        }
    }
}