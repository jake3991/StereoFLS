#!/usr/bin/env python

import sys

if sys.version_info >= (3, 5):
    import typing

import cv2
import cv_bridge
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.interpolate import interp1d
from sensor_msgs.msg import PointCloud2, PointField, Image
from sklearn.utils import shuffle
from sonar_oculus.msg import OculusPing
from std_msgs.msg import Header

from stereo_sonar.CFAR import *
from stereo_sonar.match import matchFeatures as match_features_cpp

class stereoSonar:
    """A class to handle the multi sonar array"""

    def __init__(self, ns="~"):
        # type: (str) -> None

        # define vertical transformation between sonars (meters)
        self.transformation = 0.1

        # define the implementation we want used
        self.method = rospy.get_param(ns + "method")

        #are we going to publish the feature images?
        self.vis_features = rospy.get_param(ns + "visFeatures")

        # define the max uncertainty for a match
        self.uncertaintyMax = rospy.get_param(ns + "uncertaintyMax")
        self.patchSize = rospy.get_param(ns + "patchSize")

        # define the footprint of the sonars, note we assume that theses parameters
        # are SHARED across sonars (the same)
        self.horizontalFOV = rospy.get_param(ns + "horizontalFOV")
        self.verticalAperture = rospy.get_param(ns + "verticalAperture")

        # horizontal sonar info
        self.maxRange_horizontal = 30.0  # default value, reads in new value from msg

        # vertical sonar info
        self.maxRange_vertical = 30.0  # default value, reads in new value from msg

        # CFAR parameters for horizontal sonar
        self.tcHorizontal = rospy.get_param(ns + "tcHorizontal")
        self.gcHorizontal = rospy.get_param(ns + "gcHorizontal")
        self.pfaHorizontal = rospy.get_param(ns + "pfaHorizontal")
        self.thresholdHorizontal = rospy.get_param(ns + "thresholdHorizontal")

        # CFAR parameters for vertical sonar
        self.tcVertical = rospy.get_param(ns + "tcVertical")
        self.gcVertical = rospy.get_param(ns + "gcVertical")
        self.pfaVertical = rospy.get_param(ns + "pfaVertical")
        self.thresholdVertical = rospy.get_param(ns + "thresholdVertical")

        # define the CFAR detectors
        self.detector_horizontal = CFAR(
            self.tcHorizontal, self.gcHorizontal, self.pfaHorizontal, None
        )
        self.detector_vertical = CFAR(
            self.tcVertical, self.gcVertical, self.pfaVertical, None
        )

        # define image subsciber
        self.verticalSonarSub = Subscriber(
            rospy.get_param(ns + "verticalTopic"), OculusPing
        )
        self.horizontalSonarSub = Subscriber(
            rospy.get_param(ns + "horizontalTopic"), OculusPing
        )

        # define time sync object for both sonar images
        self.timeSync = ApproximateTimeSynchronizer(
            [self.verticalSonarSub, self.horizontalSonarSub], 1, 0.5
        )

        # register callback
        self.timeSync.registerCallback(self.callback)

        # define fused point cloud publisher
        self.cloudPublisher = rospy.Publisher("SonarCloud", PointCloud2, queue_size=10)

        # define cvbridge instance
        self.CVbridge = cv_bridge.CvBridge()

        # define laser fields for fused point cloud
        self.laserFields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # for remapping from polar to cartisian
        self.res = None
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None
        self.map_x = None
        self.map_y = None
        self.f_bearings = None
        self.to_rad = lambda bearing: bearing * np.pi / 18000
        self.REVERSE_Z = 1

        self.imagePub = rospy.Publisher("hori_features",Image,queue_size = 5)
        self.imagePub_2 = rospy.Publisher("vert_features",Image,queue_size = 5)

        self.id = 0

    def generate_map_xy(self, ping):
        # type: (OculusPing) -> None
        """Generate a mesh grid map for the sonar image.

        Keyword Parameters:
        ping -- a OculusPing message
        """

        # get the parameters from the ping message
        _res = ping.range_resolution
        _height = ping.num_ranges * _res
        _rows = ping.num_ranges
        _width = (
            np.sin(self.to_rad(ping.bearings[-1] - ping.bearings[0]) / 2) * _height * 2
        )
        _cols = int(np.ceil(_width / _res))

        # check if the parameters have changed
        if (
            self.res == _res
            and self.height == _height
            and self.rows == _rows
            and self.width == _width
            and self.cols == _cols
        ):
            return

        # if they have changed do some work
        self.res, self.height, self.rows, self.width, self.cols = (
            _res,
            _height,
            _rows,
            _width,
            _cols,
        )

        # generate the mapping
        bearings = self.to_rad(np.asarray(ping.bearings, dtype=np.float32))
        f_bearings = interp1d(
            bearings,
            range(len(bearings)),
            kind="linear",
            bounds_error=False,
            fill_value=-1,
            assume_sorted=True,
        )

        # build the meshgrid
        XX, YY = np.meshgrid(range(self.cols), range(self.rows))
        x = self.res * (self.rows - YY)
        y = self.res * (-self.cols / 2.0 + XX + 0.5)
        b = np.arctan2(y, x) * self.REVERSE_Z
        r = np.sqrt(np.square(x) + np.square(y))
        self.map_y = np.asarray(r / self.res, dtype=np.float32)
        self.map_x = np.asarray(f_bearings(b), dtype=np.float32)

        # check for change in max range
        if self.maxRange_horizontal != self.height:
            self.maxRange_horizontal = self.height
            self.maxRange_vertical = self.height

    def img2Real(self, points, sonar):
        # type: (np.ndarray, str) -> typing.Tuple[np.ndarray, np.ndarray, float, float, float, float]
        """Accepts CFAR points and sonar type, filters based on vertical apature.

        Keyword Parameters:
        points -- pixels from CFAR
        sonar -- sonar type (vertical or horizontal)

        Returns:
        pixels, x, y, range (in meters) and bearing in degrees
        """

        # convert to meters
        x = points[:, 1] - self.cols / 2.0
        x = 1 * ((x / float(self.cols / 2.0)) * (self.width / 2.0))
        y = (-1 * (points[:, 0] / float(self.rows)) * self.height) + self.height

        # check the sonar type
        if sonar == "vertical":
            x -= self.transformation
        elif sonar == "horizontal":
            pass
        else:
            rospy.loginfo("Incorrect sonar info in img2real function!")

        # get range and bearing
        r = np.sqrt(x ** 2 + y ** 2)
        b = np.degrees(np.arctan(x / y))

        # div vertical aperture by two
        deg = self.verticalAperture / 2.0

        # filter based on the vertical apature of the companion sonar
        x = x[(b > -deg) & (b < deg)]  # x meters
        y = y[(b > -deg) & (b < deg)]  # y meters
        r = r[(b > -deg) & (b < deg)]  # range meters
        u = points[:, 0][(b > -deg) & (b < deg)]  # u pixel coords
        v = points[:, 1][(b > -deg) & (b < deg)]  # v pixel coords
        b = b[(b > -deg) & (b < deg)]  # bearing in degrees

        return u, v, x, y, r, b

    def extractFeatures(self, img, detector, alg, threshold):
        # type: (np.ndarray, CFAR, str, float) -> np.ndarray
        """Function to take a raw greyscale sonar image and apply CFAR.

        Keyword Parameters:
        img -- raw greyscale sonar image
        detector -- detector object
        alg -- CFAR version to be used
        threshold -- CFAR thershold

        Returns:
        CFAR points as ndarray
        """

        if detector == "vertical":
            detector = self.detector_vertical
        elif detector == "horizontal":
            detector = self.detector_horizontal
        else:
            rospy.loginfo("Incorrect sonar info in extractFeatures function!")

        # get raw detections
        peaks = detector.detect(img, alg)

        # check against threhold
        peaks &= img > threshold

        # convert to cartisian
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)

        # compile points
        points = np.c_[np.nonzero(peaks)]

        # return a numpy array
        return np.array(points)

    def extractPatches(self, v, u, sonar, img, size, ravel=False):
        """A function to get the patch around all features.

        Keyword Arguments:
        v, u -- image pixel coords for features
        sonar -- which sonar (vertical or horizontal)
        img -- the raw sonar image in cartisian format
        size -- the size of the image patch sizexsize

        Returns:
        python list of size x size patches around each feature
        """

        # container for the image patches
        patches = []

        # create a padded image to make the kernel always inside the image
        paddedImg = cv2.copyMakeBorder(
            img, size, size, size, size, cv2.BORDER_CONSTANT, value=0
        )

        # loop over the set features
        for j, i in zip(v, u):

            j = j + size
            i = i + size

            # add the kernel to a patch list, if vertical sonar, rotate the patch along the way
            if sonar == "horizontal":
                patch = paddedImg[i - size : 1 + i + size, j - size : 1 + j + size]
                if patch.shape == (size * 2 + 1, size * 2 + 1):
                    if ravel:
                        patch = np.ravel(patch)
                    patches.append(patch)
            elif sonar == "vertical":
                patch = np.rot90(
                    paddedImg[i - size : 1 + i + size, j - size : 1 + j + size], -1
                )
                if patch.shape == (size * 2 + 1, size * 2 + 1):
                    if ravel:
                        patch = np.ravel(patch)
                    patches.append(patch)
            else:
                rospy.loginfo("Incorrect sonar info in extractPatches function!")

        return patches

    def matchFeatures_2(
        self,
        rangeHorizontal,  # type: float
        bearingHorizontal,  # type: float
        patchesHorizontal,  # type: np.ndarray
        rangeVertical,  # type: float
        xVertical,  # type: float
        patchesVertical,  # type: np.ndarray
    ):

        # FIX ME, make the image size a ros param
        # convert range (meters) to pixels
        rangeHorizontal_discret = np.round(600 * (rangeHorizontal / self.maxRange_horizontal))
        rangeVertical_discret = np.round(600 * (rangeVertical / self.maxRange_vertical))

        # call the cpp function to do the matching
        matches =  match_features_cpp(rangeHorizontal_discret,
                                    rangeHorizontal,
                                    bearingHorizontal,
                                    rangeVertical_discret,
                                    rangeVertical,
                                    xVertical,
                                    patchesHorizontal,
                                    patchesVertical)

        return matches, matches.shape != (0,5)

    def matchFeatures(
        self,
        rangeHorizontal,  # type: float
        bearingHorizontal,  # type: float
        xHorizontal,  # type: float
        yHorizontal,  # type: float
        patchesHorizontal,  # type: np.ndarray
        rangeVertical,  # type: float
        bearingVertical,  # type: float
        xVertical,  # type: float
        yVertical,  # type: float
        patchesVertical,  # type: np.ndarray
    ):
        # type: (...) -> np.ndarray
        """Perform feature matching on sub problems.

        Keyword Parameters:
        rangeHorizontal -- horizontal sonar range meas (meters)
        bearingHorizontal -- horizontal sonar bearing meas (degrees)
        xHorizontal, yHorizontal -- horizontal sonar meas (meters) in cartisian
        patchesHorizontal -- image patches from horizontal sonar meas

        rangeVertical -- vertical sonar range meas (meters)
        bearingVertical -- vertical sonar bearing meas (degrees)
        xVertical, yVertical -- vertical sonar meas (meters) in cartisian
        patchesVertical -- image patches from vertical sonar meas

        Returns:
        array of matched features
        """

        # FIX ME, make the image size a ros param
        # convert range (meters) to pixels
        rangeHorizontal_discret = np.round(
            600 * (rangeHorizontal / self.maxRange_horizontal)
        )
        rangeVertical_discret = np.round(600 * (rangeVertical / self.maxRange_vertical))

        # get the unique range options
        range_options_horizontal = np.sort(list(set(rangeHorizontal_discret)))
        range_options_vertical = np.sort(list(set(rangeVertical_discret)))

        # create some convient containers
        featuresHorizontal = np.column_stack((rangeHorizontal, bearingHorizontal))
        featuresVertical = np.column_stack((xVertical, rangeVertical))

        # container for matches format: [x,y,y,z,uncer]
        matches = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # loop over the range options in horizontal sonar
        for r1 in range_options_horizontal:

            # check range options in vertical sonar
            # for r2 in ver_range_options:
            if (r1 in range_options_vertical) == True:

                # get features at this range
                hor = featuresHorizontal[rangeHorizontal_discret == r1]
                vert = featuresVertical[rangeVertical_discret == r1]

                # get the patches at this range
                hor_kernel = patchesHorizontal[rangeHorizontal_discret == r1]
                vert_kernel = patchesVertical[rangeVertical_discret == r1]

                # guard the sub problem against noise
                if len(hor) > 2 and len(vert) > 2:

                    # check which subset is larger
                    if len(hor) > len(vert):

                        # init min error as a high value
                        error_min = sys.float_info.max
                        error_min_set = None

                        # log the costs
                        cost_feature_mtx = np.zeros((len(vert), 1))

                        # loop over the larger set
                        for i in range(0, len(hor)):

                            # scramble the horizontal
                            x, y = shuffle(hor, hor_kernel, random_state=i)

                            # slice the scrambled horziontal to match dimensions
                            x = x[: len(vert)]
                            y = y[: len(vert)]

                            # get the difference between the kernel and the hyp
                            val = np.array(abs(vert_kernel - y)).reshape(
                                vert_kernel.shape[0], vert_kernel.shape[1] ** 2
                            )

                            # get the cost on a per feature basis
                            cost_feature = np.sum(val, axis=1)

                            # log this cost
                            cost_feature_mtx = np.column_stack(
                                (cost_feature_mtx, cost_feature)
                            )

                            # get the cost on a total basis
                            error = np.sum(cost_feature)

                            # if error is lower record it
                            if error < error_min:

                                # record the new error
                                error_min = error

                                # recrod the new set
                                error_min_set = x

                                # record the patches
                                error_min_patches = y

                        # calculate feature uncertainty
                        min_feature_costs = np.sort(cost_feature_mtx)
                        # feature_uncertainty = (min_feature_costs[:,2] - min_feature_costs[:,1]) / np.sum(min_feature_costs, axis=1)
                        feature_uncertainty = (
                            min_feature_costs[:, 1] / min_feature_costs[:, 2]
                        )

                        # append to output
                        matches = np.row_stack(
                            (
                                matches,
                                np.column_stack(
                                    (error_min_set, vert, feature_uncertainty)
                                ),
                            )
                        )

                    else:

                        # init min error as a high value
                        error_min = sys.float_info.max
                        error_min_set = None

                        # log the errors
                        errors = []

                        # log the costs
                        cost_feature_mtx = np.zeros((len(hor), 1))

                        for i in range(0, len(vert)):

                            # scramble the horizontal
                            x, y = shuffle(vert, vert_kernel, random_state=i)

                            # slice the scrambled horziontal to match dimensions
                            x = x[: len(hor)]
                            y = y[: len(hor)]

                            # get the difference between the kernel and the hyp
                            val = np.array(abs(hor_kernel - y)).reshape(
                                hor_kernel.shape[0], hor_kernel.shape[1] ** 2
                            )

                            # get the cost on a per feature basis
                            cost_feature = np.sum(val, axis=1)

                            # log this cost
                            cost_feature_mtx = np.column_stack(
                                (cost_feature_mtx, cost_feature)
                            )

                            # get the cost on a total basis
                            error = np.sum(cost_feature)

                            # if error is record it
                            if error < error_min:

                                # record the new error
                                error_min = error

                                # recrod the new set
                                error_min_set = x

                                # record the patches
                                error_min_patches = y

                        # calculate feature uncertainty
                        min_feature_costs = np.sort(cost_feature_mtx)
                        # feature_uncertainty = (min_feature_costs[:,2] - min_feature_costs[:,1]) / np.sum(min_feature_costs, axis=1)
                        feature_uncertainty = (
                            min_feature_costs[:, 1] / min_feature_costs[:, 2]
                        )

                        # append to output
                        matches = np.row_stack(
                            (
                                matches,
                                np.column_stack(
                                    (hor, error_min_set, feature_uncertainty)
                                ),
                            )
                        )

        return matches, matches.shape != (5,)

    def callback(self, msgVertical, msgHorizontal):
        # type: (OculusPing, OculusPing) -> None
        """Ros callback for dual sonar system.

        Keyword Parameters:
        msgVertical -- vertical sonar msg
        msgHorizontal -- horizontal sonar msg
        """

        # generate the mapping from polar to cartisian
        self.generate_map_xy(msgHorizontal)

        # decode the compressed horizontal image
        imgHorizontal = np.fromstring(msgHorizontal.ping.data, np.uint8)
        imgHorizontal = np.array(cv2.imdecode(imgHorizontal, cv2.IMREAD_COLOR)).astype(
            np.uint8
        )
        imgHorizontal = cv2.cvtColor(imgHorizontal, cv2.COLOR_BGR2GRAY)

        # decode the compressed vertical image
        imgVertical = np.fromstring(msgVertical.ping.data, np.uint8)
        imgVertical = np.array(cv2.imdecode(imgVertical, cv2.IMREAD_COLOR)).astype(
            np.uint8
        )
        imgVertical = cv2.cvtColor(imgVertical, cv2.COLOR_BGR2GRAY)

        # denoise the horizontal image, consider adding this for the vertical image
        imgHorizontal = cv2.fastNlMeansDenoising(imgHorizontal, None, 10, 7, 21)

        # check size, images must be the same size
        if imgHorizontal.shape != imgVertical.shape:
            imgVertical = cv2.resize(
                imgVertical, (imgHorizontal.shape[1], imgHorizontal.shape[0])
            )

        # get some features using CFAR
        horizontalFeatures = self.extractFeatures(
            imgHorizontal, "horizontal", "SOCA", self.thresholdHorizontal
        )
        verticalFeatures = self.extractFeatures(
            imgVertical, "vertical", "SOCA", self.thresholdVertical
        )

        # remap the raw images into cartisian coords
        imgHorizontal = cv2.remap(
            imgHorizontal, self.map_x, self.map_y, cv2.INTER_LINEAR
        )
        imgVertical = cv2.remap(imgVertical, self.map_x, self.map_y, cv2.INTER_LINEAR)

        # convert the features to meters and degrees
        uh, vh, xh, yh, rh, bh = self.img2Real(horizontalFeatures, "horizontal")
        uv, vv, xv, yv, rv, bv = self.img2Real(verticalFeatures, "vertical")

        if self.method == "python":

            # get the image kernels, used to compare pixel similarity
            patches_horizontal = np.array(
                self.extractPatches(vh, uh, "horizontal", imgHorizontal, self.patchSize)
            )
            patches_vertical = np.array(
                self.extractPatches(vv, uv, "vertical", imgVertical, self.patchSize)
            )

            # perform some matching
            matches, match_status = self.matchFeatures(
                rh, bh, xh, yh, patches_horizontal, rv, bv, xv, yv, patches_vertical
            )

            # remove the first row of zeros, only in the python implmentation
            matches = np.delete(matches, 0, 0)
        
        elif self.method == "cpp":

            # get the image kernels, used to compare pixel similarity
            patches_horizontal = np.array(
                self.extractPatches(vh, uh, "horizontal", imgHorizontal, self.patchSize,True)
            )
            patches_vertical = np.array(
                self.extractPatches(vv, uv, "vertical", imgVertical, self.patchSize,True)
            )

            # perform some matching
            matches, match_status = self.matchFeatures_2(
                rh, bh, patches_horizontal, rv, xv, patches_vertical
            )

        # protect for no matches
        if match_status:

            # remove uncertain matches
            matches = matches[matches[:, 4] < self.uncertaintyMax]

            # solve the conversion to cartsiain coords from spherical
            elevationAngle = np.arccos(
                matches[:, 2] / matches[:, 0]
            )  # get the elevation angle
            bearingAngle = np.radians(matches[:, 1])  # convert back to radians
            rangeAvg = (
                matches[:, 0] + matches[:, 3]
            ) / 2.0  # average the range between the two matches
            x = (
                rangeAvg * np.cos(bearingAngle) * np.sin(elevationAngle)
            )  # convert to cartisian
            y = rangeAvg * np.sin(bearingAngle) * np.sin(elevationAngle)
            z = matches[:, 2]

            # assemble the point cloud for ROS, the order may be different based on your
            # coordinate frame
            points = np.column_stack((x, z, -y))

            # package the point cloud
            header = Header()
            header.frame_id = "base_link"
            header.stamp = (
                msgHorizontal.header.stamp
            )  # use input msg timestamp for better sync downstream
            laserCloudOut = pc2.create_cloud(header, self.laserFields, points)

            # publish the cloud
            self.cloudPublisher.publish(laserCloudOut)

        # there are no matches, publish a blank cloud for downstream time sync
        else:

            # package as numpy array
            points = np.column_stack(([], [], []))

            # package the point cloud
            header = Header()
            header.frame_id = "base_link"
            header.stamp = (
                msgHorizontal.header.stamp
            )  # use input msg timestamp for better sync downstream
            laserCloudOut = pc2.create_cloud(header, self.laserFields, points)

            # publish the cloud
            self.cloudPublisher.publish(laserCloudOut)
